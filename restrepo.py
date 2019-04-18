"""Python code associated with Restrop et al 2019

I think one of the first things we'll need to do with the Restrpo data is to decomplex it slightly. There
are a huge number of ITS2 type profiles which in many cases are differenciated by very small differences in DIV
abundances. I have been manually looking at the data and there are definitely discrete clusterings of types
that would do well to be considered together. We can then look within these clusters at the more detailed resolutions
further on in the analysis. Certainly the way to approach this analysis will be to work on a clade by clade basis.
I think I would like to start by performing a hierarchical clustering and dendogram. Once I have done the clustering
I will look at the clusters by eye to make sure that they make sense and that there are not some artefacts causing
issues.

I will use a class appraoch to hold the entire process of doing this analysis to try to do a better organisation.
One of the first steps will be getting the dataframe read in cleanly.

As I am working through this data analysis I will keep in mind that I should be able to swap out the inital
info output documents and still be able to run the analysis from scratch easily. This is because we are updating
the remote symportal to new code soon.
What is probably easiest to do is to operate on the delete version I have running on middlechild and then eventually
I can change out the documents from the new remote symportal version.
This way I can go ahead and implement the changes to UniFrac calculations on the local_dev branch and make use of this
for this analysis now. THis way I don't have to wait for the change to the new SymPortal to be complete before
proceeding with this analysis.

Order of analysis

3 - histogram_of_all_abundance_values
4 - make_dendogram
5 - create_profile_df_with_cutoff
6 - get_list_of_clade_col_type_uids_for_unifrac
7 - remake_dendogram

"""
import os
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np
import hierarchy_sp
import pickle

class RestrepoAnalysis:
    def __init__(self, base_input_dir, profile_rel_abund_ouput_path, profile_abs_abund_ouput_path,
                 seq_rel_abund_ouput_path, seq_abs_abund_ouput_path, clade_A_dist_path, clade_C_dist_path,
                 clade_D_dist_path, clade_A_dist_cct_specific_path=None, clade_C_dist_cct_specific_path=None,
                 clade_D_dist_cct_specific_path=None, ignore_cache=False):
        # Although we see clade F in the dataset this is minimal and so we will
        # tackle this sepeately to the analysis of the A, C and D.
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.clades = list('ACD')
        self.base_input_dir = base_input_dir
        self.ignore_cache=ignore_cache
        # Paths to raw info files
        self.profile_rel_abund_ouput_path = os.path.join(self.base_input_dir, profile_rel_abund_ouput_path)
        self.profile_abs_abund_ouput_path = os.path.join(self.base_input_dir, profile_abs_abund_ouput_path)
        self.seq_rel_abund_ouput_path = os.path.join(self.base_input_dir, seq_rel_abund_ouput_path)
        self.seq_abs_abund_ouput_path = os.path.join(self.base_input_dir, seq_abs_abund_ouput_path)
        self.clade_dist_path_dict = {
            'A' : os.path.join(self.base_input_dir, clade_A_dist_path),
            'C' : os.path.join(self.base_input_dir, clade_C_dist_path),
            'D' : os.path.join(self.base_input_dir, clade_D_dist_path)}

        # Paths to the cct specific distances
        self.clade_dist_cct_specific_path_dict = {
            'A': os.path.join(self.base_input_dir, clade_A_dist_cct_specific_path),
            'C': os.path.join(self.base_input_dir, clade_C_dist_cct_specific_path),
            'D': os.path.join(self.base_input_dir, clade_D_dist_cct_specific_path)
        }

        # Iterative attributes

        # cache implementation
        self.cache_dir = os.path.join(self.cwd, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.clade_dist_dict_p_path = os.path.join(self.cache_dir, 'clade_dist_df_dict.p')
        self.clade_dist_cct_specific_dict_p_path = os.path.join(self.cache_dir, 'clade_dist_cct_specific_dict.p')

        # Info containers
        self.smp_uid_to_name_dict = None
        self.prof_uid_to_local_abund_dict = None
        self.prof_uid_to_local_abund_dict_post_cutoff = {}
        self.prof_uid_to_global_abund_dict = None
        self.prof_uid_to_name_dict = None
        self.clade_dist_df_dict = {}
        self._populate_clade_dist_df_dict()
        self.clade_dist_cct_specific_df_dict = {}
        if self.clade_dist_cct_specific_path_dict['A']:
            # if we have the cct_speicifc distances
            self._populate_clade_dist_df_dict(cct_specific=True)
        self.profile_df  = None
        self._populate_profile_df()
        self.type_uid_to_name_dict = {}
        self.cutoff_abund = 0.06
        self.prof_df_cutoff = None

        # figures
        self.figure_dir = os.path.join(self.cwd, 'figures')
        os.makedirs(self.figure_dir, exist_ok=True)

        # output paths
        self.outputs_dir = os.path.join(self.cwd, 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.uid_pairs_for_ccts_path = os.path.join(self.outputs_dir, 'dss_at_uid_tups.tsv')

    def _populate_clade_dist_df_dict(self, cct_specific=False):
        """If cct_specific is set then we are making dfs for the distance matrices that are from the bespoke
        set of CladeCollectionTypes. If not set then it is the first set of distances that have come straight out
        of the SymPortal analysis with no prior processing. I have implemented a simple cache system."""
        if not self.ignore_cache:
            self.pop_clade_dist_df_dict_from_cache_or_make_new(cct_specific)
        else:
            self._pop_clade_dict_df_dict_from_scratch_and_pickle_out(cct_specific)

    def pop_clade_dist_df_dict_from_cache_or_make_new(self, cct_specific):
        try:
            if cct_specific:
                self.clade_dist_cct_specific_df_dict = pickle.load(
                    file=open(self.clade_dist_cct_specific_dict_p_path, 'rb'))
            else:
                self.clade_dist_df_dict = pickle.load(file=open(self.clade_dist_dict_p_path, 'rb'))
        except FileNotFoundError:
            self._pop_clade_dict_df_dict_from_scratch_and_pickle_out(cct_specific)

    def _pop_clade_dict_df_dict_from_scratch_and_pickle_out(self, cct_specific):
        self._pop_clade_dist_df_dict_from_scrath(cct_specific)
        pickle.dump(obj=self.clade_dist_df_dict, file=open(self.clade_dist_dict_p_path, 'wb'))

    def _pop_clade_dist_df_dict_from_scrath(self, cct_specific):
        if cct_specific:
            path_dict_to_use = self.clade_dist_cct_specific_path_dict
        else:
            path_dict_to_use = self.clade_dist_path_dict
        for clade in self.clades:
            with open(path_dict_to_use[clade], 'r') as f:
                clade_data = [out_line.split('\t') for out_line in [line.rstrip() for line in f][1:]]

            df = pd.DataFrame(clade_data)
            self.type_uid_to_name_dict = {int(uid): name for uid, name in zip(df[1], df[0])}
            df.drop(columns=0, inplace=True)
            df.set_index(keys=1, drop=True, inplace=True)
            df.index = df.index.astype('int')
            df.columns = df.index.values.tolist()
            if cct_specific:
                self.clade_dist_cct_specific_df_dict[clade] = df.astype(dtype='float')
            else:
                self.clade_dist_df_dict[clade] = df.astype(dtype='float')

    def _populate_profile_df(self):
        # read in df
        df = pd.read_csv(filepath_or_buffer=self.profile_rel_abund_ouput_path, sep='\t', header=None)
        # collect sample uid to name info
        self.smp_uid_to_name_dict = {uid: name for uid, name in zip(df[0][7:-12], df[1][7:-12])}
        # del smp name column
        df.drop(columns=1, inplace=True)
        # reset df col headers
        df.columns = range(len(list(df)))
        # Populate prof abund dicts
        self.prof_uid_to_local_abund_dict = {int(uid): int(abund) for uid, abund in zip(df.iloc[0,1:], df.iloc[4,1:])}
        self.prof_uid_to_global_abund_dict = {int(uid): int(abund) for uid, abund in zip(df.iloc[0,1:], df.iloc[5,1:])}
        self.prof_uid_to_name_dict = {int(uid): name for uid, name in zip(df.iloc[0,1:], df.iloc[6,1:])}
        # drop meta info except for prof uid
        # top
        df.drop(index=range(1,7), inplace=True)

        # bottom
        # get the index to crop from

        index_list = df.index.values.tolist()
        for i in range(len(index_list)):
            if 'Sequence accession' in str(df.iloc[i,0]):
                # then this is the first row to get rid of
                index_to_cut_from = i
                break

        df = df.iloc[:index_to_cut_from,]
        # get prof uids to make into the column headers
        headers = [int(a) for a in df.iloc[0,1:].values.tolist()]
        # drop the prof uids
        df.drop(index=0, inplace=True)
        # promote smp uids to index
        df.set_index(keys=0, drop=True, inplace=True)
        df.columns = headers
        df = df.astype(dtype='float')
        df.index = df.index.astype('int')
        self.profile_df = df

    def make_dendogram(self, cct_specific=False):
        """I have modified the scipi.cluster.hierarchy.dendrogram so that it can take a line thickness dictionary.
        It will use this dictionary to modify the thickness of the leaves on the dendrogram so that we can see which
        types were the most abundant.

        NB The documentation for the linkage() is misleading. It cannot take a redundant distance matrix.
        The matrix must first be condensed. Again, misleadingly, this can be done using the poorly named squareform()
        that does both condensed to redundant and vice versa.

        If the cct_specific distance are not available then I will make this a plot of the the non_cct_specific
        side by side with the cct_specific so that we can see if there are any significant differences.

        """

        if cct_specific:
            # draw dendrograms in pairs
            for clade in self.clades:
                fig, axarr = plt.subplots(1,2,figsize=(16, 12))

                # plot the non cutoff dendro first
                self._make_dendrogram_figure(
                    clade=clade, ax=axarr[0], dist_df=self.clade_dist_df_dict[clade],
                    local_abundance_dict = self.prof_uid_to_local_abund_dict)

                # then the cct specific
                self._make_dendrogram_figure(
                    clade=clade, ax=axarr[1], dist_df=self.clade_dist_cct_specific_df_dict[clade],
                    local_abundance_dict=self.prof_uid_to_local_abund_dict_post_cutoff)

                plt.tight_layout()
                plt.savefig(os.path.join(self.figure_dir, f'paired_dendogram_{clade}.png'))
                plt.savefig(os.path.join(self.figure_dir, f'paired_dendogram_{clade}.svg'))


        else:
            # draw a single dendrogram per clade
            for clade in self.clades:
                fig, ax = plt.subplots(figsize=(8, 12))
                self._make_dendrogram_figure(clade=clade, ax=ax, dist_df=self.clade_dist_df_dict[clade],
                                             local_abundance_dict=self.prof_uid_to_local_abund_dict)
                plt.tight_layout()


    def _make_dendrogram_figure(self, clade, ax, dist_df, local_abundance_dict):
        """Plot a dendrogram """
        condensed_dist = scipy.spatial.distance.squareform(dist_df)
        # this creates the linkage df that will be passed into the dendogram_sp function
        linkage = scipy.cluster.hierarchy.linkage(y=condensed_dist, optimal_ordering=True)
        thickness_dict = self._make_thickness_dict(clade, dist_df, local_abundance_dict)
        labels = self._make_labels_list(dist_df, local_abundance_dict)
        self._draw_one_dendrogram(ax, labels, linkage, thickness_dict)

    def _make_thickness_dict(self, clade, dist_df, local_abundance_dict):
        # generate the thickness dictionary. Lets work with line thicknesses of 1, 2, 3 and 4
        # assign according to which quartile
        max_abund = \
        sorted([value for uid, value in local_abundance_dict.items() if clade in self.prof_uid_to_name_dict[uid]],
               reverse=True)[0]
        thickness_dict = {}
        for uid in dist_df.index.values.tolist():
            abund = local_abundance_dict[uid]
            if abund < max_abund * 0.1:
                thickness_dict[f'{self.prof_uid_to_name_dict[uid]} ({local_abundance_dict[uid]})'] = 1
            elif abund < max_abund * 0.2:
                thickness_dict[f'{self.prof_uid_to_name_dict[uid]} ({local_abundance_dict[uid]})'] = 2
            elif abund < max_abund * 0.3:
                thickness_dict[f'{self.prof_uid_to_name_dict[uid]} ({local_abundance_dict[uid]})'] = 3
            elif abund < max_abund * 0.5:
                thickness_dict[f'{self.prof_uid_to_name_dict[uid]} ({local_abundance_dict[uid]})'] = 4
            elif abund < max_abund * 0.7:
                thickness_dict[f'{self.prof_uid_to_name_dict[uid]} ({local_abundance_dict[uid]})'] = 5
            else:
                thickness_dict[f'{self.prof_uid_to_name_dict[uid]} ({local_abundance_dict[uid]})'] = 6
        return thickness_dict

    def _make_labels_list(self, dist_df, local_abundance_dict):
        labels = [f'{self.prof_uid_to_name_dict[uid]} ({local_abundance_dict[uid]})' for uid in dist_df.index.values.tolist()]
        return labels

    def _draw_one_dendrogram(self, ax, labels, linkage, thickness_dict):
        den = hierarchy_sp.dendrogram_sp(linkage, labels=labels, ax=ax,
                                         node_to_thickness_dict=thickness_dict,
                                         default_line_thickness=0.5, leaf_rotation=90)

    def histogram_of_all_abundance_values(self):
        """ Plot a histogram of all of the type-rel-abund pairings so that we can assess whether there is a sensible
        cutoff for the abundances.

        Question: Are there a disproportional number of profile instances being found that are of a very low abundance
        Answer: Yes, there are a lot more very low level profiles
        Conclusion: For the higher scale conclusions we're trying to draw let's discount these very low associations
        by implementing a threshold cutoff. We will create a profile_df that has the associations lower than 0.06
        removed. We will look to see how many profiles this discounts from the analysis.
        """
        # Linerize the values in the df for passing to the hist
        f, ax_arr = plt.subplots(1, 2, figsize=(10, 5))
        values = []
        for index, row in self.profile_df.iterrows():
            values.extend(row.iloc[row.nonzero()].values.tolist())
        temp_series = pd.Series(values)

        hist = temp_series.hist(bins=100, ax=ax_arr[0])

        # Now do the same plot with the 0.06 cutoff applied
        cutoff = 0.06
        cut_off_values = [a for a in values if a > cutoff]
        temp_series = pd.Series(cut_off_values)
        hist = temp_series.hist(bins=100, ax=ax_arr[1])

        f.suptitle('Relative abundance of ITS2 type profile in sample', fontsize=14, x=0.5, y=0.05)
        ax_arr[0].set_ylabel('Frequency of observation', fontsize=14)
        plt.savefig(os.path.join(self.figure_dir, 'hist.png'), dpi=1200)
        plt.savefig(os.path.join(self.figure_dir, 'hist.png'), dpi=1200)

    def create_profile_df_with_cutoff(self):
        """Creates a new df from the old df that has all of the values below the cutoff_abundance threshold
        made to 0. We will also calculate a new prof_uid_to_local_abund_dict_post_cutoff dictionary.
        """

        num_profs_pre_cutoff = len(list(self.profile_df))
        print(f'There are {num_profs_pre_cutoff} ITS2 type profiles before applying cutoff of {self.cutoff_abund}')
        # make new df from copy of old df
        self.prof_df_cutoff = self.profile_df.copy()
        # change values below cutoff to 0
        self.prof_df_cutoff = self.prof_df_cutoff.mask(cond=self.prof_df_cutoff < self.cutoff_abund, other=0)
        # now drop columns with 0

        # now check to see if there are any type profiles that no longer have associations
        # https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
        self.prof_df_cutoff = self.prof_df_cutoff.loc[:, (self.prof_df_cutoff != 0).any(axis=0)]
        num_profs_post_cutoff = len(list(self.prof_df_cutoff))
        print(f'There are {num_profs_post_cutoff} after.')
        num_profs_removed = num_profs_pre_cutoff - num_profs_post_cutoff
        print(f'{num_profs_removed} ITS2 type profiles have been removed from the dataframe.')
        # get list of names of profiles removed due to cutoff
        profs_removed = [self.prof_uid_to_name_dict[uid] for uid in
                         list(self.profile_df) if
                         uid not in list(self.prof_df_cutoff)]
        print('These profiles were:')
        for prof in profs_removed:
            print(prof)

        # calculate how many unique DataSetSample to ITS2 type profile associations there are.
        num_associations_pre_cutoff = len(list(self.profile_df[self.profile_df > 0].stack().index))
        num_associations_post_cutoff = len(list(self.prof_df_cutoff[self.prof_df_cutoff > 0].stack().index))
        print(f'The number of unique DataSetSample to ITS2 type profile associations was {num_associations_pre_cutoff}.')
        print(f'The number of unique DataSetSample to ITS2 type profile associations '
              f'after cutoff is {num_associations_post_cutoff}')

        # now populate the new prof_uid_to_local_abund_dict_post_cutoff dictionary
        for i in list(self.prof_df_cutoff):  # for each column of the df
            temp_series = self.prof_df_cutoff[i]
            local_count = len(temp_series[temp_series > 0].index.values.tolist())
            self.prof_uid_to_local_abund_dict_post_cutoff[i] = local_count


    def get_list_of_clade_col_type_uids_for_unifrac(self):
        """ This is code for getting tuples of (DataSetSample uid, AnalysisType uid).
        These tuples can then be used to get a list of CladeCollectionType uids from the SymPortal terminal.
        This list of clade collection type uids can then be fed in to the new distance outputs that we are writing
        so that we can get between type distances calculated using just the sample/analysis type associations
        in our cutoff abundance count table.
        Once I have written the code in the SymPortal terminal I will put it in here as a comment."""

        # CODE USED IN THE SYMPORTAL SHELL TO GET THE CladeCollectionType IDS
        # from dbApp.models import *
        # with open('dss_at_uid_tups.tsv', 'r') as f:
        #     tup_list = [line.rstrip() for line in f]
        # tup_list = [(line.split('\t')[0], line.split('\t')[1]) for line in tup_list]
        # cct_id_list = []
        # for tup in tup_list:
        #     at = AnalysisType.objects.get(id=tup[1])
        #     dss = DataSetSample.objects.get(id=tup[0])
        #     cc = CladeCollection.objects.get(data_set_sample_from=dss, clade=at.clade)
        #     cct = CladeCollectionType.objects.get(analysis_type_of=at, clade_collection_found_in=cc)
        #     cct_id_list.append(cct.id)


        # from the code here we can get a list that contains tuples of DataSetSample uids to AnalysisType uid for the
        # sample type pairings that we are interested in (i.e. the non-zeros in the cutoff df). We can then use these
        # ids to look up the CladeCollectionTypes we are interested in, get the uids of these, and pass these
        # into the distance functions of SymPortal that we are going to make.
        # we should make seperate outputs for bray vs unifrac, unifrac sqrt trans formed and not.

        # https://stackoverflow.com/questions/26854091/getting-index-column-pairs-for-true-elements-of-a-boolean-dataframe-in-pandas
        index_column_tups = list(self.prof_df_cutoff[self.prof_df_cutoff > 0].stack().index)
        with open(self.uid_pairs_for_ccts_path, 'w') as f:
            for tup in index_column_tups:
                f.write(f'{tup[0]}\t{tup[1]}\n')
        print(f'A list of tuples has been output to {self.uid_pairs_for_ccts_path} that represent the UID of paired '
              f'DataSetSample and AnalysisType objects found in the filtered dataframe. '
              f'From these you can get the CladeCollectionTypes. This can then be fed into the distance output function'
              f'to calculate ITS2 type profile distances based only on these pairings.')

        # the list of output CladeCollectionType uids is at:
        # /Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/cc_id_list

        apples = 'asdf'




if __name__ == "__main__":
    rest_analysis = RestrepoAnalysis(
        base_input_dir = os.path.join(
            '/Users', 'humebc', 'Google_Drive', 'projects', 'alejandro_et_al_2018',
            'resources', 'sp_outputs_20190417', '2019-04-17_07-14-49.317290'),
        profile_rel_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.profiles.relative.txt',
        profile_abs_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.profiles.absolute.txt',
        seq_rel_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.seqs.relative.txt',
        seq_abs_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.seqs.absolute.txt',
        clade_A_dist_path=os.path.join(
            'between_profile_distances', 'A',
            '2019-04-17_07-14-49.317290.bray_curtis_within_clade_profile_distances_A.dist'),
        clade_C_dist_path=os.path.join(
            'between_profile_distances', 'C',
            '2019-04-17_07-14-49.317290.bray_curtis_within_clade_profile_distances_C.dist'),
        clade_D_dist_path=os.path.join(
            'between_profile_distances', 'D',
            '2019-04-17_07-14-49.317290.bray_curtis_within_clade_profile_distances_D.dist'),
        clade_A_dist_cct_specific_path=os.path.join(
            'cct_specific_between_profile_distances','2019-04-16_08-37-52_564623', 'between_profiles','A',
            '2019-04-16_08-37-52.564623.bray_curtis_within_clade_profile_distances_A.dist'),
        clade_C_dist_cct_specific_path=os.path.join(
            'cct_specific_between_profile_distances', '2019-04-16_08-37-52_564623','between_profiles','C',
            '2019-04-16_08-37-52.564623.bray_curtis_within_clade_profile_distances_C.dist'),
        clade_D_dist_cct_specific_path=os.path.join(
            'cct_specific_between_profile_distances', '2019-04-16_08-37-52_564623','between_profiles','D',
            '2019-04-16_08-37-52.564623.bray_curtis_within_clade_profile_distances_D.dist'),
        ignore_cache=True)
    rest_analysis.create_profile_df_with_cutoff()
    rest_analysis.make_dendogram(cct_specific=True)
    rest_analysis.get_list_of_clade_col_type_uids_for_unifrac()
    rest_analysis.histogram_of_all_abundance_values()
