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
8 - make meta data df

TODO list:
plot up the dendogram with the meta information below
Assess correlations at differing levels of collapse
Assess why the UniFrac distance approximation is not working so well

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
import matplotlib.gridspec as gridspec
from matplotlib import collections, patches
from collections import defaultdict, Counter
import skbio.diversity.alpha
import skbio.stats.distance
import sys


class RestrepoAnalysis:
    def __init__(self, base_input_dir, profile_rel_abund_ouput_path, profile_abs_abund_ouput_path,
                 seq_rel_abund_ouput_path, seq_abs_abund_ouput_path,
                 clade_A_profile_dist_path, clade_C_profile_dist_path, clade_D_profile_dist_path,
                 clade_A_smpl_dist_path, clade_C_smpl_dist_path, clade_D_smpl_dist_path,
                 clade_A__profile_dist_cct_specific_path=None, clade_C_profile_dist_cct_specific_path=None,
                 clade_D__profile_dist_cct_specific_path=None, ignore_cache=False, meta_data_input_path=None, cutoff_abund=None):
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
        # Paths to the standard output profile distance files
        self.profile_clade_dist_path_dict = {
            'A' : os.path.join(self.base_input_dir, clade_A_profile_dist_path),
            'C' : os.path.join(self.base_input_dir, clade_C_profile_dist_path),
            'D' : os.path.join(self.base_input_dir, clade_D_profile_dist_path)}

        # Paths to the cct specific distances
        self.profile_clade_dist_cct_specific_path_dict = {
            'A': os.path.join(self.base_input_dir, clade_A__profile_dist_cct_specific_path),
            'C': os.path.join(self.base_input_dir, clade_C_profile_dist_cct_specific_path),
            'D': os.path.join(self.base_input_dir, clade_D__profile_dist_cct_specific_path)
        }

        # Paths to the smpl distance (bray curtis sqrt transformed abundance)
        self.sample_clade_dist_path_dict = {
            'A': os.path.join(self.base_input_dir, clade_A_smpl_dist_path),
            'C': os.path.join(self.base_input_dir, clade_C_smpl_dist_path),
            'D': os.path.join(self.base_input_dir, clade_D_smpl_dist_path)
        }

        # Iterative attributes

        # cache implementation
        self.cache_dir = os.path.join(self.cwd, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.profile_clade_dist_dict_p_path = os.path.join(self.cache_dir, 'clade_dist_df_dict.p')
        self.profile_clade_dist_cct_specific_dict_p_path = os.path.join(self.cache_dir, 'clade_dist_cct_specific_dict.p')
        self.sample_clade_dist_dict_p_path = os.path.join(self.cache_dir, 'sample_clade_dist_df_dict.p')
        # Info containers
        self.smp_uid_to_name_dict = None
        self.smp_name_to_uid_dict = None
        self.prof_uid_to_local_abund_dict = None
        self.prof_uid_to_local_abund_dict_post_cutoff = {}
        self.prof_uid_to_global_abund_dict = None
        self.prof_uid_to_name_dict = None
        self.prof_name_to_uid_dict = None
        self.profile_clade_dist_df_dict = {}
        self._populate_clade_dist_df_dict()
        self.profile_clade_dist_cct_specific_df_dict = {}
        if self.profile_clade_dist_cct_specific_path_dict['A']:
            # if we have the cct_speicifc distances
            self._populate_clade_dist_df_dict(cct_specific=True)
        self.sample_clade_dist_df_dict = {}
        self._populate_clade_dist_df_dict(smp_dist=True)
        self.profile_df  = None
        self._populate_profile_df()
        self.type_uid_to_name_dict = {}
        if cutoff_abund is not None:
            self.cutoff_abund = cutoff_abund
            self.prof_df_cutoff = None
            self.create_profile_df_with_cutoff()
        else:
            self.cutoff_abund = cutoff_abund
            self.prof_df_cutoff = None
        # metadata_info_df
        if meta_data_input_path is not None:
            self.metadata_info_df = self._init_metadata_info_df(meta_data_input_path)
        else:
            self.metadata_info_df = None

        # figures
        self.figure_dir = os.path.join(self.cwd, 'figures')
        os.makedirs(self.figure_dir, exist_ok=True)

        # output paths
        self.outputs_dir = os.path.join(self.cwd, 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.uid_pairs_for_ccts_path = os.path.join(self.outputs_dir, 'dss_at_uid_tups.tsv')

    def _init_metadata_info_df(self, meta_info_path):
        """The matching of names between the SP output and the meta info that Alejandro was working from was causing us
        some issues. There were 10 samples names in the meta info that were not matching up with the SP sample names.

        The following names were not finding matches or found 2 matches:
        New name Q15G6 not found in SP output 0
        New name Q15G7 not found in SP output 0
        New name SN15G10 not found in SP output 0
        New name SN15G6 not found in SP output 0
        New name SN15G7 not found in SP output 0
        New name SN15G8 not found in SP output 0
        New name SN15G9 not found in SP output 0
        New name FS15SE8 found in SP output twice 2
        New name SN15G2 not found in SP output 0
        New name T1PC4 not found in SP output 0

        It looks like some of the 15 names have been called 16 in the SP output due to the fact that this is the
        name given to the sequencing files. Obviously 15 is the correct name as this related to the depth that the
        sample was taken at. To link these old names to the new names I will manually append the correct associations
        between SP output name and meta info name to the new_name_to_old_name_dict. (annoying but necessary).

        Given that the SP sample names are derived from the fastq files and that the

        meta_info_name  sp_output_name
        Q15G6           Q16G6
        Q15G7           Q16G7
        SN15G10         SN16G10
        SN15G6         SN16G6
        SN15G7         SN16G7
        SN15G8         SN16G8
        SN15G9         SN16G9
        FS15SE8         FS15SE8(702)
        SN15G2          SN25G2
        T1PC4           TIPC4

        NB that FS15SE8 matched two samples in the SP output. I associated this sample to the SP output sample
        that most closely related samples FS15SE6 and FS15SE10 that were taken from the same reef, depth, species
        and season.

        NB we are left over with one extra sample in the SP output as there were 605 samples in the SP output
        but only 604 samples in the meta info. This appears to be due to the duplicate FS15SE8 sample.

        We should therefore probably delete this sample from the profiles and smpls df.
        """
        meta_info_df = pd.DataFrame.from_csv(meta_info_path)
        # The same names in the meta info are different from those in the SymPortal output.
        # parse through the meta info names and make sure that they are found in only one of the SP output names
        # then, we can simply modify the sample uid to name dictionary (or better make a new one)
        new_name_to_old_name_dict = {}
        old_to_search = list(self.smp_uid_to_name_dict.values())
        len_new = len(meta_info_df.index.values.tolist())
        len_old = len(old_to_search)
        for new_name in meta_info_df.index.values.tolist():
            if new_name in ['Q15G6', 'Q15G7', 'SN15G10', 'SN15G6', 'SN15G7', 'SN15G8', 'SN15G9', 'FS15SE8', 'SN15G2', 'T1PC4']:
                self._add_new_name_to_old_name_entry_manually(new_name, new_name_to_old_name_dict, old_to_search)
            else:
                count = 0
                for old_name in old_to_search:
                    if new_name in old_name:
                        old_name_match = old_name
                        count += 1
                if count != 1:
                    print(f'New name {new_name} not found in SP output {count}')
                else:
                    new_name_to_old_name_dict[new_name] = old_name_match
                    old_to_search.remove(old_name_match)

        # delete 'FS15SE8_FS15SE8_N705-S508' from the df
        for uid, name in self.smp_uid_to_name_dict.items():
            if name == 'FS15SE8_FS15SE8_N705-S508':
                self.profile_df.drop(index=uid, inplace=True)
                if self.prof_df_cutoff is not None:
                    self.prof_df_cutoff.drop(index=uid, inplace=True)
                if self.sample_clade_dist_df_dict:
                    for clade in self.clades:
                        if uid in self.sample_clade_dist_df_dict[clade].index.values.tolist():
                            self.sample_clade_dist_df_dict[clade].drop(index=uid, inplace=True)
                            self.sample_clade_dist_df_dict[clade].drop(columns=uid, inplace=True)

                break

        # now that we have a conversion from new_name to old name, we can use this to look up the uid of the
        # mata info sample names in relation to the SP outputs. And use these uids as index rather than the meta info
        # names
        new_uid_index = []
        for new_name in meta_info_df.index.values.tolist():
            new_uid_index.append(int(self.smp_name_to_uid_dict[new_name_to_old_name_dict[new_name]]))

        meta_info_df.index = new_uid_index
        meta_info_df.columns = ['reef', 'reef_type', 'depth', 'species', 'season']
        meta_info_df = meta_info_df[['species', 'reef', 'reef_type', 'depth', 'season']]

        return meta_info_df


    def _add_new_name_to_old_name_entry_manually(self, new_name, new_name_to_old_name_dict, old_to_search):
        if new_name == 'Q15G6':
            new_name_to_old_name_dict[new_name] = 'Q16G6_Q16G6_N711-S506'
            old_to_search.remove('Q16G6_Q16G6_N711-S506')
        elif new_name == 'Q15G7':
            new_name_to_old_name_dict[new_name] = 'Q16G7_Q16G7_N712-S506'
            old_to_search.remove('Q16G7_Q16G7_N712-S506')
        elif new_name == 'SN15G10':
            new_name_to_old_name_dict[new_name] = 'SN16G10_SN16G10_N712-S505'
            old_to_search.remove('SN16G10_SN16G10_N712-S505')
        elif new_name == 'SN15G6':
            new_name_to_old_name_dict[new_name] = 'SN16G6_SN16G6_N708-S505'
            old_to_search.remove('SN16G6_SN16G6_N708-S505')
        elif new_name == 'SN15G7':
            new_name_to_old_name_dict[new_name] = 'SN16G7_SN16G7_N709-S505'
            old_to_search.remove('SN16G7_SN16G7_N709-S505')
        elif new_name == 'SN15G8':
            new_name_to_old_name_dict[new_name] = 'SN16G8_SN16G8_N710-S505'
            old_to_search.remove('SN16G8_SN16G8_N710-S505')
        elif new_name == 'SN15G9':
            new_name_to_old_name_dict[new_name] = 'SN16G9_SN16G9_N711-S505'
            old_to_search.remove('SN16G9_SN16G9_N711-S505')
        elif new_name == 'FS15SE8':
            new_name_to_old_name_dict[new_name] = 'FS15SE8_FS15SE8_N702-S503'
            old_to_search.remove('FS15SE8_FS15SE8_N702-S503')
        elif new_name == 'SN15G2':
            new_name_to_old_name_dict[new_name] = 'M-17_3688_SN25G2'
            old_to_search.remove('M-17_3688_SN25G2')
        elif new_name == 'T1PC4':
            new_name_to_old_name_dict[new_name] = 'M-17_3697_TIPC4'
            old_to_search.remove('M-17_3697_TIPC4')

    def _populate_clade_dist_df_dict(self, cct_specific=False, smp_dist=False):
        """If cct_specific is set then we are making dfs for the distance matrices that are from the bespoke
        set of CladeCollectionTypes. If not set then it is the first set of distances that have come straight out
        of the SymPortal analysis with no prior processing. I have implemented a simple cache system."""
        if not self.ignore_cache:
            self.pop_clade_dist_df_dict_from_cache_or_make_new(cct_specific, smp_dist)
        else:
            self._pop_clade_dict_df_dict_from_scratch_and_pickle_out(cct_specific, smp_dist)

    def pop_clade_dist_df_dict_from_cache_or_make_new(self, cct_specific, smp_dist):
        try:
            if smp_dist:
                self.sample_clade_dist_df_dict = pickle.load(
                    file=open(self.sample_clade_dist_dict_p_path, 'rb'))
            elif cct_specific:
                self.profile_clade_dist_cct_specific_df_dict = pickle.load(
                    file=open(self.profile_clade_dist_cct_specific_dict_p_path, 'rb'))
            else:
                self.profile_clade_dist_df_dict = pickle.load(file=open(self.profile_clade_dist_dict_p_path, 'rb'))
        except FileNotFoundError:
            self._pop_clade_dict_df_dict_from_scratch_and_pickle_out(cct_specific, smp_dist)

    def _pop_clade_dict_df_dict_from_scratch_and_pickle_out(self, cct_specific, smp_dist):
        self._pop_clade_dist_df_dict_from_scrath(cct_specific, smp_dist)

    def _pop_clade_dist_df_dict_from_scrath(self, cct_specific, smp_dist):
        if smp_dist:
            path_dict_to_use = self.sample_clade_dist_path_dict
        elif cct_specific:
            path_dict_to_use = self.profile_clade_dist_cct_specific_path_dict
        else:
            path_dict_to_use = self.profile_clade_dist_path_dict
        for clade in self.clades:
            with open(path_dict_to_use[clade], 'r') as f:
                clade_data = [out_line.split('\t') for out_line in [line.rstrip() for line in f][1:]]

            df = pd.DataFrame(clade_data)

            if not smp_dist:
                self.type_uid_to_name_dict = {int(uid): name for uid, name in zip(df[1], df[0])}

            df.drop(columns=0, inplace=True)
            df.set_index(keys=1, drop=True, inplace=True)
            df.index = df.index.astype('int')
            df.columns = df.index.values.tolist()

            if smp_dist:
                self.sample_clade_dist_df_dict[clade] = df.astype(dtype='float')
            elif cct_specific:
                self.profile_clade_dist_cct_specific_df_dict[clade] = df.astype(dtype='float')
            else:
                self.profile_clade_dist_df_dict[clade] = df.astype(dtype='float')

        if smp_dist:
            pickle.dump(obj=self.sample_clade_dist_df_dict,
                        file=open(self.sample_clade_dist_dict_p_path, 'wb'))
        elif cct_specific:
            pickle.dump(obj=self.profile_clade_dist_cct_specific_df_dict,
                        file=open(self.profile_clade_dist_cct_specific_dict_p_path, 'wb'))
        else:
            pickle.dump(obj=self.profile_clade_dist_df_dict, file=open(self.profile_clade_dist_dict_p_path, 'wb'))

    def _populate_profile_df(self):
        # read in df
        df = pd.read_csv(filepath_or_buffer=self.profile_rel_abund_ouput_path, sep='\t', header=None)
        # collect sample uid to name info
        index_list = df.index.values.tolist()
        for i in range(len(index_list)):
            if 'Sequence accession' in str(df.iloc[i, 0]):
                # then this is the first row to get rid of
                index_to_cut_from = i
                break
        self.smp_uid_to_name_dict = {int(uid): name for uid, name in zip(df[0][7:index_to_cut_from], df[1][7:index_to_cut_from])}
        self.smp_name_to_uid_dict = {name: uid for uid, name in self.smp_uid_to_name_dict.items()}
        # del smp name column
        df.drop(columns=1, inplace=True)
        # reset df col headers
        df.columns = range(len(list(df)))
        # Populate prof abund dicts
        self.prof_uid_to_local_abund_dict = {int(uid): int(abund) for uid, abund in zip(df.iloc[0,1:], df.iloc[4,1:])}
        self.prof_uid_to_global_abund_dict = {int(uid): int(abund) for uid, abund in zip(df.iloc[0,1:], df.iloc[5,1:])}
        self.prof_uid_to_name_dict = {int(uid): name for uid, name in zip(df.iloc[0,1:], df.iloc[6,1:])}
        self.prof_name_to_uid_dict = {name: uid for uid, name in self.prof_uid_to_name_dict.items()}
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

    def plot_pcoa_of_cladal(self):
        # Do a quick check to see if the samples in the distance matrices are matching up with the samples
        # from the

        from skbio.stats.ordination import pcoa
        """ Here we will plot a series of PCoA for the sample distances and we will colour
        them according to the meta info"""
        fig = plt.figure(figsize=(15, 6))
        # required for getting the bbox of the text annotations
        fig.canvas.draw()
        gs = gridspec.GridSpec(3, 5)
        ax_arr = [[] for _ in range(3)]
        meta_info_categories = list(self.metadata_info_df)
        for j in range(len(self.clades)):  # for each clade
            for i in range(len(meta_info_categories)):  # for each meta info category
                temp_ax = plt.subplot(gs[j:j+1, i:i+1])
                ax_arr[j].append(temp_ax)
                temp_ax.set_xticks([])
                temp_ax.set_yticks([])
                temp_ax.set_facecolor('grey')
                if j ==0: # if this is subplot in top row
                    temp_ax.set_title(f'{meta_info_categories[i]}')
                if i == 0: # if this is subplot in first column
                    temp_ax.set_ylabel(f'{self.clades[j]}')
                if j == 2: # if last row
                    temp_ax.set_xlabel('PC1')
                if i == 4: # if last column
                    ax2 = temp_ax.twinx()
                    ax2.set_yticks([])
                    ax2.set_ylabel('PC2')

        for j in range(len(self.clades)):  # for each clade
            # We need to compute the pcoa coords for each clade. These will be the points plotted in the
            # scatter for each of the different meta info for each clade
            sample_clade_dist_df = self.sample_clade_dist_df_dict[self.clades[j]]
            pcoa_output = pcoa(sample_clade_dist_df)
            this = sum(pcoa_output.eigvals)

            for i in range(len(meta_info_categories)):
                # Here is where we need to look at each sample and colour according to the meta categories

                color_dict = {
                    'G': '#98FB98', 'GX': '#F0E68C', 'M': '#DDA0DD', 'P': '#8B008B',
                    'PC': '#00BFFF', 'SE': '#0000CD', 'ST': '#D2691E', 1: '#CAE1FF', 15: '#2E37FE', 30: '#000080',
                    'Summer': '#FF0000', 'Winter': '#00BFFF','Inshore': '#FF0000',
                    'Midshelf': '#FFFF00', 'Offshore': '#008000', 'Al Fahal':'#98FB98', 'Abu Madafi':'#F0E68C',
                    'Qita al Kirsh':'#DDA0DD', 'Shib Nazar': '#8B008B', 'Tahla':'#00BFFF', 'Fsar':'#0000CD'}

                species_category_list = ['M', 'G', 'GX', 'P', 'PC', 'SE', 'ST']
                species_category_labels = ['M. dichotoma', 'G. planulata', 'G. fascicularis', 'Porites spp.',
                                   'P. verrucosa', 'S. hystrix', 'S. pistillata']
                color_list = []
                for smp_uid in list(sample_clade_dist_df):
                    color_list.append(color_dict[self.metadata_info_df.loc[smp_uid, meta_info_categories[i]]])

                ax_arr[j][i].scatter(x=pcoa_output.samples['PC1'], y=pcoa_output.samples['PC2'], marker='.', color=color_list, s=16)
                spples = 'asdf'
        apples = 'asdf'
        plt.savefig(os.path.join(self.figure_dir, 'sample_annotated_ordination_no_legend.png'))
    def make_dendrogram_with_meta_all_clades_sample_dists(self):
        """ We will produce a similar figure to the one that we have already produced for the types."""
        raise NotImplementedError

    def make_sample_balance_figure(self):
        class BalanceFigureMap:
            def __init__(self, parent):
                # self.fig = plt.figure(figsize=(12, 12))
                self.parent = parent
                self.outer_gs = gridspec.GridSpec(2,2)
                self.depths = [1, 15, 30]
                self.seasons = ['winter', 'summer']
                self.reefs = ['Fsar', 'Tahla', 'Qita al Kirsh', 'Al Fahal', 'Shib Nazar', 'Abu Madafi']
                self.reef_types = ['onshore', 'midshore', 'offshore']
                self.depth_height = 1
                self.depth_width = 6
                self.reef_height = 3 * self.depth_height
                self.reef_width = 1
                self.reef_type_height = 2 * self.reef_height
                self.reef_type_width = 1
                self.inner_height = len(self.depths) * len(self.reefs) * 1 / 3 * self.depth_height
                self.inner_width = self.reef_width + self.reef_type_width + (len(self.seasons) * self.depth_width)


            def setup_plotting_space(self):
                import geopandas as gpd
                for i in range(4): # for each of the major gridspaces
                    if i == 0: # then plot map
                        # basemap is screwed.
                        # cartopy is difficult to get working too
                        # https://github.com/conda-forge/cartopy-feedstock/issues/36
                        # had to downgrade shapely to 1.5.17
                        import cartopy.crs as ccrs
                        import cartopy
                        ccrs.PlateCarree()
                        # plt.axes(projection=ccrs.PlateCarree())
                        plt.figure()
                        ax = plt.axes(projection=ccrs.PlateCarree())
                        ax.coastlines()

        blfm = BalanceFigureMap(parent=self)
        blfm.setup_plotting_space()



        # map_width = total_width/2
        # map_height = total_height/2
        #
        # gs = gridspec.GridSpec(total_height, total_width)
        # map_top_left = True  # if False then bottom right
        #
        #
        #
        # species = ['M', 'G', 'GX', 'P', 'PC', 'SE', 'ST']
        # species_full = ['M. dichotoma', 'G. planulata', 'G. fascicularis', 'Porites spp.',
        #                            'P. verrucosa', 'S. hystrix', 'S. pistillata']


        # # required for getting the bbox of the text annotations
        # fig.canvas.draw()
        #
        #
        # reef_type_axarr = []
        # for i in range(len(reef_types)):
        #     gs_start_reef_type = i*reef_type_height
        #     gs_stop_reef_type = (i+1)*reef_type_height
        #
        #     reef_type_axarr.append(plt.subplot(gs[gs_start_reef_type:gs_stop_reef_type, 0:reef_type_width]))
        #
        # reef_axarr = []
        # for i in range(len(reefs)):
        #     gs_start_reef = i*reef_height
        #     gs_stop_reef = (i+1)*reef_height
        #     reef_axarr.append(plt.subplot(gs[gs_start_reef:gs_stop_reef, reef_type_width:reef_type_width+reef_width]))
        #
        # winter_axarr = []
        # for i in range(len(depths) * len(reefs)):
        #     gs_start_winter = i * depth_height
        #     gs_stop_winter = (i+1) * depth_height
        #     width_start_winter = reef_type_width+reef_width
        #     width_stop_winter = reef_type_width+reef_width+depth_width
        #     winter_axarr.append(plt.subplot(gs[gs_start_winter:gs_stop_winter, width_start_winter:width_stop_winter]))
        #
        # summer_axarr = []
        # for i in range(len(depths) * len(reefs)):
        #     gs_start_summer = i * depth_height
        #     gs_stop_summer = (i + 1) * depth_height
        #     width_start_summer = reef_type_width + reef_width + depth_width
        #     width_stop_summer = reef_type_width + reef_width + depth_width + depth_width
        #     summer_axarr.append(plt.subplot(gs[gs_start_summer:gs_stop_summer, width_start_summer:width_stop_summer]))

        apples = 'asdf'

    def make_dendrogram_with_meta_all_clades(self):
        """This function will make a figure that has a dendrogram at the top, the labels under that, then
        under this it will have data that link the metainfo to each of the types found.
        NB, I have modified the returned dictionary from hierarchy_sp.dendrogram_sp so that it contains the
        tick_to_profile_name_dict that we can use to easily associate the label that should be plotted in the
        labels plot.

        NB we are having some problems with properly getting the bounding box coordinates of the labels during TkAgg
        rendering, i.e. interactive redering during debug. However it works find during actual running of the code
        using the Agg backend.

        NB getting the bounding boxes for the annoations is quite involved.
        It basically involves calling get_window_extent() on the annotation object. This will give you a bbox object
        which has its units in display units. You then have to transform this back to data units.
        This link for getting the bbox from the annotation:
        https://matplotlib.org/api/text_api.html#matplotlib.text.Annotation.get_window_extent
        This link for doing the transofrmations into the correct coordinates space:
        https://matplotlib.org/users/transforms_tutorial.html
        """

        fig = plt.figure(figsize=(15, 6))
        # required for getting the bbox of the text annotations
        fig.canvas.draw()

        apples = 'asdf'
        # order: dendro, label, species, depth, reef_type, season
        list_of_heights = [12,26,7,3,3,2]
        axarr = self._setup_grid_spec_and_axes_for_dendro_and_meta_fig_all_clades(list_of_heights)
        for i in range(len(self.clades)):
            self._make_dendrogram_with_meta_fig_for_all_clades(i, axarr)
        print('Saving image')
        plt.savefig('here.png', dpi=1200)

    def _make_dendrogram_with_meta_fig_for_all_clades(self, clade_index, axarr):
        clade = self.clades[clade_index]
        # Plot the dendrogram in first axes
        dendro_info = self._make_dendrogram_figure(
            clade=clade, ax=axarr[clade_index + 1][0], dist_df=self.profile_clade_dist_cct_specific_df_dict[clade],
            local_abundance_dict=self.prof_uid_to_local_abund_dict_post_cutoff, plot_labels=False)
        if clade_index == 0:
            axarr[clade_index + 1][0].set_yticks([0.0, 1.0])
        else:
            axarr[clade_index + 1][0].set_yticks([])

        title_list = ['Symbiodinium', 'Cladocopium', 'Durisdinium']
        axarr[clade_index + 1][0].set_title(label=title_list[clade_index], fontweight='bold', loc='center', fontstyle='italic', fontsize='small')
        self._remove_spines_from_dendro(axarr[clade_index + 1], clade_index=clade_index)

        # get the uids in order for the profiles in the dendrogram
        ordered_prof_uid_list = []
        prof_uid_to_x_loc_dict = {}
        for x_loc, lab_str in dendro_info['tick_to_profile_name_dict'].items():
            temp_uid = self.prof_name_to_uid_dict[lab_str.split(' ')[0]]
            ordered_prof_uid_list.append(temp_uid)
            prof_uid_to_x_loc_dict[temp_uid] = x_loc

        # Plot labels in second axes
        self._plot_labels_plot_for_dendro_and_meta_fig(axarr[clade_index + 1][0], dendro_info, axarr[clade_index + 1][1])
        if clade_index == 0:
            axarr[clade_index + 1][1].set_ylabel('ITS2 type profile name', fontsize='x-small', fontweight='bold', labelpad=18)


        # for each ITS2 type profile we will need to get the samples that the profile was found in
        # then we need to look up each of the samples and see which of the parameters it refers to.
        # as such that first look up of which samples the profiles were found in can be put into a dict
        # for use in each of the meta plots.
        # How to represent the mixed states is a little tricky. I think perhaps we should just use an eveness
        # index, where a very uneven distribution is light grey (i.e. almost one of the categories and
        # the more even distribution is closer to black (i.e. more of a mix).
        # to make the grey code its probably easiest to make an RGB tupple scaling from 255,255,255 which is
        # white, to 0,0,0 which is black. This would be scaled against the eveness.

        profile_uid_to_sample_uid_list_dict = self._generate_profile_uid_to_sample_uid_list_dict()

        # we will work with a class for doing the mata plotting as it will be quite involved
        mip = MetaInfoPlotter(parent_analysis=self, ordered_uid_list=ordered_prof_uid_list, meta_axarr=axarr[clade_index + 1][2:],
                              prof_uid_to_smpl_uid_list_dict=profile_uid_to_sample_uid_list_dict,
                              prof_uid_to_x_loc_dict=prof_uid_to_x_loc_dict, dend_ax=axarr[clade_index + 1][0], sub_cat_axarr=axarr[clade_index][2:], clade_index=clade_index)
        mip.plot_species_meta()
        mip.plot_depth_meta()
        mip.plot_reef_type()
        mip.plot_season()

    def _generate_profile_uid_to_sample_uid_list_dict(self, clade=None):

        profile_uid_to_sample_uid_list_dict = defaultdict(list)
        if clade is None:
            for prof_uid in list(self.prof_df_cutoff):
                self._pop_prof_uid_to_smp_name_dd_list(prof_uid, profile_uid_to_sample_uid_list_dict)
        else:
            for prof_uid in [uid for uid in list(self.prof_df_cutoff) if clade.upper() in self.prof_uid_to_name_dict[uid]]:
                self._pop_prof_uid_to_smp_name_dd_list(prof_uid, profile_uid_to_sample_uid_list_dict)
        return profile_uid_to_sample_uid_list_dict

    def _pop_prof_uid_to_smp_name_dd_list(self, prof_uid, profile_uid_to_sample_uid_list_dict):
        temp_series = self.prof_df_cutoff[prof_uid]
        temp_series_non_zero_series = temp_series[temp_series > 0]
        non_zero_indices = temp_series_non_zero_series.index.values.tolist()
        profile_uid_to_sample_uid_list_dict[prof_uid].extend(non_zero_indices)

    def make_dendrogram_with_meta_per_clade(self):
        """This function will make a figure that has a dendrogram at the top, the labels under that, then
        under this it will have data that link the metainfo to each of the types found.
        NB, I have modified the returned dictionary from hierarchy_sp.dendrogram_sp so that it contains the
        tick_to_profile_name_dict that we can use to easily associate the label that should be plotted in the
        labels plot.

        NB we are having some problems with properly getting the bounding box coordinates of the labels during TkAgg
        rendering, i.e. interactive redering during debug. However it works find during actual running of the code
        using the Agg backend.

        NB getting the bounding boxes for the annoations is quite involved.
        It basically involves calling get_window_extent() on the annotation object. This will give you a bbox object
        which has its units in display units. You then have to transform this back to data units.
        This link for getting the bbox from the annotation:
        https://matplotlib.org/api/text_api.html#matplotlib.text.Annotation.get_window_extent
        This link for doing the transofrmations into the correct coordinates space:
        https://matplotlib.org/users/transforms_tutorial.html
        """

        fig = plt.figure(figsize=(5, 6))
        # required for getting the bbox of the text annotations
        fig.canvas.draw()

        apples = 'asdf'
        # order: dendro, label, species, depth, reef_type, season
        list_of_heights = [18,18,7,3,3,2]
        # dendro_height = 18
        # label_height = 18
        # species_height = 7
        # depth_height = 3
        # reef_type_height = 3
        # season_height = 2
        # num_meta = 4
        total_plot_height = sum(list_of_heights)
        for clade in self.clades:
            self._make_dendro_with_meta_fig_for_clade(clade, list_of_heights)

    def _make_dendro_with_meta_fig_for_clade(self, clade, list_of_heights):
        axarr = self._setup_grid_spec_and_axes_for_dendro_and_meta_fig(list_of_heights)


        # Plot the dendrogram in first axes
        dendro_info = self._make_dendrogram_figure(
            clade=clade, ax=axarr[1][0], dist_df=self.profile_clade_dist_cct_specific_df_dict[clade],
            local_abundance_dict=self.prof_uid_to_local_abund_dict_post_cutoff, plot_labels=False)
        axarr[1][0].set_yticks([0.0, 1.0])
        self._remove_spines_from_dendro(axarr[1])

        # get the uids in order for the profiles in the dendrogram
        ordered_prof_uid_list = []
        prof_uid_to_x_loc_dict = {}
        for x_loc, lab_str in dendro_info['tick_to_profile_name_dict'].items():
            temp_uid = self.prof_name_to_uid_dict[lab_str.split(' ')[0]]
            ordered_prof_uid_list.append(temp_uid)
            prof_uid_to_x_loc_dict[temp_uid] = x_loc

        # Plot labels in second axes
        self._plot_labels_plot_for_dendro_and_meta_fig(axarr[1][0], dendro_info, axarr[1][1])
        axarr[1][1].set_ylabel('ITS2 type profile name')


        # for each ITS2 type profile we will need to get the samples that the profile was found in
        # then we need to look up each of the samples and see which of the parameters it refers to.
        # as such that first look up of which samples the profiles were found in can be put into a dict
        # for use in each of the meta plots.
        # How to represent the mixed states is a little tricky. I think perhaps we should just use an eveness
        # index, where a very uneven distribution is light grey (i.e. almost one of the categories and
        # the more even distribution is closer to black (i.e. more of a mix).
        # to make the grey code its probably easiest to make an RGB tupple scaling from 255,255,255 which is
        # white, to 0,0,0 which is black. This would be scaled against the eveness.

        profile_uid_to_sample_uid_list_dict = self._generate_profile_uid_to_sample_uid_list_dict()

        # we will work with a class for doing the mata plotting as it will be quite involved
        mip = MetaInfoPlotter(parent_analysis=self, ordered_uid_list=ordered_prof_uid_list, meta_axarr=axarr[1][2:],
                              prof_uid_to_smpl_uid_list_dict=profile_uid_to_sample_uid_list_dict,
                              prof_uid_to_x_loc_dict=prof_uid_to_x_loc_dict, dend_ax=axarr[1][0], sub_cat_axarr=axarr[0][2:])
        mip.plot_species_meta()
        mip.plot_depth_meta()
        mip.plot_reef_type()
        mip.plot_season()

        print('Saving image')
        plt.savefig('here.png', dpi=1200)
        # evenness can be calculated using skbio.diversity.alpha.simpson
        for profile_uid in ordered_prof_uid_list:
            list_of_smpl_uids = profile_uid_to_sample_uid_list_dict

    def _remove_spines_from_dendro(self, axarr, clade_index=None):
        if clade_index is not None:
            if clade_index == 0:
                axarr[0].set_ylabel('BrayCurtis distance', fontsize='x-small', fontweight='bold')
        else:
            axarr[0].set_ylabel('BrayCurtis distance', fontsize='x-small', fontweight='bold')

        axarr[0].spines['top'].set_visible(False)
        axarr[0].spines['right'].set_visible(False)
        axarr[0].spines['left'].set_visible(False)

    def _plot_labels_plot_for_dendro_and_meta_fig(self, dend_ax, dendro_info, labels_ax):
        # make the x axis limits of the labels plot exactly the same as the dendrogram plot
        # then we can use the dendrogram plot x coordinates to plot the labels in the labels plot.
        labels_ax.set_xlim(dend_ax.get_xlim())

        annotation_list = self._store_and_annotate_labels(dendro_info, labels_ax)

        lines = self._create_connection_lines(annotation_list, labels_ax)

        coll = self._create_lines_collection(lines)

        self._add_lines_to_axis(coll, labels_ax)


        labels_ax.spines['right'].set_visible(False)
        labels_ax.spines['left'].set_visible(False)
        labels_ax.spines['top'].set_visible(False)
        labels_ax.spines['bottom'].set_visible(False)
        labels_ax.set_xticks([])
        labels_ax.set_yticks([])

    def _add_lines_to_axis(self, coll, labels_ax):
        labels_ax.add_collection(coll)

    def _create_lines_collection(self, lines):
        coll = collections.LineCollection([ln_info.coord_list for ln_info in lines], colors=('black',),
                                          linestyles=('dotted',),
                                          linewidths=[ln_info.thickness for ln_info in lines])
        return coll

    def _create_connection_lines(self, annotation_list, labels_ax):
        lines = []
        min_gap_to_plot = 0  # the minimum dist required between label and bottom of plot
        y_val_buffer = 0.02  # the distance to leave between the line and the label
        for ann in annotation_list:
            bbox = ann.get_window_extent()
            inv = labels_ax.transData.inverted()
            bbox_data = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
            line_x = (bbox_data[1][0] + bbox_data[0][0]) / 2

            if bbox_data[0][1] > min_gap_to_plot and bbox_data[0][
                1] > y_val_buffer:  # then we should draw the connecting lines
                # the bottom connecting line
                lines.append(
                    hierarchy_sp.LineInfo([(line_x, min_gap_to_plot), (line_x, bbox_data[0][1] - y_val_buffer)],
                                          thickness=0.5, color='black'))
                # the top connecting line
                lines.append(
                    hierarchy_sp.LineInfo([(line_x, bbox_data[1][1] + y_val_buffer), (line_x, 1 - min_gap_to_plot)],
                                          thickness=0.5, color='black'))
            else:
                # the label is too large and there is no space to draw the connecting line
                pass

            # box_width = bbox_data[1][0] - bbox_data[0][0]
            # box_height = bbox_data[1][1] - bbox_data[0][1]
            # rec = patches.Rectangle((bbox_data[0][0], bbox_data[0][1]), box_width, box_height)
            # labels_ax.add_patch(rec)
        return lines

    def _store_and_annotate_labels(self, dendro_info, labels_ax):
        # Draw the annotations onto text annotations onto axes
        annotation_list = []
        for x_loc, lab_str in dendro_info['tick_to_profile_name_dict'].items():
            # fig.canvas.draw()
            annotation_list.append(
                labels_ax.annotate(s=lab_str, xy=(x_loc, 0.5), rotation='vertical', horizontalalignment='center',
                                   verticalalignment='center', fontsize='xx-small', fontweight='bold'))
        return annotation_list

    def _setup_grid_spec_and_axes_for_dendro_and_meta_fig_all_clades(self, list_of_heights):
        # in order (sub-cat, clade A, clade C, clade D)
        plot_widths_list = [6,25,42,24]

        gs = gridspec.GridSpec(sum(list_of_heights), sum(plot_widths_list))
        # 2d list where each list is a column contining multiple axes

        axarr = []
        # first set of axes that will be used to put the subcategory labels for the metainfo
        sub_cat_label_ax_list = []
        for i in range(len(list_of_heights)):
            temp_ax = plt.subplot(gs[sum(list_of_heights[:i]):sum(list_of_heights[: i + 1]), :plot_widths_list[0]])
            temp_ax.spines['top'].set_visible(False)
            temp_ax.spines['bottom'].set_visible(False)
            temp_ax.spines['right'].set_visible(False)
            temp_ax.spines['left'].set_visible(False)
            temp_ax.set_xticks([])
            temp_ax.set_yticks([])
            temp_ax.set_ylim(0,1)
            temp_ax.set_xlim(0,1)
            sub_cat_label_ax_list.append(temp_ax)
        axarr.append(sub_cat_label_ax_list)


        # then add the main data ax lists, one for each clade
        for clade in self.clades:
            clade_ax_list = []
            for i in range(len(list_of_heights)):
                if clade == 'A':
                    col_index_start = plot_widths_list[0]
                    col_index_end = sum(plot_widths_list[:2])
                elif clade == 'C':
                    col_index_start = sum(plot_widths_list[:2])
                    col_index_end = sum(plot_widths_list[:3])
                else:
                    col_index_start = sum(plot_widths_list[:3])
                    col_index_end = sum(plot_widths_list[:4])

                clade_ax_list.append(plt.subplot(gs[sum(list_of_heights[:i]):sum(list_of_heights[: i + 1]), col_index_start:col_index_end]))
            axarr.append(clade_ax_list)


        return axarr

    def _setup_grid_spec_and_axes_for_dendro_and_meta_fig(self, list_of_heights):
        gs = gridspec.GridSpec(sum(list_of_heights), 10)

        # 2d list where each list is a column contining multiple axes

        axarr = []
        # first set of axes that will be used to put the subcategory labels for the metainfo
        sub_cat_label_ax_list = []
        for i in range(len(list_of_heights)):
            temp_ax = plt.subplot(gs[sum(list_of_heights[:i]):sum(list_of_heights[: i + 1]), :2])
            temp_ax.spines['top'].set_visible(False)
            temp_ax.spines['bottom'].set_visible(False)
            temp_ax.spines['right'].set_visible(False)
            temp_ax.spines['left'].set_visible(False)
            temp_ax.set_xticks([])
            temp_ax.set_yticks([])
            temp_ax.set_ylim(0,1)
            temp_ax.set_xlim(0,1)
            sub_cat_label_ax_list.append(temp_ax)
        axarr.append(sub_cat_label_ax_list)

        # add the main data ax
        clade_a_ax_list = []
        for i in range(len(list_of_heights)):
            clade_a_ax_list.append(plt.subplot(gs[sum(list_of_heights[:i]):sum(list_of_heights[: i + 1]), 2:]))
        axarr.append(clade_a_ax_list)


        return axarr

    def make_dendrogram(self, cct_specific=False):
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
            # draw dendrograms in pairs where left is passed the types and abundances before the cutoff is applied
            # and before the specific cct distances have been caclulated, and the right is after these processing
            # steps have been applied.
            for clade in self.clades:
                fig, axarr = plt.subplots(1,2,figsize=(16, 12))

                # plot the non cutoff dendro first
                self._make_dendrogram_figure(
                    clade=clade, ax=axarr[0], dist_df=self.profile_clade_dist_df_dict[clade],
                    local_abundance_dict = self.prof_uid_to_local_abund_dict)

                # then the cct specific
                self._make_dendrogram_figure(
                    clade=clade, ax=axarr[1], dist_df=self.profile_clade_dist_cct_specific_df_dict[clade],
                    local_abundance_dict=self.prof_uid_to_local_abund_dict_post_cutoff)

                plt.tight_layout()
                plt.savefig(os.path.join(self.figure_dir, f'paired_dendogram_{clade}.png'))
                plt.savefig(os.path.join(self.figure_dir, f'paired_dendogram_{clade}.svg'))


        else:
            # draw a single dendrogram per clade
            for clade in self.clades:
                fig, ax = plt.subplots(figsize=(8, 12))
                self._make_dendrogram_figure(clade=clade, ax=ax, dist_df=self.profile_clade_dist_df_dict[clade],
                                             local_abundance_dict=self.prof_uid_to_local_abund_dict)
                plt.tight_layout()

    def _make_dendrogram_figure(self, clade, ax, dist_df, local_abundance_dict, plot_labels=True):
        """This is a method specifically aimed at plotting a single dendrogram and is used
        inside the methods that put larger figures together e.g. make_dendogram """
        condensed_dist = scipy.spatial.distance.squareform(dist_df)
        # this creates the linkage df that will be passed into the dendogram_sp function
        linkage = scipy.cluster.hierarchy.linkage(y=condensed_dist, optimal_ordering=True)
        thickness_dict = self._make_thickness_dict(clade, dist_df, local_abundance_dict)
        labels = self._make_labels_list(dist_df, local_abundance_dict)
        return self._draw_one_dendrogram(ax, labels, linkage, thickness_dict, plot_labels)

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

    def _draw_one_dendrogram(self, ax, labels, linkage, thickness_dict, plot_labels=True):
        return hierarchy_sp.dendrogram_sp(linkage, labels=labels, ax=ax,
                                         node_to_thickness_dict=thickness_dict,
                                         default_line_thickness=0.5, leaf_rotation=90, no_labels=not plot_labels)

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

    def permute_profile_permanova(self):
        """ Compute a PERMANOVA based on the between type distance matrixces.  This is not totally straight forward.
        Because the current profile distance matrix we have has each profile only listed once, and yet the
        profiles can have been found multiple times, we will need to create a new distance matrix where the
        sampling units are esentially CladeCollectionTypes, i.e. unique associations between a given sample and
        an AnalysisType. The individual distances that will be used in this new matrix will be looked up from the
        original profile distance matrix that was calculated using the specific set of CladeCollectionType IDs.

        Pseudo-code
        1 - Generate the profile uid to sample uid list dict
        2 - generate empty df that is the new distance matrix
        2a - Make a list of prof_uid_smp_uid strings in order.
        2b - use this as the columns and index of the new dataframe to create empty df.
        2c - create a DistanceMatrix object from the df
        2d - on the way to collecting the distance matrix df also populate a list that will
        be the groupings of the samples. I think this will need to be a single string that holds the grouping
        data. E.g. pist_1_in_win for an S. pistillata sample from 1m depth from an inshore reef in the winter.
        3 - once we have both the distance matrix and the group list we should be able to run the permanova and have
        a look at the results!

        It turns out that we can't do a factorial permanova using the skbio implementation. As such we will have to
        do the permanova in R using adonis. We can run r from within python using the rpy2 package.
        """

        # import rpy2.robjects as robjects


        for clade in self.clades:  # we will do a PERMANOVA per clade
            prof_dist_df = self.profile_clade_dist_cct_specific_df_dict[clade]
            profile_uid_to_sample_uid_list_dict = self._generate_profile_uid_to_sample_uid_list_dict(clade=clade)
            sample_unit_list = []
            # this list contains only the prof uid info from the above sample_unit_list, in int form for use
            # in the distance df lookup.
            prof_uid_list = []
            # the indexes in order that we will want to take from the meta df and pass to our PERMANOVA
            meta_info_indices = []
            for profile_uid, smple_uid_list in profile_uid_to_sample_uid_list_dict.items():
                for smp_uid in smple_uid_list:
                    # get the series from the meta df
                    meta_info_indices.append(smp_uid)
                    prof_uid_list.append(profile_uid)
                    sample_unit_list.append(f'{profile_uid}_{smp_uid}')

            meta_info_df_for_clade = self.metadata_info_df.loc[meta_info_indices, :]

            output_path_dist_matrix = os.path.join(self.outputs_dir, f'dists_permanova_types_{clade}.csv')
            output_pickle_dist_matrix = os.path.join(self.cache_dir, f'dists_permanova_types_{clade}.p')
            output_path_meta_info = os.path.join(self.outputs_dir, f'meta_info_{clade}.csv')
            if os.path.exists(output_pickle_dist_matrix):
                dist_df = pickle.load(open(output_pickle_dist_matrix, 'rb'))
            else:
                dist_df = pd.DataFrame(columns=sample_unit_list, index=sample_unit_list)
                print(f'Populating clade {clade} {len(list(dist_df))}x{len(list(dist_df))} distance matrix for PERMANOVA.')
                for r_ind in range(len(list(dist_df))):
                    print(f'row {r_ind} populated')
                    for c_ind in range(len(list(dist_df))):
                        dist_df.iat[r_ind, c_ind] = prof_dist_df[prof_uid_list[r_ind]][prof_uid_list[c_ind]]
                pickle.dump(dist_df, open(output_pickle_dist_matrix, 'wb'))

            # here we have the distance matrix that we will want to compute on
            # we should write this out as a csv with no rows or headers so that we can read it in in R
            dist_df.to_csv(path_or_buf=output_path_dist_matrix, sep=',', header=False, index=False, line_terminator='\n')

            # we will also need to output the metainfo df for the analysis type instances in question
            meta_info_df_for_clade.to_csv(
                path_or_buf=output_path_meta_info, sep=',', header=True, index=False, line_terminator='\n')

            import rpy2.robjects as robjects
            import rpy2.robjects.packages as rpackages
            from rpy2.robjects.vectors import StrVector
            package_names = ('vegan',)

            if all(rpackages.isinstalled(x) for x in package_names):
                have_packages = True
            else:
                have_packages = False

            if not have_packages:
                utils = rpackages.importr('utils')
                utils.chooseCRANmirror(ind=1)
                package_names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
                if package_names_to_install:
                    utils.install_packages(StrVector(package_names_to_install))

            r_distance_data = robjects.r(f'read.table(file="/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/dists_permanova_types_A.csv", sep=",", header=FALSE)')
            r_meta_info = robjects.r(f'read.table(file="/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/meta_info_A.csv", sep=",", header=TRUE)')
            vegan = rpackages.importr('vegan')

            perm_results = robjects.r('')
            print(data)
            apples = 'asdf'



        apples = 'pies'

class MetaInfoPlotter:
    def __init__(self, parent_analysis, ordered_uid_list, meta_axarr, prof_uid_to_smpl_uid_list_dict, prof_uid_to_x_loc_dict, dend_ax, sub_cat_axarr, clade_index):
        self.parent_analysis = parent_analysis
        self.ordered_prof_uid_list = ordered_uid_list
        self.clade_index = clade_index
        # these are the axes that will display the actual data
        self.meta_axarr = meta_axarr
        # these are the axes that will hold the subcategory labels
        self.sub_cat_axarr = sub_cat_axarr
        # set the x axis lims to match the dend_ax
        for ax, cat_ax, label, labpad in zip(self.meta_axarr, self.sub_cat_axarr, ['Species', 'Depth', 'Reef\nType', 'Season'], [0,10,0,10]):
            ax.set_xlim(dend_ax.get_xlim())
            # ax.spines['top'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim((0, 1))
            # if clade_index ==0:
            #     cat_ax.set_ylabel(label, rotation='vertical', fontweight='bold', fontsize='x-small',
            #                        verticalalignment='center', labelpad=labpad)


        self.prof_uid_to_smpl_uid_list_dict = prof_uid_to_smpl_uid_list_dict
        self.prof_uid_to_x_loc_dict = prof_uid_to_x_loc_dict
        self.smpl_meta_df = self.parent_analysis.metadata_info_df
        # the space left between the info boxes of the plot
        # this should be set dynmaically at some point rather than hard coded
        self.meta_box_buffer = 1
        self.meta_box_height = 0.8

        # species
        self.species_plotter = None
        self.depth_plotter = None
        self.reef_type_plotter = None
        self.season = None

    def plot_species_meta(self):
        # Plot species, season, depth, reef type
        color_dict = {
            'G': '#98FB98', 'GX': '#F0E68C', 'M': '#DDA0DD', 'P': '#8B008B',
            'PC': '#00BFFF', 'SE': '#0000CD', 'ST': '#D2691E'}
        category_list = ['M', 'G', 'GX',  'P', 'PC', 'SE', 'ST']
        category_labels = ['M. dichotoma', 'G. planulata', 'G. fascicularis', 'Porites spp.', 'P. verrucosa', 'S. hystrix', 'S. pistillata']
        self.species_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[0], cat_ax=self.sub_cat_axarr[0], color_dict=color_dict,
                                               category_list=category_list, category_df_header='species', category_labels=category_labels)
        self.species_plotter.plot()

    def plot_depth_meta(self):
        color_dict = {
            1:'#CAE1FF', 15: '#2E37FE', 30: '#000080'}
        category_list = [30, 15, 1]
        category_labels = ['30 m', '15 m', '1 m']
        self.depth_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[1], cat_ax=self.sub_cat_axarr[1],color_dict=color_dict,
                                               category_list=category_list, category_df_header='depth', category_labels=category_labels)
        self.depth_plotter.plot()

    def plot_reef_type(self):
        color_dict = {
            'Inshore': '#FF0000', 'Midshelf': '#FFFF00', 'Offshore': '#008000'}
        category_list = ['Offshore', 'Midshelf', 'Inshore']
        category_labels = ['Offshore', 'Midshelf', 'Inshore']
        self.depth_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[2], cat_ax=self.sub_cat_axarr[2],color_dict=color_dict,
                                             category_list=category_list, category_df_header='reef_type', category_labels=category_labels)
        self.depth_plotter.plot()

    def plot_season(self):
        color_dict = {
            'Summer': '#FF0000', 'Winter': '#00BFFF'}
        category_list = ['Summer', 'Winter']
        category_labels = ['Summer', 'Winter']
        self.depth_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[3], cat_ax=self.sub_cat_axarr[3],color_dict=color_dict,
                                             category_list=category_list, category_df_header='season', category_labels=category_labels)
        self.depth_plotter.plot()

    class CatPlotter:
        def __init__(self, parent_meta_plotter, ax, cat_ax, color_dict, category_list, category_df_header, category_labels):
            self.parent_meta_plotter = parent_meta_plotter
            self.prof_uid_list = self.parent_meta_plotter.ordered_prof_uid_list
            self.prof_uid_to_smpl_uid_list_dict = self.parent_meta_plotter.prof_uid_to_smpl_uid_list_dict
            self.prof_x_loc_dict = self.parent_meta_plotter.prof_uid_to_x_loc_dict
            self.meta_df = self.parent_meta_plotter.smpl_meta_df
            self.ax = ax
            self.cat_ax = cat_ax
            x_loc_one = self.prof_x_loc_dict[self.prof_uid_list[0]]
            x_loc_two = self.prof_x_loc_dict[self.prof_uid_list[1]]
            self.dist_betwee_x_locs = x_loc_two - x_loc_one
            # the space left between the info boxes of the plot
            # this should be set dynmaically at some point rather than hard coded
            self.meta_box_buffer = self.parent_meta_plotter.meta_box_buffer
            self.meta_box_height = self.parent_meta_plotter.meta_box_height
            self.color_dict = color_dict
            self.category_list = category_list
            self.category_df_header = category_df_header
            self.cat_labels = category_labels

        def plot(self):
            y0_list, height_list = self._plot_data_ax()

            if self.parent_meta_plotter.clade_index == 0:
                # only have to make the sub category plot once.
                self._make_sub_category_plot(height_list, y0_list)

        def _make_sub_category_plot(self, height_list, y0_list):
            # now populate the category axis with the sub category labels
            # y values will be the y0list + half the height
            bar_height = height_list[0]
            for cat_lab, y0_val in zip(self.cat_labels, y0_list):
                if self.category_df_header == 'species':  # italics
                    self.cat_ax.annotate(s=cat_lab, xy=(1, y0_val + (0.5 * bar_height)), horizontalalignment='right',
                                         verticalalignment='center', fontsize='xx-small', fontstyle='italic')
                else:
                    self.cat_ax.annotate(s=cat_lab, xy=(1, y0_val + (0.5 * bar_height)), horizontalalignment='right',
                                         verticalalignment='center', fontsize='xx-small')

        def _plot_data_ax(self):
            """We will plot a horizontal bar plot using rectangle patches"""
            for prof_uid in self.prof_uid_list:
                list_of_sample_uids = self.prof_uid_to_smpl_uid_list_dict[prof_uid]
                list_of_cat_instances = [self.meta_df.at[smpl_uid, self.category_df_header] for smpl_uid in
                                         list_of_sample_uids]
                # calculate eveness
                counter = Counter(list_of_cat_instances)

                # Then this only contains the one species and it should simply be the species color
                x0_list, y0_list, width_list, height_list = self._get_rect_attributes(prof_uid, counter)

                for x, y, w, h, s in zip(x0_list, y0_list, width_list, height_list, self.category_list):
                    if w > 0:
                        rect_p = patches.Rectangle(
                            xy=(x, y), width=w, height=h, facecolor=self.color_dict[s], edgecolor='none')
                        self.ax.add_patch(rect_p)
            return y0_list, height_list

        def _get_rect_attributes(self, prof_uid, counter):

            num_categories = len(self.color_dict.items())

            bar_height = (1/(num_categories))
            y0_list = [i * bar_height for i in range(num_categories)]
            height_list = [bar_height for _ in range(num_categories)]

            x_loc_of_prof = self.prof_x_loc_dict[prof_uid]
            data_x0 = (x_loc_of_prof - (self.dist_betwee_x_locs / 2)) + self.meta_box_buffer
            data_x1 = (x_loc_of_prof + (self.dist_betwee_x_locs / 2)) - self.meta_box_buffer
            rect_width = data_x1 - data_x0

            width_list = []
            num_samples = sum(counter.values())
            for cat in self.category_list:
                if cat in counter:
                    width_list.append((counter[cat]/num_samples)*rect_width)
                else:
                    width_list.append(0)
            x0_list = [data_x0 for _ in range(num_categories)]
            return x0_list, y0_list, width_list, height_list


if __name__ == "__main__":
    rest_analysis = RestrepoAnalysis(
        base_input_dir = os.path.join(
            '/Users', 'humebc', 'Google_Drive', 'projects', 'alejandro_et_al_2018',
            'resources', 'sp_outputs_20190417', '2019-04-17_07-14-49.317290'),
        profile_rel_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.profiles.relative.txt',
        profile_abs_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.profiles.absolute.txt',
        seq_rel_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.seqs.relative.txt',
        seq_abs_abund_ouput_path='37_six_analysis_2019-04-17_07-14-49.317290.seqs.absolute.txt',
        clade_A_profile_dist_path=os.path.join(
            'between_profile_distances', 'A',
            '2019-04-17_07-14-49.317290.bray_curtis_within_clade_profile_distances_A.dist'),
        clade_C_profile_dist_path=os.path.join(
            'between_profile_distances', 'C',
            '2019-04-17_07-14-49.317290.bray_curtis_within_clade_profile_distances_C.dist'),
        clade_D_profile_dist_path=os.path.join(
            'between_profile_distances', 'D',
            '2019-04-17_07-14-49.317290.bray_curtis_within_clade_profile_distances_D.dist'),
        clade_A__profile_dist_cct_specific_path=os.path.join(
            'cct_specific_between_profile_distances','2019-04-16_08-37-52_564623', 'between_profiles','A',
            '2019-04-16_08-37-52.564623.bray_curtis_within_clade_profile_distances_A.dist'),
        clade_C_profile_dist_cct_specific_path=os.path.join(
            'cct_specific_between_profile_distances', '2019-04-16_08-37-52_564623','between_profiles','C',
            '2019-04-16_08-37-52.564623.bray_curtis_within_clade_profile_distances_C.dist'),
        clade_D__profile_dist_cct_specific_path=os.path.join(
            'cct_specific_between_profile_distances', '2019-04-16_08-37-52_564623','between_profiles','D',
            '2019-04-16_08-37-52.564623.bray_curtis_within_clade_profile_distances_D.dist'),
        meta_data_input_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/resources/meta_info.csv',
        clade_A_smpl_dist_path=os.path.join(
            'between_sample_distance_sqrt_trans', 'A',
            '2019-05-06_05-07-17.800728.bray_curtis_sample_distances_A.dist'),
        clade_C_smpl_dist_path=os.path.join(
            'between_sample_distance_sqrt_trans', 'C',
            '2019-05-06_05-07-17.800728.bray_curtis_sample_distances_C.dist'),
        clade_D_smpl_dist_path=os.path.join(
            'between_sample_distance_sqrt_trans', 'D',
            '2019-05-06_05-07-17.800728.bray_curtis_sample_distances_D.dist'),
        ignore_cache=True, cutoff_abund=0.06)
    # rest_analysis.make_dendrogram_with_meta_all_clades()
    rest_analysis.make_sample_balance_figure()
    rest_analysis.plot_pcoa_of_cladal()
    rest_analysis.permute_profile_permanova()
    rest_analysis.get_list_of_clade_col_type_uids_for_unifrac()
    rest_analysis.histogram_of_all_abundance_values()






