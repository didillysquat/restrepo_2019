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
    def __init__(self, profile_rel_abund_ouput_path, profile_abs_abund_ouput_path,
                 seq_rel_abund_ouput_path, seq_abs_abund_ouput_path, clade_A_dist_path, clade_C_dist_path,
                 clade_D_dist_path):
        # Although we see clade F in the dataset this is minimal and so we will
        # tackle this sepeately to the analysis of the A, C and D.
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.clades = list('ACD')

        # Paths to raw info files
        self.profile_rel_abund_ouput_path = profile_rel_abund_ouput_path
        self.profile_abs_abund_ouput_path = profile_abs_abund_ouput_path
        self.seq_rel_abund_ouput_path = seq_rel_abund_ouput_path
        self.seq_abs_abund_ouput_path = seq_abs_abund_ouput_path
        self.clade_dist_path_dict = {'A' : clade_A_dist_path, 'C' : clade_C_dist_path, 'D' : clade_D_dist_path}

        # Iterative attributes

        # cache implementation
        self.cache_dir = os.path.join(self.cwd, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.clade_dist_dict_p_path = os.path.join(self.cache_dir, 'clade_dist_df_dict.p')

        # Info containers
        self.smp_uid_to_name_dict = None
        self.prof_uid_to_local_abund_dict = None
        self.prof_uid_to_global_abund_dict = None
        self.prof_uid_to_name_dict = None
        self.clade_dist_df_dict = {}
        self._populate_clade_dist_df_dict()
        self.profile_df  = {}
        self._populate_profile_df()

    def _populate_clade_dist_df_dict(self):
        try:
            self.clade_dist_df_dict = pickle.load(file=open(self.clade_dist_dict_p_path, 'rb'))
        except FileNotFoundError:
            for clade in self.clades:
                with open(self.clade_dist_path_dict[clade], 'r') as f:
                    clade_data = [out_line.split('\t') for out_line in [line.rstrip() for line in f][1:]]

                df = pd.DataFrame(clade_data)
                df.set_index(keys=0, drop=True, inplace=True)
                df.columns = df.index.values.tolist()
                self.clade_dist_df_dict[clade] = df.astype(dtype='float')
            pickle.dump(obj=self.clade_dist_df_dict, file=open(self.clade_dist_dict_p_path, 'wb'))


    def _populate_profile_df(self):
        # read in df
        df = pd.read_csv(filepath_or_buffer=self.profile_rel_abund_ouput_path, sep='\t', header=None)
        # collect sample uid to name info
        self.smp_uid_to_name_dict = {uid: name for uid, name in zip(df[0][7:-12], df[1][7:-12])}
        # del smp name column
        del df[1]
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
        df = df.iloc[:-12,]
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

    def make_dendogram(self):
        """I have modified the scipi.cluster.hierarchy.dendrogram so that it can take a line thickness dictionary.
        It will use this dictionary to modify the thickness of the leaves on the dendrogram so that we can see which
        types were the most abundant.

        NB The documentation for the linkage() is misleading. It cannot take a redundant distance matrix.
        The matrix must first be condensed. Again, misleadingly, this can be done using the poorly named squareform()
        that does both condensed to redundant and vice versa.

        """
        for clade in self.clades:
            dist_df = self.clade_dist_df_dict[clade]
            condensed_dist = scipy.spatial.distance.squareform(dist_df)
            # this creates the linkage df that will be passed into the dendogram_sp function
            linkage = scipy.cluster.hierarchy.linkage(y=condensed_dist, optimal_ordering=True)
            fig, ax = plt.subplots(figsize=(8,12))

            # generate the thickness dictionary. Lets work with line thicknesses of 1, 2, 3 and 4
            # assign according to which quartile
            max_abund = sorted([value for uid, value in self.prof_uid_to_local_abund_dict.items() if clade in self.prof_uid_to_name_dict[uid]], reverse=True)[0]
            thickness_dict = {}
            for uid in list(self.profile_df):
                if clade in self.prof_uid_to_name_dict[uid]:
                    abund = self.prof_uid_to_local_abund_dict[uid]
                    if abund < max_abund* 0.1:
                        thickness_dict[self.prof_uid_to_name_dict[uid]] = 1
                    elif abund < max_abund * 0.2:
                        thickness_dict[self.prof_uid_to_name_dict[uid]] = 2
                    elif abund < max_abund * 0.3:
                        thickness_dict[self.prof_uid_to_name_dict[uid]] = 3
                    else:
                        thickness_dict[self.prof_uid_to_name_dict[uid]] = 4

            den = hierarchy_sp.dendrogram_sp(linkage, labels=dist_df.index.values.tolist(), ax=ax,
                                            node_to_thickness_dict=thickness_dict,
                                             default_line_thickness=0.5)
            plt.tight_layout()

if __name__ == "__main__":
    rest_analysis = RestrepoAnalysis(
        profile_rel_abund_ouput_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/'
                                     'resources/new_analysis_output/'
                                     '65_DBV_20190401_2019-04-05_00-01-08.517088.profiles.relative.txt',
        profile_abs_abund_ouput_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/'
                                     'resources/new_analysis_output/'
                                     '65_DBV_20190401_2019-04-05_00-01-08.517088.profiles.absolute.txt',
        seq_rel_abund_ouput_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/'
                                     'resources/new_analysis_output/'
                                     '65_DBV_20190401_2019-04-05_00-01-08.517088.seqs.relative.txt',
        seq_abs_abund_ouput_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/'
                                 'resources/new_analysis_output/'
                                 '65_DBV_20190401_2019-04-05_00-01-08.517088.seqs.absolute.txt',
        clade_A_dist_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/resources/'
                          'new_analysis_output/between_type_dist/A/'
                          'test.txt',
        clade_C_dist_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/resources/'
                          'new_analysis_output/between_type_dist/C/'
                          '2019-04-07_14-16-40.814173.bray_curtis_within_clade_sample_distances.dist',
        clade_D_dist_path='/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/resources/'
                          'new_analysis_output/between_type_dist/D/'
                          '2019-04-07_14-16-40.814173.bray_curtis_within_clade_sample_distances.dist')
    rest_analysis.make_dendogram()