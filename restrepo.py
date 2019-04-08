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

import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import numpy as np
import hierarchy_sp

class RestrepoAnalysis:
    def __init__(self, profile_rel_abund_ouput_path, profile_abs_abund_ouput_path,
                 seq_rel_abund_ouput_path, seq_abs_abund_ouput_path, clade_A_dist_path, clade_C_dist_path,
                 clade_D_dist_path):
        # Although we see clade F in the dataset this is minimal and so we will
        # tackle this sepeately to the analysis of the A, C and D.
        self.clades = list('ACD')
        self.profile_rel_abund_ouput_path = profile_rel_abund_ouput_path
        self.profile_abs_abund_ouput_path = profile_abs_abund_ouput_path
        self.seq_rel_abund_ouput_path = seq_rel_abund_ouput_path
        self.seq_abs_abund_ouput_path = seq_abs_abund_ouput_path
        self.clade_dist_path_dict = {'A' : clade_A_dist_path, 'C' : clade_C_dist_path, 'D' : clade_D_dist_path}
        self.clade_dist_df_cache_path_dict = {}
        self.clade_dist_df_dict = {}

        # Iterative attributes

    def populate_clade_dist_df_dict(self):
        for clade in self.clades:
            with open(self.clade_dist_path_dict[clade], 'r') as f:
                clade_data = [out_line.split('\t') for out_line in [line.rstrip() for line in f][1:]]

            df = pd.DataFrame(clade_data)
            df.set_index(keys=0, drop=True, inplace=True)
            df.columns = df.index.values.tolist()
            z = scipy.cluster.hierarchy.linkage(y=df, optimal_ordering=True)
            self.clade_dist_df_dict[clade] = df
            fig, ax = plt.subplots(figsize=(8,12))
            dn = hierarchy_sp.dendrogram_sp(z, labels=df.index.values.tolist(), ax=ax,
                                            node_to_thickness_dict={name : 1.5 for name in df.index.values.tolist()})

            plt.tight_layout()
            apples = 'asdf'



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
    rest_analysis.populate_clade_dist_df_dict()