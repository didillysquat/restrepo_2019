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
from matplotlib.patches import Polygon
from collections import defaultdict, Counter
import skbio.diversity.alpha
import skbio.stats.distance
import sys
from cartopy.mpl.gridliner import Gridliner
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy
from scipy.spatial.distance import braycurtis
import itertools
from skbio.stats.ordination import pcoa
import statistics
from multiprocessing import Pool
from statistics import variance
from sklearn import linear_model
import re
import seaborn as sns
import scipy.stats
from matplotlib_venn import venn2
from netCDF4 import Dataset

def braycurtis_tup(u_v_tup, w=None):
    import scipy.spatial.distance as distance
    """
    Compute the Bray-Curtis distance between two 1-D arrays.

    Bray-Curtis distance is defined as

    .. math::

       \\sum{|u_i-v_i|} / \\sum{|u_i+v_i|}

    The Bray-Curtis distance is in the range [0, 1] if all coordinates are
    positive, and is undefined if the inputs are of length zero.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    braycurtis : double
        The Bray-Curtis distance between 1-D arrays `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.braycurtis([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.braycurtis([1, 1, 0], [0, 1, 0])
    0.33333333333333331

    """
    u = u_v_tup[0]
    v = u_v_tup[1]
    u = distance._validate_vector(u)
    v = distance._validate_vector(v, dtype=np.float64)
    l1_diff = abs(u - v)
    l1_sum = abs(u + v)
    if w is not None:
        w = distance._validate_weights(w)
        l1_diff = w * l1_diff
        l1_sum = w * l1_sum
    return l1_diff.sum() / l1_sum.sum()

class RestrepoAnalysis:
    """The class responsible for doing all python based analyses for the restrepo et al. 2019 paper.
    NB although we see clade F in the dataset this is minimal and so we will
    tackle this sepeately to the analysis of the A, C and D."""
    def __init__(self, cutoff_abund, seq_distance_method, profile_distance_method='unifrac', ignore_cache=False, remove_se=False, maj_only=False, remove_se_clade_props=False):
        # root_dir is the root dir of the git repo
        self.root_dir = os.path.dirname(os.path.realpath(__file__))

        # base input dir
        self.base_input_dir = os.path.join(self.root_dir, 'input')


        # path to the sample meta information such as species, reef etc.
        self.meta_data_input_path = os.path.join(self.base_input_dir, 'meta_info.csv')


        # hobo data
        self.hobo_dir = os.path.join(self.base_input_dir, 'hobo_csv')

        # gis data
        self.gis_input_base_path = os.path.join(self.base_input_dir, 'gis')

        # dir containing the sp output files
        self.base_sp_output_dir = os.path.join(self.base_input_dir, 'sp_outputs_20190807', '2019-08-06_09-21-49.148787')
        # init all of the sp_output paths that lead to the sp files used in this analysis
        self._init_sp_output_paths()

        # Figure output dir
        self.figure_dir = os.path.join(self.root_dir, 'figures')
        os.makedirs(self.figure_dir, exist_ok=True)

        # Output dir
        self.outputs_dir = os.path.join(self.root_dir, 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Cache dir
        self.cache_dir = os.path.join(self.root_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # whether to use the cache system
        self.ignore_cache = ignore_cache
        # whether to remove S. hystrix samples from the clade D matrix
        self.remove_se=remove_se
        # whether to only include only samples into a given clades distance matrix that have
        # that clade as their majoirty
        self.maj_only=maj_only
        # whether to produce the between sample calde proportion dfs with the s. hystrix samples removed
        # along with the paried info df that will be used in R to conduct a permanova
        self.remove_se_clade_props = remove_se_clade_props

        self.clades = list('ACD')
        self.clade_genera_labels = ['Symbiodinium', 'Cladocopium', 'Durusdinium']

        # Info containers
        self.smp_uid_to_name_dict = None
        self.smp_name_to_uid_dict = None
        self.prof_uid_to_local_abund_dict = None
        self.prof_uid_to_local_abund_dict_post_cutoff = {}
        self.prof_uid_to_global_abund_dict = None
        self.prof_uid_to_name_dict = None
        self.prof_name_to_uid_dict = None

        self.seq_distance_method = seq_distance_method
        self.profile_distance_method = profile_distance_method

        # Between profile distances dataframe holder dict before 0.05 cutoff
        # This same dict will be used to hold the pre-cutoff profile dfs regardless of the dist method used.
        self.between_profile_clade_dist_df_dict = {}
        self._populate_clade_dist_df_dict()

        self.cutoff_abund = cutoff_abund
        # This one dict will be used to hold the post-cutoff profile dfs regardless of what cutoff
        # is used and regardless of which distance method is employed.
        self.between_profile_clade_dist_cct_specific_df_dict = {}

        if self.profile_distance_method == 'braycurtis':
            if self.cutoff_abund == 0.05:
                # Between profile distances dataframe holder dict after 0.05 cutoff
                if self.between_profile_clade_dist_cct_005_braycurtis_specific_path_dict['A']:
                    # if we have the cct_speicifc distances
                    self._populate_clade_dist_df_dict(cct_specific='005')
            else:
                # Between profile distances dataframe holder dict after 0.40 cutoff
                if self.between_profile_clade_dist_cct_040_braycurtis_specific_path_dict['A']:
                    # if we have the cct_speicifc distances
                    self._populate_clade_dist_df_dict(cct_specific='040')
        else: #  'unifrac'
            if self.cutoff_abund == 0.05:
                # Between profile distances dataframe holder dict after 0.05 cutoff
                if self.between_profile_clade_dist_cct_005_unifrac_specific_path_dict['A']:
                    # if we have the cct_speicifc distances
                    self._populate_clade_dist_df_dict(cct_specific='005')
            else:
                # Between profile distances dataframe holder dict after 0.40 cutoff
                if self.between_profile_clade_dist_cct_040_unifrac_specific_path_dict['A']:
                    # if we have the cct_speicifc distances
                    self._populate_clade_dist_df_dict(cct_specific='040')

        # Between sample distances dataframe
        # This one dict will be used to hold the between-sample distance dfs regardless of what
        # distance method is employed.
        self.between_sample_clade_dist_df_dict = {}
        self._populate_clade_dist_df_dict(smp_dist=True)


        # ITS2 type profile abundance dataframe before 0.05 cutoff
        self.profile_abundance_df  = None
        # ITS2 type profile meta info dataframe
        self.profile_meta_info_df = None
        self._populate_profile_abund_meta_dfs_and_info_containers()

        # ITS2 type profile abundance dataframe after 0.05 cutoff
        self.profile_abundance_df_cutoff = None
        self.create_profile_df_with_cutoff()

        # TODO we want to move towards having two ITS2 type profile schematics with hard coded cutoffs
        # The first will be >0.40 and the second will be >0.05 and <0.40. We will move away from having
        # specific distance outputs for these clade collection types and rather use the profile distances
        # that we have from the main output (i.e. containing all types). We will simply pull out the
        # distances for the types that we need by getting rid of those types that are not represented in the
        # lower or higher collection of sample-ITS2 type profile abundances.

        # The simplest way to do this will be to calculate the df of abundance first and then the distance matrix that
        # goes with it and then pass this pairing into the schematic making code.
        self.profile_abundance_df_cutoff_high = None
        self.prof_uid_to_local_abund_dict_cutoff_high = {}
        self.profile_distance_df_dict_cutoff_high = {}
        self._create_profile_df_with_cutoff_high_low(cutoff_low=0.40)
        self._populate_clade_dist_df_dict(cct_specific='040')
        self.profile_abundance_df_cutoff_low = None
        self.prof_uid_to_local_abund_dict_cutoff_low = {}
        self.profile_distance_df_dict_cutoff_low = {}
        self._create_profile_df_with_cutoff_high_low(cutoff_low=0.05, cutoff_high=0.40)
        self.get_list_of_clade_col_type_uids_for_unifrac(high_low='low')
        self._populate_clade_dist_df_dict(cct_specific='low')
        # here we should have all of the items that we'll want to be passing into the dendrogram figure

        # we will also create a profile abundance df that is only the <0.05 profiles so that we can have a look
        # at them and see how genuine these
        self.profile_abundance_df_cutoff_background = None
        self.prof_uid_to_local_abund_dict_cutoff_background = {}
        self.profile_distance_df_dict_cutoff_background = {}
        self._create_profile_df_with_cutoff_high_low(cutoff_low=0.00, cutoff_high=0.05)
        self.get_list_of_clade_col_type_uids_for_unifrac(high_low='background')
        self._populate_clade_dist_df_dict(cct_specific='background')

        # Temperature dataframe
        self.temperature_df = None
        self.daily_temperature_av_df = None
        self.daily_temperature_max_df = None
        self.daily_temperature_min_df = None
        self.daily_temperature_range_df = None
        self._make_temp_df()
        self.remotely_sensed_sst_df = self._make_remotely_sensed_sst_df()

        # ITS2 sequence abundance (post-MED) dataframe
        self.post_med_seq_abundance_relative_df = self._post_med_seq_abundance_relative_df()

        # self.pre_med_seq_abundance_relative_df = self._pre_med_seq_abundance_relative_df()

        # ITS2 sequence abundance meta info for sequencing (QC steps)
        self.seq_meta_data_df = self._populate_seq_meta_data_df()

        # Sample meta info, reef, species etc.
        self.experimental_metadata_info_df = self._init_metadata_info_df()

        # Clade proportion dataframes (must be made after meta_data)
        # this df actually has the proportions normalised to 100 000 sequences
        self.clade_proportion_df = pd.DataFrame(columns=list('ACD'),
                                                index=self.post_med_seq_abundance_relative_df.index.values.tolist())
        # this one is the actual proportions (i.e. 0.03, 0.97, 0.00)
        self.clade_proportion_df_non_normalised = pd.DataFrame(columns=list('ACD'),
                                                               index=self.post_med_seq_abundance_relative_df.index.values.tolist())
        self.between_sample_clade_proportion_distances_df = None
        self.clade_prop_pcoa_coords = None
        self._create_clade_prop_distances()
        self.clade_proportion_df_non_normalised = self.clade_proportion_df_non_normalised.astype(float)

        if self.remove_se_clade_props:
            self._output_clade_prop_df_no_se()

        # info dictionaries
        self.old_color_dict = {
            'G': '#98FB98', 'GX': '#F0E68C', 'M': '#DDA0DD', 'P': '#8B008B',
            'PC': '#00BFFF', 'SE': '#0000CD', 'ST': '#D2691E',
            1: '#CAE1FF', 15: '#2E37FE', 30: '#000080',
            'Summer': '#FF0000', 'Winter': '#00BFFF',
            'Inshore': '#FF0000', 'Midshelf': '#FFFF00', 'Offshore': '#008000',
            'Al Fahal': '#A7414A', 'Abu Madafi': '#563838',
            'Qita al Kirsh': '#6A8A82', 'Shib Nazar': '#A37C27',
            'Tahla': '#1ECFD6', 'Fsar': '#6465A5'}
        self.reefs = ['Fsar', 'Tahla', 'Qita al Kirsh', 'Al Fahal', 'Shib Nazar', 'Abu Madafi']

        self.species_category_list = ['SE','PC', 'M', 'G', 'P','GX', 'ST']
        self.species_category_labels = ['S. hystrix','P. verrucosa','M. dichotoma', 'G. planulata',
                                        'Porites spp.','G. fascicularis', 'S. pistillata']
        self.reef_types = ['Inshore', 'Midshelf', 'Offshore']
        self.depths = [1, 15, 30]
        self.seasons = ['Winter', 'Summer']
        sites = ['Abu Madafi', "Shib Nazar", 'Al Fahal', 'Qita al Kirsh', 'Tahla', 'Fsar']
        x_site_coords = [38.778333, 38.854283, 38.960533, 38.992800, 39.055275, 39.030267, ]
        y_site_coords = [22.109143, 22.322533, 22.306233, 22.430717, 22.308564, 22.232617, ]
        self.site_lat_long_tups = {site: (lat, long) for site, lat, long in zip(sites, y_site_coords, x_site_coords)}


        self._del_propblem_sample()

    def populate_data_sheet(self):
        path_to_data_sheet_csv = os.path.join(self.base_input_dir, 'restrepo_data_sheet_20190904.csv')
        df = pd.read_csv(path_to_data_sheet_csv, skiprows=1)
        df = df.iloc[:,:14]

        list_of_files_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/input/list_of_files.txt'
        with open(list_of_files_path, 'r') as f:
            list_of_file_names = [line.rstrip() for line in f]
        list_of_file_name_components = [filename.split('_') for filename in list_of_file_names]
        # populate the fwd and rev files
        # date is 12
        # long is 11
        # lat is 10
        # fwd is 1
        # rev is 2
        df['fastq_fwd_file_name'] = df['fastq_fwd_file_name'].astype(str)
        df['fastq_rev_file_name'] = df['fastq_rev_file_name'].astype(str)
        df['collection_date'] = df['collection_date'].astype(str)
        for i, row in df.iterrows():
            count = 0
            sample_name = row['sample_name']
            for file_name_list in list_of_file_name_components:
                if sample_name in file_name_list:
                    if 'R1' in file_name_list:
                        df.iat[i, 1] = '_'.join(file_name_list)
                        count += 1
                    elif 'R2' in file_name_list:
                        df.iat[i, 2] = '_'.join(file_name_list)
                        count += 1
            if count != 2:
                print('whoops')
            for k, v in self.smp_name_to_uid_dict.items():
                if sample_name in k:
                    sample_uid = v
                    break
            if sample_name == 'FS15SE8':
                site = 'Fsar'
            else:
                site = self.experimental_metadata_info_df.at[sample_uid, 'reef']

            lat, long = self.site_lat_long_tups[site]
            df.iat[i, 10] = lat
            df.iat[i, 11] = long

            if sample_name == 'FS15SE8':
                df.iat[i, 12] = "17.07.25"
            else:
                if self.experimental_metadata_info_df.at[sample_uid, 'season'] == 'Winter':
                    df.iat[i,12] = "17.03.01"
                else:
                    df.iat[i,12] = "17.07.25"
        df.to_csv(path_or_buf=os.path.join(self.base_input_dir, 'restrepo_data_sheet_20190904_pop.csv'), index=False)


    def _make_remotely_sensed_sst_df(self):
        """This code is concerend with creating a df that contains the sst data for each of the sites as
        derived from the remotely sensed coral reef watch 5km data:
        ftp://ftp.star.nesdis.noaa.gov/pub/sod/mecb/crw/data/coraltemp/v1.0/nc/2017/
        """

        if os.path.exists(os.path.join(self.cache_dir, 'r_sst_df')):
            r_sst_df = pickle.load(open(os.path.join(self.cache_dir, 'r_sst_df'), 'rb'))
            for index, row in r_sst_df.iterrows():
                if row.isnull().values.any():
                    self._complete_r_sst_df(r_sst_df)
            return r_sst_df
        else:
            # dataframe to hold all of the remotely sensed sst data
            r_sst_df = pd.DataFrame(index=self.daily_temperature_av_df.index,
                                    columns=["Shib Nazar", 'Al Fahal', 'Qita al Kirsh', 'Tahla'])

            self._complete_r_sst_df(r_sst_df)
            return r_sst_df

    def _complete_r_sst_df(self, r_sst_df):
        data_lats, data_lons, site_to_lat_lon_data_tup_dict = self._make_closest_lat_lon_dict()
        self._populate_r_sst_df(data_lats, data_lons, r_sst_df, site_to_lat_lon_data_tup_dict)

    def _populate_r_sst_df(self, data_lats, data_lons, r_sst_df, site_to_lat_lon_data_tup_dict):
        # now go day by day to populate the r_sst_df
        for day_index in self.daily_temperature_av_df.index.values.tolist():
            print(f"Populating r_sst_df for {day_index}")
            # first check to see if this day is already populated
            if not r_sst_df.loc[day_index].isnull().values.any():
                continue
            # make the month and day in same format as in the nc files
            month, day, year = day_index.split('/')
            month = '0' + month
            if len(day) == 1:
                day = "0" + day

            nc_date = f'20{year}{month}{day}'
            file_name = f'coraltemp_v1.0_{nc_date}.nc'
            file_path = os.path.join(self.gis_input_base_path, 'crw', file_name)
            dataset = Dataset(file_path)

            dataset_df = pd.DataFrame(dataset.variables['analysed_sst'][0], index=data_lats, columns=data_lons)
            # for every site
            for site, tup in self.site_lat_long_tups.items():
                if site not in ["Shib Nazar", 'Al Fahal', 'Qita al Kirsh', 'Tahla']:
                    continue

                closest_var_lat, closest_var_lon = site_to_lat_lon_data_tup_dict[site]
                # now get the temperature array
                # for each day that we have temp datavailable
                sst = dataset_df.at[closest_var_lat, closest_var_lon]
                r_sst_df.at[day_index, site] = sst

            # pickle after everyday as this could represent quite a lot of work.
            pickle.dump(r_sst_df, open(os.path.join(self.cache_dir, 'r_sst_df'), 'wb'))

    def _make_closest_lat_lon_dict(self):
        # first make a dictionary that is the site to the closest lat and long value
        # we can just use one of the .nc files at random to get this info
        dataset = Dataset(os.path.join(self.gis_input_base_path, 'crw', "coraltemp_v1.0_20170101.nc"))
        # the lat and lon keys of the dataset
        data_lats = [float(_) for _ in list(dataset.variables['lat'])]
        data_lons = [float(_) for _ in list(dataset.variables['lon'])]
        site_to_lat_lon_data_tup_dict = {}
        for site, tup in self.site_lat_long_tups.items():
            if site not in ["Shib Nazar", 'Al Fahal', 'Qita al Kirsh', 'Tahla']:
                continue
            lat_of_site = tup[0]
            lon_of_site = tup[1]

            closest_var_lat = min(data_lats, key=lambda x: abs(x - lat_of_site))
            closest_var_lon = min(data_lons, key=lambda x: abs(x - lon_of_site))

            site_to_lat_lon_data_tup_dict[site] = (closest_var_lat, closest_var_lon)
        return data_lats, data_lons, site_to_lat_lon_data_tup_dict

    def make_networks(self):
        # The collections of ITS2 type profiles that we will make networks for.
        network_dict = {
            'M_1':[
                'A1k/A1', 'A1k/A1-A1ea', 'A1/A1k-A1b-A1z', 'A1/A1w-A1y'
            ],
            'SE_1':[
                'A1-A1x-A1r-A1u-A1g'
            ],
            'ST_1':[
                'A1-A1dl-A1bh', 'A1-A1m-A1z', 'A1-A1m', 'A1-A1m-A1n'
            ],
            'PC_1':[
                'A1-A1c-A1h-A1q-A1a', 'A1-A1c-A1h-A1i', 'A1-A1c-A1h-A1q', 'A1-A1c-A1k-A1h-A1q', 'A1-A1c-A1h-A1cv', 'A1-A1cc-A1c-A1h-A1q-A1i', 'A1/A1c-A1h', 'A1/A1c-A1h-A1ce'
            ],
            's_pist_2':[
                'A1/A1l/A1g-A1o-A1p', 'A1g/A1l/A1-A1o-A1cr-A1dp-A1p-A1dq-A1dn'
            ],
            'GX_1':[
                'C1/C39-C1b-C39a-C1af', 'C1-C1b-C39-C41-C1af-C1ae'
            ],
            'GX_2':[
                'C1-C1b-C39-C41-C1ae-C41f-C41a-C1af', 'C1-C1b-C41f-C41-C41a-C39-C41e-C1ae-C1f'
            ],
            'P_1':[
                'C15/C60a-C15b-C15e', 'C15', 'C15-C15by-C15ai', 'C15-C15y-C15df-C15x-C15a-C15w-C15z-C15aa-C15ab', 'C15-C15y-C15a-C15z-C15x-C15aa-C15v', 'C15-C15df-C15y-C15v-C15x-C15a-C15z-C15aa', 'C15-C15x-C15v-C15ab-C15u-C15c-C15d', 'C15-C15x-C15v-C15u-C15d-C15c-C15ab-C15dg'
            ],
            'P_2':[
                'C15/C22b/C15h', 'C22b/C15-C15a-C15an-C15de', 'C22b/C15-C15r', 'C15/C15r', 'C22b/C15-C15r-C15s', 'C15/C22b'
            ],
            'G_1':[
                'D4/D1/D1ab-D6', 'D1-D4-D6-D2b-D2a-D1d', 'D1/D4-D6-D6b', 'D1-D4-D6-D6b-D1d-D1i-D1j-D10', 'D1-D4-D6b-D6-D1d-D1q-D1j'
            ]
        }

        # Firstly go through each of the networks to make sure that the type names match up. I.e. check for typos.
        print('\n\nChecking for profile names in meta info')
        for profile_list in network_dict.values():
            for profile_name in profile_list:
                match_count = self.profile_meta_info_df['ITS2 type profile'].values.tolist().count(profile_name)
                if match_count > 1:
                    print(f'{profile_name} name found {match_count} times')
                elif match_count == 0:
                    print(f'{profile_name} name not found')
        print('Checking complete')

        # we want the network to have various characteristics.
        # The size of the nodes should represent the total abundance of that sequenece throughout all of the samples
        # whilst the greyscale colour of the node and the outline thickness
        # should represent the proportion of samples that the sequence was found in.

        # to make this happen we will need seq abundances for both the post-med and the pre-med


        # for each network in the network dict
        # Get a list of the samples that the profile was found in from the high cutoff df
        # Then have two default dicts for counting the number of samples given sequences are in
        # and for counting the relative abundance of those sequences within the samples (as a proportion of the type)
        # first one will be key is sequences, value is samples found in.
        # the second will be sequnces, the second will be cumulative relative abundance from samples
        # we should also have a master fasta that will hold the actual nucleotide sequences of the seqs in the net
        # Then, for each of these samples populate the default dicts. And populate the master fasta too.
        # at ths point we are ready to make a network!

        # for network in

    def investigate_background(self):
        """ This will be code associated with having an initial investigation of the low level ITS2 type profile
        1 - The first thing will be to look at how well defined the low level ITS2 type profiles"""
        # do a clade count to see how these profile instances are split over the clades



        dd_clade_count = defaultdict(int)
        div_count_distrb_list = []
        for profile_uid in self.profile_abundance_df_cutoff_background.columns:
            # get the name of the profile so that we can get the number of divs from that
            name = self.profile_meta_info_df.loc[profile_uid]['ITS2 type profile']
            div_count = len(re.split('-|/', name))
            ser = self.profile_abundance_df_cutoff_background[profile_uid]
            num_occurences = len(ser.iloc[ser.to_numpy().nonzero()].values.tolist())
            div_count_distrb_list.extend([div_count for _ in range(num_occurences)])
            dd_clade_count[self.profile_meta_info_df.at[profile_uid, 'Clade']] += num_occurences


        # Linerize the values in the df for passing to the hist
        f, axarr = plt.subplots(2, 3, figsize=(15, 9))

        ax_zero_second = axarr[0][0].twinx()
        sns.distplot(div_count_distrb_list, hist=False, kde=True,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=ax_zero_second, norm_hist=False)
        sns.distplot(div_count_distrb_list, hist=True, kde=False,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=axarr[0][0], norm_hist=False)

        # on the other plot plot the distribution of the number of divs found in types in above the 0.05 cutoff
        num_divs_list = []
        for profile_uid in self.profile_abundance_df:
            name = self.profile_meta_info_df.loc[profile_uid]['ITS2 type profile']
            div_count = len(re.split('-|/', name))
            ser = self.profile_abundance_df[profile_uid]
            ser = ser[ser>0.05]
            num_occurences = len(ser)
            num_divs_list.extend([div_count for _ in range(num_occurences)])

        ax_one_second = axarr[0][1].twinx()
        sns.distplot(num_divs_list, hist=False, kde=True,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=ax_one_second, norm_hist=False)
        sns.distplot(num_divs_list, hist=True, kde=False,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=axarr[0][1], norm_hist=False)
        ax_one_second.set_ylim(0,0.25)
        ax_zero_second.set_ylim(0,0.25)
        foo = 'bar'

        # we want to look at the correlation of aundandace of the background ITS2 type profiles versus the high
        # cutoff profiles.
        # I want to visualise this using a scatter plot.
        # this should compare whether the background profile abundances are found in both the high profile and the low profile
        back_sample_num = []
        high_low_sample_num = []
        found_back_not_high_low = 0
        found_high_low_not_back = 0
        found_both = 0
        back_tot = len(self.profile_abundance_df_cutoff_background.columns.values.tolist())
        high_low_tot = len(set(self.profile_abundance_df_cutoff_high.columns.values.tolist() + self.profile_abundance_df_cutoff_low.columns.values.tolist()))
        for profile_uid_back in self.profile_abundance_df_cutoff_background.columns:
            ser_back = self.profile_abundance_df_cutoff_background[profile_uid_back]
            num_occurences_back = len(ser_back.iloc[ser_back.to_numpy().nonzero()].values.tolist())
            back_sample_num.append(num_occurences_back)
            if profile_uid_back in self.profile_abundance_df_cutoff_high or profile_uid_back in self.profile_abundance_df_cutoff_low:
                found_both += 1
                num_occurences_high = 0
                num_occurences_low = 0
                if profile_uid_back in self.profile_abundance_df_cutoff_high:
                    ser_high = self.profile_abundance_df_cutoff_high[profile_uid_back]
                    num_occurences_high = len(ser_high.iloc[ser_high.to_numpy().nonzero()].values.tolist())
                if profile_uid_back in self.profile_abundance_df_cutoff_low:
                    ser_low = self.profile_abundance_df_cutoff_low[profile_uid_back]
                    num_occurences_low = len(ser_low.iloc[ser_low.to_numpy().nonzero()].values.tolist())
                num_occurences_high_low = num_occurences_high + num_occurences_low
                high_low_sample_num.append(num_occurences_high_low)
            else:
                found_back_not_high_low += 1
                high_low_sample_num.append(0)
                if num_occurences_back > 20:
                    profile_name = self.profile_meta_info_df.at[profile_uid_back, 'ITS2 type profile']
                    local = self.profile_meta_info_df.at[profile_uid_back, 'ITS2 type abundance local']
                    global_abund = self.profile_meta_info_df.at[profile_uid_back, 'ITS2 type abundance DB']
                    print(f'interesting profile {profile_name} that was found in the background collection at an abundance of {num_occurences_back} and was found at an abundance of 0 in the high, had a local abundance of {local} and a global abundance of {global_abund}')

        # now populate with the uids that were found in the high abundance but not in the
        for uid_high_low in set(self.profile_abundance_df_cutoff_high.columns.values.tolist() + self.profile_abundance_df_cutoff_low.columns.values.tolist()):
            if uid_high_low not in self.profile_abundance_df_cutoff_background:
                num_occurences_high = 0
                num_occurences_low = 0
                if uid_high_low in self.profile_abundance_df_cutoff_high:
                    ser_high = self.profile_abundance_df_cutoff_high[uid_high_low]
                    num_occurences_high = len(ser_high.iloc[ser_high.to_numpy().nonzero()].values.tolist())
                if uid_high_low in self.profile_abundance_df_cutoff_low:
                    ser_low = self.profile_abundance_df_cutoff_low[uid_high_low]
                    num_occurences_low = len(ser_low.iloc[ser_low.to_numpy().nonzero()].values.tolist())
                num_occurences_high_low = num_occurences_high + num_occurences_low
                high_low_sample_num.append(num_occurences_high_low)
                back_sample_num.append(0)
                found_high_low_not_back += 1


        back = 0
        high_low = 0
        common = 0
        for back_abund, high_low_abund in zip(back_sample_num, high_low_sample_num):
            if back_abund !=0 and high_low_abund != 0:
                common += (back_abund + high_low_abund)
            elif high_low_abund != 0:
                high_low += high_low_abund
            else:
                back += back_abund

        venn2(subsets=(back, high_low, common),
              set_labels=('background', 'non-background'), ax=axarr[1][2])

        # venn2(subsets=(found_back_not_high_low,found_high_low_not_back ,found_both), set_labels=('background','non-background'), ax=axarr[1][2])

        axarr[0][2].scatter(x=back_sample_num, y=high_low_sample_num, marker='o', color='black', s=20, alpha=0.1)
        axarr[0][2].set_xlabel('background_profile_abundances')
        axarr[0][2].set_ylabel('high_profile_abundances')
        axarr[0][2].set_ylim(-2,50)
        axarr[0][2].set_xlim(-2,50)
        f.tight_layout()

        # how many of the profiles from the background were found in the high
        print(f'{back_tot-found_back_not_high_low} out of {back_tot} ({(back_tot-found_back_not_high_low)/back_tot}) of the low profiles were found in the high.')
        # how many of the high were found in the background
        print(f'{high_low_tot - found_high_low_not_back} out of {high_low_tot} ({(high_low_tot - found_high_low_not_back) / high_low_tot}) of the high profiles were found in the low.')


        # do regressions of number of samples found in versus the number of species they were found in
        # for both the background and high collections
        x_back = []
        y_back = []
        x_high = []
        y_high = []

        for uid_back in self.profile_abundance_df_cutoff_background.columns:
            ser_back = self.profile_abundance_df_cutoff_background[uid_back]
            ser_back_non_zero_ind = ser_back[ser_back > 0].index.values.tolist()
            list_of_speceies = [self.experimental_metadata_info_df.at[smp_uid, 'species'] for smp_uid in ser_back_non_zero_ind]
            set_of_species = set(list_of_speceies)
            x_back.append(len(list_of_speceies))
            y_back.append(len(set_of_species))

        for uid_high_low in self.profile_abundance_df_cutoff_high.columns:
            ser_high = self.profile_abundance_df_cutoff_high[uid_high_low]
            ser_high_non_zero_ind = ser_high[ser_high > 0].index.values.tolist()
            list_of_speceies = [self.experimental_metadata_info_df.at[smp_uid, 'species'] for smp_uid in ser_high_non_zero_ind]
            set_of_species = set(list_of_speceies)
            x_high.append(len(list_of_speceies))
            y_high.append(len(set_of_species))


        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=x_back, y=y_back)
        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = scipy.stats.linregress(x=x_high, y=y_high)

        # # I want to see what the local to global ratios look like for these background ratios.
        # # a good way to do this is probably to plot the number of samples the profile was found in against
        # # the local to global ratio
        # x_local_global = []
        # y_local_global = []
        # for profile_uid_back in self.profile_abundance_df_cutoff_background.columns:
        #     ser_back = self.profile_abundance_df_cutoff_background[profile_uid_back]
        #     num_occurences_back = len(ser_back.iloc[ser_back.to_numpy().nonzero()].values.tolist())
        #     low_sample_num.append(num_occurences_back)
        #     local = self.profile_meta_info_df.at[profile_uid_back, 'ITS2 type abundance local']
        #     global_abund = self.profile_meta_info_df.at[profile_uid_back, 'ITS2 type abundance DB']
        #     x_local_global.append(num_occurences_back)
        #     y_local_global.append(local/global_abund)
        #
        # axarr[1][0].scatter(x=x_local_global, y=y_local_global, marker='o', color='black', s=20, alpha=0.1)
        # axarr[1][0].set_xlabel('number 0f samples found in')
        # axarr[1][0].set_ylabel('local/global ratio')
        # axarr[1][0].set_ylim(-0.02, 1.02)
        # # axarr[1][0].set_xlim(-0.02, 1.02)
        #
        # # now compare this to the high abundance set of profiles
        # # I want to see what the local to global ratios look like for these background ratios.
        # # a good way to do this is probably to plot the number of samples the profile was found in against
        # # the local to global ratio
        # x_local_global = []
        # y_local_global = []
        # for profile_uid_back in self.profile_abundance_df_cutoff_high.columns:
        #     ser_back = self.profile_abundance_df_cutoff_high[profile_uid_back]
        #     num_occurences_back = len(ser_back.iloc[ser_back.to_numpy().nonzero()].values.tolist())
        #     low_sample_num.append(num_occurences_back)
        #     local = self.profile_meta_info_df.at[profile_uid_back, 'ITS2 type abundance local']
        #     global_abund = self.profile_meta_info_df.at[profile_uid_back, 'ITS2 type abundance DB']
        #     x_local_global.append(num_occurences_back)
        #     y_local_global.append(local / global_abund)
        #
        # axarr[1][1].scatter(x=x_local_global, y=y_local_global, marker='o', color='black', s=20, alpha=0.1)
        # axarr[1][1].set_xlabel('number 0f samples found in')
        # axarr[1][1].set_ylabel('local/global ratio')
        # axarr[1][1].set_ylim(-0.02, 1.02)
        # # axarr[1][1].set_xlim(-0.02, 1.02)

        # I want to plot the number of sequences found against the percentage of the sample the clade collection
        # made up. We will do this for all clade collections less than 10%
        # we also want to see if the given minor clade collection had a profile found in it
        x_rel_abund_yes = []
        y_abs_seqs_yes = []
        x_rel_abund_no = []
        y_abs_seqs_no = []
        x_rel_abund = []
        y_abs_seqs = []
        smp_count = 0
        if os.path.exists(os.path.join(self.cache_dir, 'x_rel_abund_yes')):
            x_rel_abund_yes = pickle.load(open(os.path.join(self.cache_dir, 'x_rel_abund_yes'), 'rb'))
            y_abs_seqs_yes = pickle.load(open(os.path.join(self.cache_dir, 'y_abs_seqs_yes'), 'rb'))
            x_rel_abund_no = pickle.load(open(os.path.join(self.cache_dir, 'x_rel_abund_no'), 'rb'))
            y_abs_seqs_no = pickle.load(open(os.path.join(self.cache_dir, 'y_abs_seqs_no'), 'rb'))
            x_rel_abund = pickle.load(open(os.path.join(self.cache_dir, 'x_rel_abund'), 'rb'))
            y_abs_seqs = pickle.load(open(os.path.join(self.cache_dir, 'y_abs_seqs'), 'rb'))
        else:
            for smp_uid in self.seq_meta_data_df.index:
                smp_count += 1
                print(f'checking {smp_count}')
                total_seqs = self.seq_meta_data_df.at[smp_uid, 'post_med_absolute']
                clade_props_ser = self.clade_proportion_df_non_normalised.loc[smp_uid]
                max_clade = clade_props_ser.idxmax()
                for clade, clade_prop in clade_props_ser.items():
                    if clade_prop > 0 and clade != max_clade:
                        x_rel_abund.append(clade_prop)
                        y_abs_seqs.append(clade_prop * total_seqs)
                        # now check to see if the clade collection in question had a profile found in it
                        has = []
                        for df in [self.profile_abundance_df_cutoff_background, self.profile_abundance_df_cutoff_low, self.profile_abundance_df_cutoff_high]:
                            has.append(self._check_if_cc_has_prof(clade, smp_uid, df=df))
                        if True in has:
                            if has[1] or has[2]:
                                foo = 'asdf'
                            x_rel_abund_yes.append(clade_prop)
                            y_abs_seqs_yes.append(clade_prop*total_seqs)
                        else:
                            x_rel_abund_no.append(clade_prop)
                            y_abs_seqs_no.append(clade_prop * total_seqs)
            pickle.dump(x_rel_abund_yes, open(os.path.join(self.cache_dir, 'x_rel_abund_yes'), 'wb'))
            pickle.dump(y_abs_seqs_yes, open(os.path.join(self.cache_dir, 'y_abs_seqs_yes'), 'wb'))
            pickle.dump(x_rel_abund_no, open(os.path.join(self.cache_dir, 'x_rel_abund_no'), 'wb'))
            pickle.dump(y_abs_seqs_no, open(os.path.join(self.cache_dir, 'y_abs_seqs_no'), 'wb'))
            pickle.dump(x_rel_abund, open(os.path.join(self.cache_dir, 'x_rel_abund'), 'wb'))
            pickle.dump(y_abs_seqs, open(os.path.join(self.cache_dir, 'y_abs_seqs'), 'wb'))

        below_200_prop = len([_ for _ in y_abs_seqs if _ < 200]) / len(y_abs_seqs)
        axarr[1][1].scatter(x=x_rel_abund_yes, y=y_abs_seqs_yes, marker='o', color='green', s=20, alpha=0.1, zorder=3)
        axarr[1][1].scatter(x=x_rel_abund_no, y=y_abs_seqs_no, marker='o', color='black', s=20, alpha=0.1, zorder=3)
        axarr[1][1].set_xlabel('relative abundance of clade collection')
        axarr[1][1].set_ylabel('absolute sequences of clade collection')
        axarr[1][1].hlines(y=200, xmin=axarr[1][1].get_xlim()[0], xmax=axarr[1][1].get_xlim()[1], colors='red')
        axarr[1][1].text(x=-0.015, y=-3000, s=f'{below_200_prop:.2f}')


        # work out the percentage of the sequences that these bleow 200 clade collections represented
        non_screened_seqs = 0
        for rel_abund, abs_abund in zip(x_rel_abund, y_abs_seqs):
            if abs_abund < 200:
                non_screened_seqs += abs_abund
        prop_of_low_seq_cct = non_screened_seqs/sum(y_abs_seqs)
        axarr[1][0].scatter(x=x_rel_abund, y=y_abs_seqs, marker='o', color='black', s=20, alpha=0.1, zorder=3)
        axarr[1][0].set_xlabel('relative abundance of clade collection')
        axarr[1][0].set_ylabel('absolute sequences of clade collection')
        this = axarr[1][0].get_xlim()
        axarr[1][0].hlines(y=200, xmin=axarr[1][0].get_xlim()[0], xmax=axarr[1][0].get_xlim()[1], colors='red')
        axarr[1][0].text(x=-0.015, y=-3000, s=f'{below_200_prop:.2f}')
        # axarr[1][1].set_ylim(-0.02, 1.02)

        # now look to see if the profiles were found in the minor or maj clade collections of a given sample
        maj_cc_list = []
        min_cc_list = []
        for prof_uid in self.profile_abundance_df_cutoff_background:
            # for each sample that the profile was found in
            ser = self.profile_abundance_df_cutoff_background[prof_uid]
            ser = ser[ser>0]
            for smp_uid, value in ser.items():
                max_clade = self.clade_proportion_df_non_normalised.loc[smp_uid].idxmax()
                clade_of_profile = self.profile_meta_info_df.at[prof_uid, 'Clade']
                if max_clade == clade_of_profile:
                    maj_cc_list.append(value)
                else:
                    min_cc_list.append(value)

        print(f'{len(maj_cc_list)} of the profiles were found in maj {len(min_cc_list)} were found in minor')

        multi_smp_list = []
        single_smp_list = []
        no_smp_list = []
        tot = 0
        for smp_uid in self.profile_abundance_df_cutoff_background.index:
            clade_props_ser = self.clade_proportion_df_non_normalised.loc[smp_uid]
            max_clade = clade_props_ser.idxmax()
            for clade in clade_props_ser.index:
                if clade == max_clade or clade_props_ser[clade] == 0:
                    continue
                # get number of profiles of this sample in the background and of the clade
                tot += 1
                prof_abund_ser = self.profile_abundance_df_cutoff_background.loc[smp_uid]
                prof_abund_ser = prof_abund_ser[prof_abund_ser > 0]
                minor_prof_of_clade_list = [value for prof_uid, value in prof_abund_ser.items() if self.profile_meta_info_df.at[prof_uid, 'Clade'] == clade]
                if minor_prof_of_clade_list:
                    if len(minor_prof_of_clade_list) > 1:
                        multi_smp_list.append(smp_uid)
                    else:
                        single_smp_list.append(smp_uid)
                else:
                    no_smp_list.append(smp_uid)

        print(f'{len(multi_smp_list)} clade collections had multiple, {len(single_smp_list)} clade collections had single. tot was {tot}. non list was {len(no_smp_list)}')


        # want to show the disproportionate number of A, C, D instances in the back ground vs. the others.
        # first get number of instances for high and low and then for background
        high_low_dd_dict_profile_instances = defaultdict(int)
        back_dd_dict_profile_instances = defaultdict(int)
        for profile_uid in self.profile_abundance_df_cutoff_high:
            ser = self.profile_abundance_df_cutoff_high[profile_uid]
            ser = ser[ser>0]
            clade_of_profile = self.profile_meta_info_df.at[profile_uid, 'Clade']
            count_of_profile = len(ser.values.tolist())
            high_low_dd_dict_profile_instances[clade_of_profile] += count_of_profile

        for profile_uid in self.profile_abundance_df_cutoff_low:
            ser = self.profile_abundance_df_cutoff_low[profile_uid]
            ser = ser[ser>0]
            clade_of_profile = self.profile_meta_info_df.at[profile_uid, 'Clade']
            count_of_profile = len(ser.values.tolist())
            high_low_dd_dict_profile_instances[clade_of_profile] += count_of_profile

        for profile_uid in self.profile_abundance_df_cutoff_background:
            ser = self.profile_abundance_df_cutoff_background[profile_uid]
            ser = ser[ser>0]
            clade_of_profile = self.profile_meta_info_df.at[profile_uid, 'Clade']
            count_of_profile = len(ser.values.tolist())
            back_dd_dict_profile_instances[clade_of_profile] += count_of_profile

        print(high_low_dd_dict_profile_instances)
        print(back_dd_dict_profile_instances)

        for dd_dict, dd_dict_name in zip([high_low_dd_dict_profile_instances, back_dd_dict_profile_instances], ['high_low', 'background']):
            for clade in list('ACD'):
                prop = dd_dict[clade]/sum(dd_dict.values())
                print(f'For {dd_dict_name}, clade {clade} proportion was {prop}')

        # finally need to compare this to the clade ratios of the minor clade collections
        minor_cc_dd_dict = defaultdict(int)
        for smp_uid in self.clade_proportion_df_non_normalised.index:
            clade_absolute_abundances = self.clade_proportion_df_non_normalised.loc[smp_uid]*self.seq_meta_data_df.at[smp_uid, 'post_med_absolute']
            max_clade = self.clade_proportion_df_non_normalised.loc[smp_uid].idxmax()
            for clade in clade_absolute_abundances.index:
                if clade_absolute_abundances[clade] > 200 and clade != max_clade:
                    minor_cc_dd_dict[clade] += 1
        print(f'\n{minor_cc_dd_dict}')
        print(minor_cc_dd_dict)

        for clade in list('ACD'):
            prop = minor_cc_dd_dict[clade] / sum(minor_cc_dd_dict.values())
            print(f'For minor_dd_dict, clade {clade} proportion was {prop}')

        f.savefig(os.path.join(self.figure_dir, 'background_profiles_instances.png'), dpi=1200)
        f.savefig(os.path.join(self.figure_dir, 'background_profiles_instances.svg'), dpi=1200)
        foo = 'bar'

        # colour the points by number of codom seqs in the profile (on averge) and

    def _check_if_cc_has_prof(self, clade, smp_uid, df):
        ser_back = df.loc[smp_uid]
        ser_back = ser_back[ser_back > 0]
        prof_of_clade_list = [value for prof_uid, value in ser_back.items() if
                              self.profile_meta_info_df.at[prof_uid, 'Clade'] == clade]
        if prof_of_clade_list:
            return True
        else:
            return False

    def report_on_reef_type_effect_metrics(self):
        """This will report on the proportion of ITS2 type profile instances that were found in both reefs belonging
        to a given reef type divided by the total number of instances for those two reef types. We will use this metric
        to investigate whether the inshore reef type had a more specialist set of ITS2 type profiles compared to the
        other reef types"""
        x_y_df = pd.DataFrame(columns=['num_samples', 'num_reefs', 'num_reef_types'], index=self.profile_abundance_df_cutoff_high.columns, dtype='float')
        for reef_type in self.reef_types:
            total = 0
            numerator = 0
            for profile_uid in self.profile_abundance_df_cutoff_high.columns:
                # get a set of the reefs that this profile was found in. If the set does not contain any
                # of the reefs from the oother reef tyeps, we will add the occurences to the numerator
                # we will also keep track of the total. Then divide.
                temp_series = self.profile_abundance_df_cutoff_high[profile_uid]
                temp_series_non_zero_series = temp_series[temp_series > 0]
                smpl_index_list = temp_series_non_zero_series.index.values.tolist()
                reefs = [self.experimental_metadata_info_df.at[smpl_uid, 'reef'] for smpl_uid in smpl_index_list]
                reef_types = [self.experimental_metadata_info_df.at[smpl_uid, 'reef_type'] for smpl_uid in smpl_index_list]
                reefs_set = set(reefs)
                if self._contains_other_reef_types(reef_type=reef_type, reefs_set=reefs_set):
                    # Then this does not get added to the numerator, only the denominator
                    # Also we should only be adding according to the number of type profiles that were in samples
                    # of the reef_type in question
                    total += reef_types.count(reef_type)
                else:
                    total += len(smpl_index_list)
                    numerator += len(smpl_index_list)
                x_y_df.at[profile_uid, 'num_samples'] = len(smpl_index_list)
                x_y_df.at[profile_uid, 'num_reefs'] = len(reefs_set)
                x_y_df.at[profile_uid, 'num_reef_types'] = len(set(reef_types))
            print(f'For reef type {reef_type}: the specificity metric was {numerator/total:.2f}')

        # The second thing to look at is the correlation between the number of samples a given ITS2 type profile was
        # found in and the number of different reefs it was found in

        # we want to create x and y variables where number of reefs is y and the number of samples a given
        # ITS2 type profile was found in is x. We will collect X and Y above to prevent us from having
        # to run essentially the same loop structures again.
        print(x_y_df.dtypes)
        lm = linear_model.LinearRegression()
        X = x_y_df['num_samples'].values.reshape(-1,1)
        y = x_y_df['num_reefs'].values.reshape(-1,1)
        model = lm.fit(X,y)
        score = lm.score(X,y)
        print(model)
        print(score)
        print(f'The R2 value for number of samples predicting number of reefs was: {score}')



        X = x_y_df['num_samples']
        y = x_y_df['num_reefs']
        y2 = x_y_df['num_reef_types']
        this = len(X)
        that = len(y)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=X, y=y)
        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = scipy.stats.linregress(x=X, y=y2)

        foo = 'bar'

    def _contains_other_reef_types(self, reef_type, reefs_set):
        """Checks to see if a set contains reefs that are from other reef types than the one provided"""
        if reef_type == 'Inshore':
            if len(list(reefs_set & {'Qita al Kirsh', 'Al Fahal', 'Shib Nazar', 'Abu Madafi'})) >= 1 or len(reefs_set) == 1:
                return True
        elif reef_type == 'Midshelf':
            if len(list(reefs_set & {'Fsar', 'Tahla', 'Shib Nazar', 'Abu Madafi'})) >= 1 or len(reefs_set) == 1:
                return True
        elif reef_type == 'Offshore':
            if len(list(reefs_set & {'Fsar', 'Tahla', 'Qita al Kirsh', 'Al Fahal'})) >= 1 or len(reefs_set) == 1:
                return True
        return False

    def report_on_fidelity_proxies_for_profile_associations(self):
        """Report the stats for the two its2 type profile schematics. Looks at how many types for each of the
        schematics, the number of associations represented. And then report on the average number of different
        groups per profile per factor."""

        # stats for pre-cutoff
        num_of_profiles_pre_cutoff = len(list(self.profile_abundance_df))
        num_associations_pre_cutoff = len(
            list(self.profile_abundance_df[self.profile_abundance_df > 0].stack().index))

        # stats for post-cutoff high
        num_of_profiles_post_cutoff_high = len(list(self.profile_abundance_df_cutoff_high))
        num_associations_post_cutoff_high = len(
            list(self.profile_abundance_df_cutoff_high[self.profile_abundance_df_cutoff_high > 0].stack().index))

        # stats for post-cutoff low
        num_of_profiles_post_cutoff_low = len(list(self.profile_abundance_df_cutoff_low))
        num_associations_post_cutoff_low = len(
            list(self.profile_abundance_df_cutoff_low[self.profile_abundance_df_cutoff_low > 0].stack().index))


        print(f'The total number of predicted ITS2 type profiles was {num_of_profiles_pre_cutoff} representing {num_associations_pre_cutoff} associations.')
        print(f'After screening for >0.40 there were {num_of_profiles_post_cutoff_high} profiles representing {num_associations_post_cutoff_high} associations.')
        print(f'After screening for <0.40 but >0.05 there were {num_of_profiles_post_cutoff_low} profiles representing {num_associations_post_cutoff_low} associations.')

        # for the high
        print('\n\n')
        abund_df_to_work_from = self.profile_abundance_df_cutoff_high
        df_high = self._get_df_of_average_unique_groups_per_profile_per_factor(abund_df_to_work_from)

        self._report_on_fidelity(df_high, high_low='high')
        print('\n\n')
        # for the low
        abund_df_to_work_from = self.profile_abundance_df_cutoff_low
        df_low = self._get_df_of_average_unique_groups_per_profile_per_factor(abund_df_to_work_from)

        self._report_on_fidelity(df_low, high_low='low')

        # for the background
        abund_df_to_work_from = self.profile_abundance_df_cutoff_background
        df_background = self._get_df_of_average_unique_groups_per_profile_per_factor(abund_df_to_work_from)
        print('\n\n')
        self._report_on_fidelity(df_background, high_low='background')

        # stats for high vs low
        print('\n\nRunning stats for high vs low:')
        for factor in self.experimental_metadata_info_df.columns:
            stat, p = scipy.stats.mannwhitneyu(df_high[factor], df_low[factor])
            print(f'{factor}: {stat}: {p}')

        # stats for high vs background
        print('\n\nRunning stats for high vs low:')
        for factor in self.experimental_metadata_info_df.columns:
            stat, p = scipy.stats.mannwhitneyu(df_high[factor], df_background[factor])
            print(f'{factor}: {stat}: {p}')

        # I think it would also be really useful to know what proportion of the overall sample abundances were accounted for by the lower upper and discounted ITS2 type profile abundances
        total = sum(self.profile_abundance_df.sum())
        total_high = sum(self.profile_abundance_df_cutoff_high.sum())
        high_perc = total_high/total*100
        total_low = sum(self.profile_abundance_df_cutoff_low.sum())
        low_perc = total_low/total*100
        total_discount = total-total_high-total_low
        discount_perc = total_discount/total*100

        print(f'The high cutoff collection of ITS2 type profile instances represented {high_perc:.2f}% of the sequences')
        print(
            f'The low cutoff collection of ITS2 type profile instances represented {low_perc:.2f}% of the sequences')
        print(
            f'The instances discounted within the <0.05 cutoff represented {discount_perc:.2f}% of the sequences')
        foo = 'bar'

    def _report_on_fidelity(self, df, high_low):
        # now it just remains to work out the averages for each of the factor columns
        print(f'For the {high_low} cutoff')
        for factor in self.experimental_metadata_info_df.columns:
            # temp_ser = df[factor]
            # non_one = temp_ser[temp_ser > 1]
            # percent = len(non_one)/len(temp_ser) * 100
            # print(f'{factor}: {percent}% of profiles had > 1 group')
            av_val = df[factor].mean()
            print(f'{factor}: {av_val} average unique groups')

    def _get_df_of_average_unique_groups_per_profile_per_factor(self, abund_df_to_work_from):
        # df = pd.DataFrame(columns=self.experimental_metadata_info_df.columns,
        #                   index=self.profile_abundance_df_cutoff_high.columns)
        dict_for_df = {}
        for type_uid in abund_df_to_work_from.columns:
            temp_series = abund_df_to_work_from[type_uid]
            temp_series_non_zero_series = temp_series[temp_series > 0]
            smp_uid_list = temp_series_non_zero_series.index.values.tolist()
            if len(smp_uid_list) > 1:
                temp_list = []
                for factor in self.experimental_metadata_info_df.columns:
                    unique_groups = set(
                        [self.experimental_metadata_info_df.loc[smp_uid, factor] for smp_uid in smp_uid_list])
                    temp_list.append(len(unique_groups))
                    # df.at[type_uid, factor] = len(unique_groups)
                dict_for_df[type_uid] = temp_list
        df = pd.DataFrame.from_dict(dict_for_df, columns=self.experimental_metadata_info_df.columns, orient='index')
        return df

    def _create_profile_distance_dict_per_clade_cutoff_high_low(self, profile_abundance_df, high_low):
        """This should load in the """

        for clade in self.clades:
            # Uids of abundance dict
            prof_uids_to_drop = set([ind for ind in profile_abundance_df.index if self.profile_meta_info_df[ind, 'Clade'] == clade]) ^ set(self.between_profile_clade_dist_df_dict[clade].index.values.tolist())
            prof_dists_clade_df = self.between_profile_clade_dist_df_dict[clade].copy()
            prof_dists_clade_df = prof_dists_clade_df.drop(prof_uids_to_drop, axis=1).drop(prof_uids_to_drop, axis=0)

    def _output_clade_prop_df_no_se(self):
        clade_prop_df_no_se = self._remove_se_from_df(self.between_sample_clade_proportion_distances_df)
        output_path_dist_matrix = os.path.join(self.outputs_dir,
                                               f'between_sample_clade_proportion_distances_no_se.csv')
        clade_prop_df_no_se.to_csv(path_or_buf=output_path_dist_matrix, sep=',', header=False, index=False,
                                   line_terminator='\n')
        meta_info_df_for_clade = self.experimental_metadata_info_df.loc[clade_prop_df_no_se.index.values.tolist(), :]
        output_path_meta_info = os.path.join(self.outputs_dir, f'sample_meta_for_clade_proportion_permanova_no_se.csv')
        meta_info_df_for_clade.to_csv(path_or_buf=output_path_meta_info, sep=',', header=True, index=False,
                                      line_terminator='\n')

    def _init_sp_output_paths(self):

        # Paths to profile count tables
        self.profile_rel_abund_ouput_path = os.path.join(self.base_sp_output_dir, 'its2_type_profiles', '77_DBV_20190721_2019-08-06_09-21-49.148787.profiles.relative.abund_and_meta.txt')
        self.profile_abs_abund_ouput_path = os.path.join(self.base_sp_output_dir, 'its2_type_profiles', '77_DBV_20190721_2019-08-06_09-21-49.148787.profiles.absolute.abund_and_meta.txt')

        # Paths to seq count tables post_med
        self.seq_rel_abund_post_med_ouput_path = os.path.join(self.base_sp_output_dir, 'post_med_seqs', '77_DBV_20190721_2019-08-06_09-21-49.148787.seqs.relative.abund_only.txt')
        self.seq_abs_abund_post_med_ouput_path = os.path.join(self.base_sp_output_dir, 'post_med_seqs', '77_DBV_20190721_2019-08-06_09-21-49.148787.seqs.absolute.abund_and_meta.txt')

        # Paths to seq count tables pre_med
        self.seq_rel_abund_pre_med_ouput_path_list = [os.path.join(self.base_sp_output_dir, 'pre_med_seqs',
                                                              f'pre_med_relative_abundance_df_{uid}.csv') for uid in ['341', '349', '350']]


        # Paths to the standard output profile distance files braycurtis derived
        self.between_profile_clade_braycurtis_dist_path_dict = {
            'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis', 'A', '2019-07-21_01-14-05.826803.bray_curtis_within_clade_profile_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis', 'C', '2019-07-21_01-14-05.826803.bray_curtis_within_clade_profile_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis', 'D', '2019-07-21_01-14-05.826803.bray_curtis_within_clade_profile_distances_D.dist')}

        # Paths to the standard output profile distance files unifrac derived
        self.between_profile_clade_unifrac_dist_path_dict = {
            'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac', 'A',
                              '2019-08-06_09-21-49.148787_unifrac_btwn_profile_distances_A_fixed.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac', 'C',
                              '2019-08-06_09-21-49.148787_unifrac_btwn_profile_distances_C_fixed.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac', 'D',
                              '2019-08-06_09-21-49.148787_unifrac_btwn_profile_distances_D_fixed.dist')}


        # Paths to the cct specific distances at 0.05 with braycurtis
        self.between_profile_clade_dist_cct_005_braycurtis_specific_path_dict = {
            'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis_cct_005', 'A', '2019-07-21_01-14-05.826803.bray_curtis_within_clade_profile_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis_cct_005', 'C', '2019-07-21_01-14-05.826803.bray_curtis_within_clade_profile_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis_cct_005', 'D', '2019-07-21_01-14-05.826803.bray_curtis_within_clade_profile_distances_D.dist')
        }

        # Paths to the cct specific distances at < 0.05 with unifrac
        self.between_profile_clade_dist_cct_background_unifrac_specific_path_dict = {
            'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_background', 'A',
                              '2019-08-25_05-16-01.568715_unifrac_btwn_profile_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_background', 'C',
                              '2019-08-25_05-16-01.568715_unifrac_btwn_profile_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_background', 'D',
                              '2019-08-25_05-16-01.568715_unifrac_btwn_profile_distances_D.dist')
        }

        # Paths to the cct specific distances at abundances between 0.05 and 0.40 unifrac
        self.between_profile_clade_dist_cct_low_unifrac_specific_path_dict = {
            'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_low', 'A',
                              '2019-08-14_06-06-38.037792_unifrac_btwn_profile_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_low', 'C',
                              '2019-08-14_06-06-38.037792_unifrac_btwn_profile_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_low', 'D',
                              '2019-08-14_06-06-38.037792_unifrac_btwn_profile_distances_D.dist')
        }

        # Paths to the cct specific distances at 0.40 with braycurtis
        self.between_profile_clade_dist_cct_040_braycurtis_specific_path_dict = {
        'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis_cct_040', 'A', '2019-08-07_07-39-36.704269.bray_curtis_within_clade_profile_distances_A.dist'),
        'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis_cct_040', 'C', '2019-08-07_07-39-36.704269.bray_curtis_within_clade_profile_distances_C.dist'),
        'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_braycurtis_cct_040', 'D', '2019-08-07_07-39-36.704269.bray_curtis_within_clade_profile_distances_D.dist')
        }

        # Paths to the cct specific distances at 0.05 with unifrac
        self.between_profile_clade_dist_cct_005_unifrac_specific_path_dict = {
            'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_005', 'A', '2019-08-07_07-28-12.765828_unifrac_btwn_profile_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_005', 'C', '2019-08-07_07-28-12.765828_unifrac_btwn_profile_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_005', 'D', '2019-08-07_07-28-12.765828_unifrac_btwn_profile_distances_D.dist')
        }

        # Paths to the cct specific distances at 0.40 with unifrac
        self.between_profile_clade_dist_cct_040_unifrac_specific_path_dict = {
            'A': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_040', 'A', '2019-08-07_07-41-48.044287_unifrac_btwn_profile_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_040', 'C', '2019-08-07_07-41-48.044287_unifrac_btwn_profile_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_profile_distances_unifrac_cct_040', 'D', '2019-08-07_07-41-48.044287_unifrac_btwn_profile_distances_D.dist')
        }

        # Paths to the sample distances (BrayCurtis sqrt transformed)
        self.between_sample_clade_dist_path_dict_braycurtis = {
            'A': os.path.join(self.base_sp_output_dir, 'between_sample_distances_braycurtis', 'A', '2019-08-07_00-18-03.168246.bray_curtis_sample_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_sample_distances_braycurtis', 'C', '2019-08-07_00-18-03.168246.bray_curtis_sample_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_sample_distances_braycurtis', 'D', '2019-08-07_00-18-03.168246.bray_curtis_sample_distances_D.dist')
        }

        # Paths to the smpl distance (Unifrac sqrt transformed abundance)
        self.between_sample_clade_dist_path_dict_unifrac = {
            'A': os.path.join(self.base_sp_output_dir, 'between_sample_distances_unifrac', 'A', '2019-08-06_09-21-49.148787_unifrac_btwn_sample_distances_A.dist'),
            'C': os.path.join(self.base_sp_output_dir, 'between_sample_distances_unifrac', 'C', '2019-08-06_09-21-49.148787_unifrac_btwn_sample_distances_C.dist'),
            'D': os.path.join(self.base_sp_output_dir, 'between_sample_distances_unifrac', 'D', '2019-08-06_09-21-49.148787_unifrac_btwn_sample_distances_D.dist')
        }

    def _populate_seq_meta_data_df(self):
        """This method will produce a dataframe that has sample UID as the key and the QC metadata items as the
        columns"""
        if os.path.exists((os.path.join(self.cache_dir, 'seq_meta_data_df.p'))):
            return pickle.load(open(os.path.join(self.cache_dir, 'seq_meta_data_df.p'), 'rb'))
        else:
            df = pd.read_csv(filepath_or_buffer=self.seq_abs_abund_post_med_ouput_path, sep='\t', header=0)
            # columns = ['sample_uid'] + df.columns.values.tolist()[1:]
            # df.columns = columns
            df = df.iloc[:-1,:]
            df['sample_uid'] = pd.to_numeric(df['sample_uid'])
            df.set_index('sample_uid', drop=True, inplace=True)
            pickle.dump(df, open(os.path.join(self.cache_dir, 'seq_meta_data_df.p'), 'wb'))
            return df

    def _quaternary_plot(self):
        class QuantPlot:
            def __init__(self, parent):
                self.parent = parent
                # first port of call is to see what the maximum number of types is
                dd = defaultdict(int)
                df = self.parent.profile_df
                for uid_ind in df.index.values.tolist():
                    temp_series = df.loc[uid_ind]
                    temp_series_non_zero_series = temp_series[temp_series > 0]
                    non_zero_indices = temp_series_non_zero_series.index.values.tolist()
                    if len(non_zero_indices) == 4:
                        foo = 'asdf'
                    dd[len(non_zero_indices)] += 1

                self.fig = plt.figure(figsize=(8, 8))
                # one row for each clade, for the between sample ordinations
                # one row for the clade proportion ordination
                # one row for the legends
                # one column per meta info category i.e. species, reef, reef_type, etc.
                self.gs = gridspec.GridSpec(1, 1)
                self.quat_ax = plt.subplot(self.gs[0:1, 0:1])
                # we will work with a box that is 10 by 10
                self.quat_ax.set_ylim(0,10)
                self.quat_ax.set_xlim(0,10)
                # now set up the gridlines which will join all integer points on both axes

        qp = QuantPlot(self)

    def _make_temp_df(self):
        # We should plot the daily averages rather then every point as readability suffers. We can then do
        # a seperate subplot that has a comparison of the daily variance
        if os.path.exists(os.path.join(self.cache_dir, 'daily_temperature_min_df.p')):
            self.daily_temperature_av_df = pickle.load(open(os.path.join(self.cache_dir, 'daily_temperature_av_df.p'), 'rb'))
            self.daily_temperature_range_df = pickle.load(open(os.path.join(self.cache_dir, 'daily_temperature_range_df.p'), 'rb'))
            self.temperature_df = pickle.load(open(os.path.join(self.cache_dir, 'temperature_df.p'), 'rb'))
            self.daily_temperature_max_df = pickle.load(open(os.path.join(self.cache_dir, 'daily_temperature_max_df.p'), 'rb'))
            self.daily_temperature_min_df = pickle.load(open(os.path.join(self.cache_dir, 'daily_temperature_min_df.p'), 'rb'))
        else:
            temperature_df = pd.DataFrame()
            files = [f for f in os.listdir(self.hobo_dir) if '.csv' in f]
            for f in files:
                temp_df = pd.read_csv(os.path.join(self.hobo_dir, f), names=['ind', 'date', 'temp'])
                temp_df.set_index('date', drop=True, inplace=True)
                temp_df.drop(columns='ind', inplace=True)
                temp_df = temp_df.dropna()
                temperature_df[f.replace('.csv', '')] = temp_df['temp']
            self.temperature_df = temperature_df.loc[:'7/23/17 00:00', :]

            # now calculate the daily averages. Work from 00:00 to 23:00 for each bin
            # at the same time we can calculate the daily range for each day
            daily_average_temp_dict = {}
            daily_max_temp_dict = {}
            daily_min_temp_dict = {}
            daily_range_temp_dict = {}
            # a list for each of the hobo columns
            daily_temperature_bin_lists = [[] for _ in range(len(list(self.temperature_df)))]
            current_day = None
            for time_ind in self.temperature_df.index.values.tolist():
                if '00:00' in time_ind:
                    if daily_temperature_bin_lists and len(daily_temperature_bin_lists[0]) == 24: # must be a full set of data
                        # then we should bank calculate what we have in the bin_list and populate a new item in the bin_dict
                        average_values = [sum(bin_list)/len(bin_list) for bin_list in daily_temperature_bin_lists]
                        max_values = [max(bin_list) for bin_list in daily_temperature_bin_lists]
                        min_values = [min(bin_list) for bin_list in daily_temperature_bin_lists]
                        daily_ranges = [max(bin_list)-min(bin_list) for bin_list in daily_temperature_bin_lists]
                        daily_average_temp_dict[current_day] = average_values
                        daily_max_temp_dict[current_day] = max_values
                        daily_min_temp_dict[current_day] = min_values
                        daily_range_temp_dict[current_day] = daily_ranges
                        daily_temperature_bin_lists = [[] for _ in range(len(list(self.temperature_df)))]
                        self._populate_bin_list_for_single_time(daily_temperature_bin_lists, time_ind)
                    elif daily_temperature_bin_lists and len(daily_temperature_bin_lists[0]) != 24:
                        # then we have a partial set of data and we don't want to calculate an average from this
                        daily_temperature_bin_lists = [[] for _ in range(len(list(self.temperature_df)))]
                        self._populate_bin_list_for_single_time(daily_temperature_bin_lists, time_ind)
                else:
                    # update the current day we are on and add the temp data to the bin_lists
                    current_day = time_ind.split(' ')[0]
                    self._populate_bin_list_for_single_time(daily_temperature_bin_lists, time_ind)
            if len(daily_temperature_bin_lists[0]) == 24:
                # then we have one last full set of data
                average_values = [sum(bin_list) / len(bin_list) for bin_list in daily_temperature_bin_lists]
                max_values = [max(bin_list) for bin_list in daily_temperature_bin_lists]
                min_values = [min(bin_list) for bin_list in daily_temperature_bin_lists]
                daily_ranges = [max(bin_list) - min(bin_list) for bin_list in daily_temperature_bin_lists]

                daily_average_temp_dict[current_day] = average_values
                daily_max_temp_dict[current_day] = max_values
                daily_min_temp_dict[current_day] = min_values
                daily_range_temp_dict[current_day] = daily_ranges

            self.daily_temperature_av_df = pd.DataFrame.from_dict(data=daily_average_temp_dict, orient='index', columns=self.temperature_df.columns.values.tolist())
            self.daily_temperature_max_df = pd.DataFrame.from_dict(data=daily_max_temp_dict, orient='index', columns=self.temperature_df.columns.values.tolist())
            self.daily_temperature_min_df = pd.DataFrame.from_dict(data=daily_min_temp_dict, orient='index', columns=self.temperature_df.columns.values.tolist())
            self.daily_temperature_range_df = pd.DataFrame.from_dict(data=daily_range_temp_dict, orient='index', columns=self.temperature_df.columns.values.tolist())

            pickle.dump(self.daily_temperature_av_df, open(os.path.join(self.cache_dir, 'daily_temperature_av_df.p'), 'wb'))
            pickle.dump(self.daily_temperature_max_df,
                        open(os.path.join(self.cache_dir, 'daily_temperature_max_df.p'), 'wb'))
            pickle.dump(self.daily_temperature_min_df,
                        open(os.path.join(self.cache_dir, 'daily_temperature_min_df.p'), 'wb'))
            pickle.dump(self.daily_temperature_range_df, open(os.path.join(self.cache_dir, 'daily_temperature_range_df.p'), 'wb'))
            pickle.dump(self.temperature_df, open(os.path.join(self.cache_dir, 'temperature_df.p'), 'wb'))

    def _populate_bin_list_for_single_time(self, bin_lists, time_ind):
        for i, temp_val in enumerate(self.temperature_df.loc[time_ind].values.tolist()):
            bin_lists[i].append(temp_val)

    def _plot_temperature(self):
        fig = plt.figure(figsize=(12, 6))
        # we will put a gap of 1 row in for the site plots
        gs = gridspec.GridSpec(30, 27)

        # average plots
        one_m_all_sites = plt.subplot(gs[:6, :8])
        fifteen_m_all_sites = plt.subplot(gs[7:13, :8])
        thirty_m_all_sites = plt.subplot(gs[14:20, :8])

        # inshore
        inshore_fsar = plt.subplot(gs[:6, 9:13])
        inshore_tahala = plt.subplot(gs[:6, 14:18])

        # midshore
        midshore_al_fahal = plt.subplot(gs[7:13, 9:13])
        midshore_quita_al_kirsh = plt.subplot(gs[7:13, 14:18])

        # offshore
        offshore_shib_nazar = plt.subplot(gs[14:20, 9:13])
        offshore_abud_madafi = plt.subplot(gs[14:20, 14:18])

        # boxplot
        box_plot_ax = plt.subplot(gs[:10, 19:])

        # remotely sensed vs 1m plot
        r_sst_ax = plt.subplot(gs[10:20, 19:])

        # site separated sensed vs 1m plot
        r_sst_ax_arr = []
        r_sst_ax_arr.append(plt.subplot(gs[20:, :6]))
        r_sst_ax_arr.append(plt.subplot(gs[20:, 6:12]))
        r_sst_ax_arr.append(plt.subplot(gs[20:, 12:18]))
        r_sst_ax_arr.append(plt.subplot(gs[20:, 18:24]))

        indi_axarr = []
        indi_axarr.extend([inshore_fsar, inshore_tahala, midshore_al_fahal, midshore_quita_al_kirsh, offshore_shib_nazar, offshore_abud_madafi])

        reef_order = ['Fsar', 'Tahla', 'Al Fahal', 'Qita al Kirsh', 'Shib Nazar', 'Abu Madafi']
        abbrev_dict = {'Fsar': 'f', 'Tahla': 't', 'Qita al Kirsh': 'q', 'Al Fahal': 'af',
                                                         'Shib Nazar': 'sn', 'Abu Madafi': 'am'}
        # line_style_dict = {'1':'-', '15':'-.', '30':'--'}
        line_color_dict = {'1': '#CAE1FF', '15': '#2E37FE', '30': '#000080'}
        x = self.daily_temperature_av_df.index.values.tolist()
        ax_ind = -1
        for reef in reef_order:
            count = 0
            ax_ind += 1
            for depth in ['1', '15', '30']:
                column_header = f'{abbrev_dict[reef]}_{depth}'
                if column_header in list(self.temperature_df):
                    count += 1
                    y = self.daily_temperature_av_df[column_header].values.tolist()
                    # indi_axarr[ax_ind].plot(x,y, linestyle=line_style_dict[depth], c=line_color_dict[depth], lw='0.5')
                    indi_axarr[ax_ind].plot(x,y,  c=line_color_dict[depth], lw='0.2')

            if count == 0:
                # then there was no data for this site and we should plot the words 'no data'
                indi_axarr[ax_ind].set_ylim(0, 1)
                indi_axarr[ax_ind].set_xlim(0, 1)
                indi_axarr[ax_ind].text(x=0.5, y=0.5, s='no data', horizontalalignment='center', verticalalignment='center', fontsize='small')
                indi_axarr[ax_ind].set_xticks([])
                indi_axarr[ax_ind].set_yticks([])
            else:
                indi_axarr[ax_ind].set_ylim(24,34)
                indi_axarr[ax_ind].set_yticks([24, 26, 28, 30, 32, 34])
                indi_axarr[ax_ind].grid()
                indi_axarr[ax_ind].set_xticks([])
                # indi_axarr[ax_ind].patch.set_facecolor('#DCDCDC')

        # now plot the other two

        # x = self._plot_one_m_temp(one_m_all_sites, reef_order, abbrev_dict)
        #
        # self._plot_fifteen_m_temp(fifteen_m_all_sites, reef_order, abbrev_dict)

        self._plot_depth_plot(ax=one_m_all_sites, reef_order=reef_order, abbrev_dict=abbrev_dict, depth_label='1')
        self._plot_depth_plot(ax=fifteen_m_all_sites, reef_order=reef_order, abbrev_dict=abbrev_dict, depth_label='15')
        self._plot_depth_plot(ax=thirty_m_all_sites, reef_order=reef_order, abbrev_dict=abbrev_dict, depth_label='30')
        # now do the box plot
        self._plot_temp_box_plots(box_plot_ax)

        # # now do the remotely sensed plotting
        # self._plot_remotely_sensed_data_1m(r_sst_ax)

        # plot the remotely sensed data compared to the insidu data on a site separated basis
        self._plot_remotely_sensed_data_1m_site_separated(r_sst_ax_arr)

        # plt.tight_layout()
        apples = 'asdf'
        plt.savefig(os.path.join(self.figure_dir, 'temp_plot_with_remotely_sensed.png'), dpi=1200)
        plt.savefig(os.path.join(self.figure_dir, 'temp_plot_with_remotely_sensed.svg'), dpi=1200)

    def _plot_remotely_sensed_data_1m_site_separated(self, ax_arr):
        """This function will plot up a three lines that represent the difference between the remotely sensed
        temperature SST and the max, min and average daily in situ temperature
        """

        x = self.remotely_sensed_sst_df.index.values.tolist()
        for i, reef in enumerate(["Shib Nazar", 'Al Fahal', 'Qita al Kirsh', 'Tahla']):
            key = self._get_reef_key(reef)
            remote_y = self.remotely_sensed_sst_df[reef]
            max_y = self.daily_temperature_max_df[key]
            av_y = self.daily_temperature_av_df[key]
            min_y = self.daily_temperature_min_df[key]


            # # plot top line (diff to max)
            # ax_arr[i].plot(x, max_y - remote_y, c='grey', lw='0.5')

            # plot middle line (diff to av)
            ax_arr[i].plot(x, av_y - remote_y, c='black', lw='0.5', zorder=2)
            ax_arr[i].set_ylim(-1, 4)

            # # plot bottom line (diff to min)
            # ax_arr[i].plot(x, min_y - remote_y, c='grey', lw='0.5')

            # The x's for the lines are currently the days but we will need the data coordinates if we are going
            # to plot a polygon as the background.
            x_axis = ax_arr[i].get_xaxis()
            tick_loc_arr = x_axis.get_majorticklocs()
            # We have to fix the x lim as it was autoadjusting when adding the gridlines
            ax_arr[i].set_xlim(min(tick_loc_arr), max(tick_loc_arr))

            poly_1 = [(x, y) for x, y in zip(tick_loc_arr, (min_y - remote_y))]
            poly_2 = [(x, y) for x, y in zip(tick_loc_arr, (max_y - remote_y))]
            poly_2.reverse()
            poly = poly_1 + poly_2
            range_poly = Polygon(poly, closed=True, fill=True, color='lightgrey', zorder=1)
            ax_arr[i].add_patch(range_poly)
            ax_arr[i].set_xticks([])
            if i > 0:
                ax_arr[i].set_yticks([])

            # put on a grid that we will want to be behind the plotting
            # zorder is broken for the grid system so we will quickly put in some grids by hand with the correct
            # zorder honoured
            for y in range(4):
                ax_arr[i].hlines(y=y, xmin=ax_arr[i].get_xlim()[0], xmax=ax_arr[i].get_xlim()[1], colors='darkgrey', zorder=0, linewidth=0.5)
                # ax_arr[i].set_xlim(min(tick_loc_arr), max(tick_loc_arr))
            ax_arr[i].grid(zorder=0)
        foo = 'bar'

    def _get_reef_key(self, reef):
        if reef == 'Tahla':
            key = 't_1'
        elif reef == 'Qita al Kirsh':
            key = 'q_1'
        elif reef == 'Al Fahal':
            key = 'af_1'
        else:  # reef == 'Shib Nazar'
            key = 'sn_1'
        return key

    def _plot_remotely_sensed_data_1m(self, ax):
        # first plot the remotely sensed data to see what we're looking at
        # The x will be the same (hopefully) for both the insitu data and the remotely sensed data
        x = self.remotely_sensed_sst_df.index.values.tolist()
        for reef in ["Shib Nazar", 'Al Fahal', 'Qita al Kirsh', 'Tahla']:
            y = self.remotely_sensed_sst_df[reef].values.tolist()
            ax.plot(x, y, c='grey', lw='0.5')

        # now plot up the insitu data
        # we can first try plotting this up as the daily averages but it may need to be that
        # draw a polygon that is made up of the max and min values for each day, as we don't
        # really know what time the remotely sensed data refers to.
        for reef in ["Shib Nazar", 'Al Fahal', 'Qita al Kirsh', 'Tahla']:
            key = self._get_reef_key(reef)
            y = self.daily_temperature_av_df[key].values.tolist()
            ax.plot(x, y, c=self.old_color_dict[reef], lw='0.5')

        foo = 'bar'

    def _plot_temp_box_plots(self, box_plot_ax):
        # first get the individual hobo lists
        list_to_plot = [self.daily_temperature_range_df[col_name].values.tolist() for col_name in
                        list(self.daily_temperature_range_df)]
        # then get a list for each of the depth averages
        for depth in ['1', '15', '30']:
            temp_depth_list = []
            for col in list(self.daily_temperature_range_df):
                if col.endswith(depth):
                    temp_depth_list.extend(self.daily_temperature_range_df[col].values.tolist())
            list_to_plot.append(temp_depth_list)
        box_plot_ax.boxplot(list_to_plot, flierprops={'markersize': 1})
        labels = list(self.daily_temperature_range_df)
        labels.extend(['1m', '15m', '30m'])
        box_plot_ax.set_xticklabels(labels, rotation='vertical')
        # box_plot_ax.patch.set_facecolor('#DCDCDC')

    def _plot_fifteen_m_temp(self, fifteen_m_all_sites, reef_order, abbrev_dict):
        x = self.daily_temperature_av_df.index.values.tolist()
        for reef in reef_order:
            column_header = f'{abbrev_dict[reef]}_{15}'
            if column_header in list(self.temperature_df):
                y = self.daily_temperature_av_df[column_header].values.tolist()
                fifteen_m_all_sites.plot(x, y, c=self.old_color_dict[reef], lw='0.5')
        fifteen_m_all_sites.set_ylim(24, 34)
        fifteen_m_all_sites.set_yticks([24, 26, 28, 30, 32, 34])
        fifteen_m_all_sites.grid()
        fifteen_m_all_sites.set_xticks([])
        # fifteen_m_all_sites.patch.set_facecolor('#DCDCDC')

    def _plot_one_m_temp(self, one_m_all_sites, reef_order,abbrev_dict):
        x = self.daily_temperature_av_df.index.values.tolist()
        for reef in reef_order:
            column_header = f'{abbrev_dict[reef]}_{1}'
            if column_header in list(self.temperature_df):
                y = self.daily_temperature_av_df[column_header].values.tolist()
                one_m_all_sites.plot(x, y, c=self.old_color_dict[reef], lw='0.5')
        one_m_all_sites.set_ylim(24, 34)
        one_m_all_sites.set_yticks([24, 26, 28, 30, 32, 34])
        one_m_all_sites.grid()
        one_m_all_sites.set_xticks([])
        # one_m_all_sites.patch.set_facecolor('#DCDCDC')
        return x

    def _plot_depth_plot(self, ax, reef_order, abbrev_dict, depth_label):
        x = self.daily_temperature_av_df.index.values.tolist()
        for reef in reef_order:
            column_header = f'{abbrev_dict[reef]}_{depth_label}'
            if column_header in list(self.temperature_df):
                y = self.daily_temperature_av_df[column_header].values.tolist()
                ax.plot(x, y, c=self.old_color_dict[reef], lw='0.5')
        ax.set_ylim(24, 34)
        ax.set_yticks([24, 26, 28, 30, 32, 34])
        ax.grid()
        ax.set_xticks([])
        # one_m_all_sites.patch.set_facecolor('#DCDCDC')
        return x

    def _del_propblem_sample(self):
        """ THis is originally done in the meta info df creation but using the cache system sometimes
        meant that this was being skipped. I have put it in here so that it is never skipped and the sample
        is always removed."""
        # delete 'FS15SE8_FS15SE8_N705-S508' from the df
        for uid, name in self.smp_uid_to_name_dict.items():
            if name == 'FS15SE8_FS15SE8_N705-S508':
                self.post_med_seq_abundance_relative_df.drop(index=uid, inplace=True, errors='ignore')
                self.profile_abundance_df.drop(index=uid, inplace=True, errors='ignore')
                self.clade_prop_pcoa_coords.drop(index=uid, inplace=True, errors='ignore')
                self.clade_proportion_df.drop(index=uid, inplace=True, errors='ignore')
                self.clade_proportion_df_non_normalised.drop(index=uid, inplace=True, errors='ignore')
                self.seq_meta_data_df.drop(index=uid, inplace=True, errors='ignore')

                if self.profile_abundance_df_cutoff is not None:
                    self.profile_abundance_df_cutoff.drop(index=uid, inplace=True, errors='ignore')
                if self.between_sample_clade_dist_df_dict:
                    for clade in self.clades:
                        if uid in self.between_sample_clade_dist_df_dict[clade].index.values.tolist():
                            self.between_sample_clade_dist_df_dict[clade].drop(index=uid, inplace=True, errors='ignore')
                            self.between_sample_clade_dist_df_dict[clade].drop(columns=uid, inplace=True, errors='ignore')
                self.profile_abundance_df_cutoff_high.drop(index=uid, inplace=True, errors='ignore')
                self.profile_abundance_df_cutoff_low.drop(index=uid, inplace=True, errors='ignore')
                self.profile_abundance_df_cutoff_background.drop(index=uid, inplace=True, errors='ignore')
                break

    def _del_problem_sample_from_a_df(self, df=None, list_of_dfs=None):
        if df is not None:
            for uid, name in self.smp_uid_to_name_dict.items():
                if name == 'FS15SE8_FS15SE8_N705-S508':
                    df.drop(index=uid, inplace=True, errors='ignore')
                    break
        else:
            for uid, name in self.smp_uid_to_name_dict.items():
                if name == 'FS15SE8_FS15SE8_N705-S508':
                    for df_ind_list in list_of_dfs:
                        df_ind_list.drop(index=uid, inplace=True, errors='ignore')
                    break

    def _if_clade_proportion_df_cache_exists(self):
        return os.path.exists(os.path.join(self.cache_dir, 'clade_proportion_df.p'))

    def _if_clade_proportion_distance_dict_chache_exists(self):
        return os.path.exists(os.path.join(self.cache_dir, 'clade_prop_distance_dict.p'))

    def _create_clade_prop_distances(self):
        """Go through the self.parent.seq_df and get the proportion of A, C and D sequences
        for each sample and populate this into the self.clade_proportion_df."""
        if os.path.exists(os.path.join(self.cache_dir, 'betweeen_sample_clade_proportion_distances.p')):
            self.clade_prop_pcoa_coords = pickle.load(open(os.path.join(self.cache_dir, 'clade_prop_pcoa_coords.p'), 'rb'))
            self.clade_proportion_df = pickle.load(
                open(os.path.join(self.cache_dir, 'clade_proportion_df.p'), 'rb'))
            self.clade_proportion_df_non_normalised = pickle.load(
                open(os.path.join(self.cache_dir, 'clade_proportion_df_non_norm.p'), 'rb'))
            self.between_sample_clade_proportion_distances_df = pickle.load(
                open(os.path.join(self.cache_dir, 'betweeen_sample_clade_proportion_distances.p'), 'rb'))

            self._report_clade_abund_proportions()

        else:
            self._del_problem_sample_from_a_df(df=self.clade_proportion_df)
            sample_uids = self.clade_proportion_df.index.values.tolist()
            if self._if_clade_proportion_df_cache_exists():
                self._set_clade_proportion_df_from_cache()
                self.clade_proportion_df = pickle.load(
                    open(os.path.join(self.cache_dir, 'clade_proportion_df.p'), 'rb'))
                self.clade_proportion_df_non_normalised = pickle.load(
                    open(os.path.join(self.cache_dir, 'clade_proportion_df_non_norm.p'), 'rb'))
                self._del_problem_sample_from_a_df(list_of_dfs = [self.clade_proportion_df, self.clade_proportion_df_non_normalised])
            else:
                self._set_clade_proportion_df_from_scratch(sample_uids)

            self._report_clade_abund_proportions()

            # populate a dictionary that will hold the distances between each of the samples
            if self._if_clade_proportion_distance_dict_chache_exists():
                clade_prop_distance_dict = self._set_clade_proportion_distance_dict_from_chache()
            else:
                clade_prop_distance_dict = self._make_clade_proportion_distance_dict_from_scratch(sample_uids)

            dist_file_as_list = self._make_clade_prop_distance_matrix_2dlist(clade_prop_distance_dict, sample_uids)

            self._clade_proportion_pcoa_coords_df(dist_file_as_list, sample_uids)

    def _report_clade_abund_proportions(self):
        # Two results to output here:
        # 1 - how many samples has 1, 2, and 3 clades representative
        # then, what were the average abundances of the 1, 2, and 3rd most abundant clades
        # and how many samples were each of those numbers generated from.
        num_clade_dd = defaultdict(int)
        av_abund_of_the_nth_abundant_clade_dd = defaultdict(list)
        for index, row in self.clade_proportion_df_non_normalised.iterrows():
            non_zero_vals = row.iloc[row.to_numpy().nonzero()].values.tolist()
            num_clade_dd[len(non_zero_vals)] += 1
            sorted_vals = sorted(non_zero_vals, reverse=True)
            for i, val in enumerate(sorted_vals):
                av_abund_of_the_nth_abundant_clade_dd[i].append(val)
        # now we have the containers populated and we can output the required results summaries
        print('\n\n')
        for i in range(3):
            print(f'{num_clade_dd[i+1]} samples had sequences from only {i+1} family')
        print('\n\n')
        for i in range(3):
            num_samp = len(av_abund_of_the_nth_abundant_clade_dd[i])
            av_abund = sum(av_abund_of_the_nth_abundant_clade_dd[i]) / num_samp
            std = statistics.pstdev(av_abund_of_the_nth_abundant_clade_dd[i])
            print(f'The average abundance of the {i+1} most abundant family '
                  f'was {av_abund} (stdv: {std})from {num_samp} samples')

    def _clade_proportion_pcoa_coords_df(self, dist_file_as_list, sample_uids):
        pcoa_output = pcoa(np.array(dist_file_as_list))
        # rename the pcoa dataframe index as the sample uids
        pcoa_output.samples['sample_uid'] = sample_uids
        renamed_pcoa_dataframe = pcoa_output.samples.set_index('sample_uid')
        # now add the variance explained as a final row to the renamed_dataframe
        self.clade_prop_pcoa_coords = renamed_pcoa_dataframe.append(
            pcoa_output.proportion_explained.rename('proportion_explained'))
        self.clade_prop_pcoa_coords.to_csv(
            os.path.join(self.outputs_dir, 'sample_clade_props_pcoa_coords.csv'),
            index=True, header=True, sep=',')
        self._del_problem_sample_from_a_df(df=self.clade_prop_pcoa_coords)
        pickle.dump(self.clade_prop_pcoa_coords, open(os.path.join(self.cache_dir, 'clade_prop_pcoa_coords.p'), 'wb'))

    def _make_clade_prop_distance_matrix_2dlist(self, clade_prop_distance_dict, sample_uids):
        dist_file_as_list = []
        for uid_outer in sample_uids:
            temp_at_string = []

            for uid_inner in sample_uids:
                if uid_outer == uid_inner:
                    temp_at_string.append(0)
                else:
                    temp_at_string.append(
                        clade_prop_distance_dict[frozenset({uid_outer, uid_inner})])
            dist_file_as_list.append(temp_at_string)
        self.between_sample_clade_proportion_distances_df = pd.DataFrame(dist_file_as_list, columns=sample_uids, index=sample_uids)
        self._del_problem_sample_from_a_df(df=self.between_sample_clade_proportion_distances_df)
        # output the dataframe so that it can be used for PERMANOVA analysis
        self.between_sample_clade_proportion_distances_df.to_csv(path_or_buf=os.path.join(self.outputs_dir, 'between_sample_clade_proportion_distances.csv'), index=False, header=False)
        # also output a corresponding meta info df
        temp_meta_df = self.experimental_metadata_info_df.loc[sample_uids,]
        self._del_problem_sample_from_a_df(df=temp_meta_df)
        temp_meta_df.to_csv(path_or_buf=os.path.join(self.outputs_dir, 'sample_meta_for_clade_proportion_permanova.csv'), index=False, header=True)
        pickle.dump(self.between_sample_clade_proportion_distances_df, open(os.path.join(self.cache_dir, 'betweeen_sample_clade_proportion_distances.p'), 'wb'))
        return dist_file_as_list

    def _make_clade_proportion_distance_dict_from_scratch(self, sample_uids):
        uid_list_dict = {uid: self.clade_proportion_df.loc[uid].values.tolist() for uid in sample_uids}
        list_of_list_tups = []
        name_pair_list = []
        for uid_one, uid_two in itertools.combinations(sample_uids, 2):
            name_pair_list.append((uid_one, uid_two))
            list_of_list_tups.append((uid_list_dict[uid_one], uid_list_dict[uid_two]))

        with Pool(7) as p:
            dist_list_out = p.map(braycurtis_tup, list_of_list_tups)

        foo = 'bar'

        clade_prop_distance_dict = {frozenset({name_tup[0], name_tup[1]}) : dist for name_tup, dist in zip(name_pair_list, dist_list_out)}

        # tot = len(sample_uids) * len(sample_uids)
        # count = 0
        # for uid_one, uid_two in itertools.combinations(sample_uids, 2):
        #     count += 1
        #     sys.stdout.write(f'\r{str(count)}/{tot}')
        #     distance = braycurtis(self.clade_proportion_df.loc[uid_one].values.tolist(),
        #                           self.clade_proportion_df.loc[uid_two].values.tolist())
        #     clade_prop_distance_dict[frozenset({uid_one, uid_two})] = distance
        pickle.dump(clade_prop_distance_dict,
                    open(os.path.join(self.cache_dir, 'clade_prop_distance_dict.p'), 'wb'))
        return clade_prop_distance_dict

    def _set_clade_proportion_distance_dict_from_chache(self):
        clade_prop_distance_dict = pickle.load(
            open(os.path.join(self.cache_dir, 'clade_prop_distance_dict.p'), 'rb'))
        return clade_prop_distance_dict

    def _set_clade_proportion_df_from_scratch(self, sample_uids):
        for sample_uid in sample_uids:
            print(f'Counting clade abundances for sample {sample_uid}')
            sample_series = self.post_med_seq_abundance_relative_df.loc[sample_uid]
            clade_prop_dict = {'A': 0.0, 'C': 0.0, 'D': 0.0}
            for seq_name in sample_series.index.values.tolist():
                if 'A' in seq_name:
                    clade_prop_dict['A'] += sample_series[seq_name]
                elif 'C' in seq_name:
                    clade_prop_dict['C'] += sample_series[seq_name]
                elif 'D' in seq_name:
                    clade_prop_dict['D'] += sample_series[seq_name]
            self.clade_proportion_df_non_normalised.at[sample_uid, 'A'] = clade_prop_dict['A']
            self.clade_proportion_df_non_normalised.at[sample_uid, 'C'] = clade_prop_dict['C']
            self.clade_proportion_df_non_normalised.at[sample_uid, 'D'] = clade_prop_dict['D']
            # here we have the totals of the seqs for a given sample separated by clades
            self.clade_proportion_df.at[sample_uid, 'A'] = int(clade_prop_dict['A'] * 100000)
            self.clade_proportion_df.at[sample_uid, 'C'] = int(clade_prop_dict['C'] * 100000)
            self.clade_proportion_df.at[sample_uid, 'D'] = int(clade_prop_dict['D'] * 100000)
        self._del_problem_sample_from_a_df(
            list_of_dfs=[self.clade_proportion_df, self.clade_proportion_df_non_normalised])
        pickle.dump(self.clade_proportion_df,
                    open(os.path.join(self.cache_dir, 'clade_proportion_df.p'), 'wb'))
        pickle.dump(self.clade_proportion_df_non_normalised,
                    open(os.path.join(self.cache_dir, 'clade_proportion_df_non_norm.p'), 'wb'))

    def _set_clade_proportion_df_from_cache(self):
        self.clade_proportion_df = pickle.load(
            open(os.path.join(self.cache_dir, 'clade_proportion_df.p'), 'rb'))
        self.clade_proportion_df_non_normalised = pickle.load(
            open(os.path.join(self.cache_dir, 'clade_proportion_df_non_norm.p'), 'rb'))

    def _post_med_seq_abundance_relative_df(self):
        if os.path.exists(os.path.join(self.cache_dir, 'seq_df.p')):
            return pickle.load(open(os.path.join(self.cache_dir, 'seq_df.p'), 'rb'))
        else:
            # with open(self.seq_rel_abund_post_med_ouput_path, 'r') as f:
            #     seq_data = [out_line.split('\t') for out_line in [line.rstrip() for line in f]]

            df = pd.read_csv(filepath_or_buffer=self.seq_rel_abund_post_med_ouput_path, sep='\t', index_col=0, header=0)
            # df = pd.DataFrame(seq_data)
            # df.iat[0,0] = 'sample_uid'
            # df.columns = df.iloc[0]
            # df.drop(index=0, inplace=True)
            # df.drop(columns='sample_name', inplace=True)
            # df.set_index('sample_uid', drop=True, inplace=True)
            # # Get rid of all of the superflous columns only leaving the seq rel counts
            # df = df.iloc[:, 20:]
            # df = df[:-5]
            # df.index = df.index.astype('int')
            # df = df.astype('float')
            pickle.dump(df, open(os.path.join(self.cache_dir, 'seq_df.p'), 'wb'))
            return df

    def _pre_med_seq_abundance_relative_df(self):
        """we will produce a df that is index as sample_uid and sequence name as cols.
        The only problem is that we have three separate pre-med csvs to work with so we will have to do some
        consolidation. We will also likely have to grab a list of uids or names from which to get a master
        fasta out of the SymPortal db."""
        if os.path.exists(os.path.join(self.cache_dir, 'seq_df_pre_med.p')):
            return pickle.load(open(os.path.join(self.cache_dir, 'seq_df_pre_med.p'), 'rb'))
        else:
            list_of_dfs = []
            for pre_seq_output_path in self.seq_rel_abund_pre_med_ouput_path_list:
                with open(pre_seq_output_path, 'r') as f:
                    seq_data = [out_line.split(',') for out_line in [line.rstrip() for line in f]]
                cols = seq_data[0][2:]
                # change the strange format of the unk_C_XXX
                new_cols = []
                for col_item in cols:
                    if 'unk' in col_item:
                        split_list = col_item.split('_')
                        new_cols.append(f'{split_list[2]}_{split_list[1]}')
                    else:
                        new_cols.append(col_item)
                seq_data = seq_data[1:]
                ind = [int(sub[0]) for sub in seq_data]
                seq_data = [sub[2:] for sub in seq_data]
                print('a pre-med df was created')
                df = pd.DataFrame(seq_data, columns=new_cols, index=ind).astype('float')
                list_of_dfs.append(df)
                # df = pd.read_csv(filepath_or_buffer=pre_seq_output_path, sep='\t', index_col=0, header=0)
            print('first append')
            master_df = list_of_dfs[0].append(list_of_dfs[1]).fillna(0).astype('float')
            print('second append')
            master_df = master_df.append(list_of_dfs[2])
            pickle.dump(master_df, open(os.path.join(self.cache_dir, 'seq_df_pre_med.p'), 'wb'))
            return master_df

    def _init_metadata_info_df(self):
        """ This method produces a dataframe that has sample UID as key and
        'species', 'reef', 'reef_type', 'depth' 'season' as columns.

        The matching of names between the SP output and the meta info that Alejandro was working from was causing us
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
        if os.path.exists(os.path.join(self.cache_dir, 'meta_info_df.p')):
            return pickle.load(open(os.path.join(self.cache_dir, 'meta_info_df.p'), 'rb'))
        else:
            meta_info_df = pd.DataFrame.from_csv(self.meta_data_input_path)
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
                    self.post_med_seq_abundance_relative_df.drop(index=uid, inplace=True)
                    self.profile_abundance_df.drop(index=uid, inplace=True)
                    if self.profile_abundance_df_cutoff is not None:
                        self.profile_abundance_df_cutoff.drop(index=uid, inplace=True)
                    if self.between_sample_clade_dist_df_dict:
                        for clade in self.clades:
                            if uid in self.between_sample_clade_dist_df_dict[clade].index.values.tolist():
                                self.between_sample_clade_dist_df_dict[clade].drop(index=uid, inplace=True)
                                self.between_sample_clade_dist_df_dict[clade].drop(columns=uid, inplace=True)
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
            pickle.dump(meta_info_df, open(os.path.join(self.cache_dir, 'meta_info_df.p'), 'wb'))
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

    def _populate_clade_dist_df_dict(self, cct_specific=None, smp_dist=False):
        """If cct_specific is set then we are making dfs for the distance matrices that are from the bespoke
        set of CladeCollectionTypes. If not set then it is the first set of distances that have come straight out
        of the SymPortal analysis with no prior processing. I have implemented a simple cache system.
        This code will also work for the population of between sample distances if the smp_dist is set to either
        braycurtis or unifrac (depending on which distance outputs we will use)."""
        if not self.ignore_cache:
            self.pop_clade_dist_df_dict_from_cache_or_make_new(cct_specific, smp_dist)
        else:
            self._pop_clade_dict_df_dict_from_scratch_and_pickle_out(cct_specific, smp_dist)

    def pop_clade_dist_df_dict_from_cache_or_make_new(self, cct_specific, smp_dist):
        try:
            if smp_dist:
                self.between_sample_clade_dist_df_dict = pickle.load(
                    file=open(os.path.join(self.cache_dir, f'sample_clade_dist_df_dict_{self.seq_distance_method}.p'), 'rb'))
            elif cct_specific:
                if cct_specific == 'low':
                    self.profile_distance_df_dict_cutoff_low = pickle.load(
                        file=open(os.path.join(self.cache_dir,
                                               f'clade_dist_cct_{cct_specific}_{self.profile_distance_method}_specific_dict.p'),
                                  'rb'))
                elif cct_specific == 'background':
                    self.profile_distance_df_dict_cutoff_background = pickle.load(
                        file=open(os.path.join(self.cache_dir,
                                               f'clade_dist_cct_{cct_specific}_{self.profile_distance_method}_specific_dict.p'),
                                  'rb'))
                else:
                    if cct_specific == '040':
                        self.profile_distance_df_dict_cutoff_high = pickle.load(
                            file=open(os.path.join(self.cache_dir,
                                                   f'clade_dist_cct_{cct_specific}_{self.profile_distance_method}_specific_dict.p'),
                                      'rb'))
                    self.between_profile_clade_dist_cct_specific_df_dict = pickle.load(
                        file=open(os.path.join(self.cache_dir, f'clade_dist_cct_{cct_specific}_{self.profile_distance_method}_specific_dict.p'), 'rb'))
            else:
                self.between_profile_clade_dist_df_dict = pickle.load(file=open(os.path.join(self.cache_dir, f'profile_clade_dist_df_dict_{self.profile_distance_method}.p'), 'rb'))
        except FileNotFoundError:
            self._pop_clade_dict_df_dict_from_scratch_and_pickle_out(cct_specific, smp_dist)

    def _pop_clade_dict_df_dict_from_scratch_and_pickle_out(self, cct_specific, smp_dist):
        self._pop_clade_dist_df_dict_from_scratch(cct_specific, smp_dist)

    def _pop_clade_dist_df_dict_from_scratch(self, cct_specific, smp_dist):
        if smp_dist:
            if self.seq_distance_method == 'braycurtis':
                path_dict_to_use = self.between_sample_clade_dist_path_dict_braycurtis
            elif self.seq_distance_method == 'unifrac':
                path_dict_to_use = self.between_sample_clade_dist_path_dict_unifrac
        elif cct_specific == 'low':
            path_dict_to_use = self.between_profile_clade_dist_cct_low_unifrac_specific_path_dict
        elif cct_specific == 'background':
            path_dict_to_use = self.between_profile_clade_dist_cct_background_unifrac_specific_path_dict
        elif cct_specific == '005':
            if self.profile_distance_method == 'braycurtis':
                path_dict_to_use = self.between_profile_clade_dist_cct_005_braycurtis_specific_path_dict
            else: #  'unifrac'
                path_dict_to_use = self.between_profile_clade_dist_cct_005_unifrac_specific_path_dict
        elif cct_specific == '040':
            if self.profile_distance_method == 'braycurtis':
                path_dict_to_use = self.between_profile_clade_dist_cct_040_braycurtis_specific_path_dict
            else:  # 'unifrac'
                path_dict_to_use = self.between_profile_clade_dist_cct_040_unifrac_specific_path_dict
        else:
            if self.profile_distance_method == 'braycurtis':
                path_dict_to_use = self.between_profile_clade_braycurtis_dist_path_dict
            else: #  'unifrac'
                path_dict_to_use = self.between_profile_clade_unifrac_dist_path_dict

        for clade in self.clades:
            with open(path_dict_to_use[clade], 'r') as f:
                if smp_dist and self.seq_distance_method == 'braycurtis':
                    clade_data = [out_line.split('\t') for out_line in [line.rstrip() for line in f][1:]]
                else:
                    clade_data = [out_line.split('\t') for out_line in [line.rstrip() for line in f]]

            df = pd.DataFrame(clade_data)


            df.drop(columns=0, inplace=True)
            df.set_index(keys=1, drop=True, inplace=True)
            df.index = df.index.astype('int')
            df.columns = df.index.values.tolist()

            if smp_dist:
                self.between_sample_clade_dist_df_dict[clade] = df.astype(dtype='float')
            elif cct_specific:
                if cct_specific == 'low':
                    self.profile_distance_df_dict_cutoff_low[clade] = df.astype(dtype='float')
                elif cct_specific == 'background':
                    self.profile_distance_df_dict_cutoff_background[clade] = df.astype(dtype='float')
                else:
                    self.between_profile_clade_dist_cct_specific_df_dict[clade] = df.astype(dtype='float')
                    if cct_specific == '040':
                        self.profile_distance_df_dict_cutoff_high[clade] = df.astype(dtype='float')
            else:
                self.between_profile_clade_dist_df_dict[clade] = df.astype(dtype='float')

        # pickle out the distance dataframe dictionaries according to what sort of dist they are.
        if smp_dist:
            pickle.dump(obj=self.between_sample_clade_dist_df_dict,
                        file=open(os.path.join(self.cache_dir, f'sample_clade_dist_df_dict_{self.seq_distance_method}.p'), 'wb'))
        elif cct_specific:
            if cct_specific == 'low':
                pickle.dump(obj=self.profile_distance_df_dict_cutoff_low,
                            file=open(os.path.join(self.cache_dir,
                                                   f'clade_dist_cct_{cct_specific}_{self.profile_distance_method}_specific_dict.p'),
                                      'wb'))
            elif cct_specific == 'background':
                pickle.dump(obj=self.profile_distance_df_dict_cutoff_background,
                            file=open(os.path.join(self.cache_dir,
                                                   f'clade_dist_cct_{cct_specific}_{self.profile_distance_method}_specific_dict.p'),
                                      'wb'))
            else:
                pickle.dump(obj=self.between_profile_clade_dist_cct_specific_df_dict,
                            file=open(os.path.join(self.cache_dir, f'clade_dist_cct_{cct_specific}_{self.profile_distance_method}_specific_dict.p'), 'wb'))
        else:
            pickle.dump(obj=self.between_profile_clade_dist_df_dict, file=open(os.path.join(self.cache_dir, f'profile_clade_dist_df_dict_{self.profile_distance_method}.p'), 'wb'))

    def _populate_profile_abund_meta_dfs_and_info_containers(self):
        if os.path.exists(os.path.join(self.cache_dir, 'profile_abund_df.p')):
            self.profile_abundance_df = pickle.load(open(os.path.join(self.cache_dir, 'profile_abund_df.p'), 'rb'))
            self.smp_uid_to_name_dict = pickle.load(open(os.path.join(self.cache_dir, 'smp_uid_to_name_dict.p'), 'rb'))
            self.smp_name_to_uid_dict = pickle.load(open(os.path.join(self.cache_dir, 'smp_name_to_uid_dict.p'), 'rb'))
            self.prof_uid_to_local_abund_dict = pickle.load(open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict.p'), 'rb'))
            self.prof_uid_to_global_abund_dict = pickle.load(open(os.path.join(self.cache_dir, 'prof_uid_to_global_abund_dict.p'), 'rb'))
            self.prof_uid_to_name_dict = pickle.load(open(os.path.join(self.cache_dir, 'prof_uid_to_name_dict.p'), 'rb'))
            self.prof_name_to_uid_dict = pickle.load(open(os.path.join(self.cache_dir, 'prof_name_to_uid_dict.p'), 'rb'))
            self.profile_meta_info_df = pickle.load(open(os.path.join(self.cache_dir, 'profile_meta_info_df.p'), 'rb'))
            self.profile_meta_info_df['ITS2 type abundance local'] = self.profile_meta_info_df['ITS2 type abundance local'].astype('float')
        else:
            # read in df
            df = pd.read_csv(filepath_or_buffer=self.profile_rel_abund_ouput_path, sep='\t', header=None)

            profile_meta_info_df = df.iloc[:7, :].T
            profile_meta_info_df = profile_meta_info_df.drop(index=1)
            profile_meta_info_df.iat[0,0] = 'profile_uid'
            profile_meta_info_df.columns = profile_meta_info_df.iloc[0]
            profile_meta_info_df = profile_meta_info_df.iloc[1:,:]
            profile_meta_info_df['profile_uid'] = pd.to_numeric(profile_meta_info_df['profile_uid'])
            profile_meta_info_df.set_index('profile_uid', drop=True, inplace=True)
            profile_meta_info_df['ITS2 type abundance DB'] = pd.to_numeric(profile_meta_info_df['ITS2 type abundance DB'])
            self.profile_meta_info_df = profile_meta_info_df

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
            self.profile_abundance_df = df
            pickle.dump(self.profile_abundance_df, open(os.path.join(self.cache_dir, 'profile_abund_df.p'), 'wb'))
            pickle.dump(self.smp_uid_to_name_dict, open(os.path.join(self.cache_dir, 'smp_uid_to_name_dict.p'), 'wb'))
            pickle.dump(self.smp_name_to_uid_dict, open(os.path.join(self.cache_dir, 'smp_name_to_uid_dict.p'), 'wb'))
            pickle.dump(self.prof_uid_to_local_abund_dict, open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict.p'), 'wb'))
            pickle.dump(self.prof_uid_to_global_abund_dict, open(os.path.join(self.cache_dir, 'prof_uid_to_global_abund_dict.p'), 'wb'))
            pickle.dump(self.prof_uid_to_name_dict, open(os.path.join(self.cache_dir, 'prof_uid_to_name_dict.p'), 'wb'))
            pickle.dump(self.prof_name_to_uid_dict, open(os.path.join(self.cache_dir, 'prof_name_to_uid_dict.p'), 'wb'))
            pickle.dump(self.profile_meta_info_df, open(os.path.join(self.cache_dir, 'profile_meta_info_df.p'), 'wb'))

    def plot_pcoa_of_cladal(self):
        class PCOAByClade:
            """Code for plotting a series of PCoAs for the sample distances and we will colour
            them according to the meta info"""
            def __init__(self, parent):
                self.parent = parent
                self.fig = plt.figure(figsize=(8, 8))
                # one row for each clade, for the between sample ordinations
                # one row for the clade proportion ordination
                # one row for the legends
                # one column per meta info category i.e. species, reef, reef_type, etc.
                self.gs = gridspec.GridSpec(5, 5)
                # axis for plotting the between sample ordinations
                self.ax_arr = [[] for _ in range(5)]
                self.clade_proportion_ax = []
                self.leg_axarr = []
                self.meta_info_categories = list(self.parent.experimental_metadata_info_df)

                self._setup_axarr()

            def _setup_axarr(self):
                # axis setup for the between sample ordinations
                for j in range(len(self.meta_info_categories)):  # for each meta info category
                    for i in range(len(self.parent.clades) + 1):  # each clade and the clade prop ordination
                        temp_ax = plt.subplot(self.gs[j:j + 1, i:i + 1])
                        temp_ax.set_xticks([])
                        temp_ax.set_yticks([])
                        temp_ax.set_facecolor('gray')
                        if j == 0:  # if this is subplot in top row
                            if i == 3:
                                temp_ax.set_title('Genera proportions', fontweight='bold', fontsize='x-small', )
                            else:
                                temp_ax.set_title(f'{self.parent.clade_genera_labels[i]}', fontweight='bold', fontsize='small', fontstyle='italic')
                        if i == 0:  # if this is subplot in first column
                            # then this is the cladal proportion
                            temp_ax.set_ylabel(self.meta_info_categories[j], fontweight='bold', fontsize='small')

                        if j == 4:  # if last row
                            temp_ax.set_xlabel('PC1', fontsize='x-small')
                        if i == 3:  # if last column
                            ax2 = temp_ax.twinx()
                            ax2.set_yticks([])
                            ax2.set_ylabel('PC2', fontsize='x-small')
                        self.ax_arr[j].append(temp_ax)

                # Then add the legend axis array
                # The x and y coordinates for the legen symbols can be set.
                # the x can be 0.1. The Y should be aligned for all meta categories
                # therefore we should set it according to the meta categorie with the largest number of levels.
                # This is species with 7 levels.
                x_vals = [0.1 for _ in range(7)]
                y_vals_odd = [y * 1/7  for y in range(7)]
                y_vals_even = [y * 1/8  for y in range(8)]

                for i in range(len(self.meta_info_categories)):
                    temp_ax = plt.subplot(self.gs[i:i+1, 4:5])
                    temp_ax.set_ylim(-0.2,1)
                    temp_ax.set_xlim(0,1)
                    temp_ax.invert_yaxis()
                    temp_ax.spines['top'].set_visible(False)
                    temp_ax.spines['bottom'].set_visible(False)
                    temp_ax.spines['right'].set_visible(False)
                    temp_ax.spines['left'].set_visible(False)
                    temp_ax.set_yticks([])
                    temp_ax.set_xticks([])

                    # Species
                    if i == 0:
                        colors = [self.parent.old_color_dict[s] for s in self.parent.species_category_list]
                        temp_ax.scatter(x_vals, y_vals_odd, color=colors)
                        for x, y, label in zip(x_vals, y_vals_odd, self.parent.species_category_labels):
                            temp_ax.text(x=x + 0.1, y=y, s=label, verticalalignment='center', fontstyle='italic')
                    # Reef
                    elif i==1:
                        colors = [self.parent.old_color_dict[r] for r in self.parent.reefs]
                        temp_ax.scatter(x_vals[:6], y_vals_even[1:7], color=colors)
                        for x, y, label in zip(x_vals, y_vals_even[1:7], self.parent.reefs):
                            temp_ax.text(x=x + 0.1, y=y, s=label, verticalalignment='center', fontstyle='italic')
                    # Reef type
                    elif i==2:
                        colors = [self.parent.old_color_dict[r] for r in self.parent.reef_types]
                        temp_ax.scatter(x_vals[:3], y_vals_odd[2:5], color=colors)
                        for x, y, label in zip(x_vals, y_vals_odd[2:5], self.parent.reef_types):
                            temp_ax.text(x=x + 0.1, y=y, s=label, verticalalignment='center', fontstyle='italic')
                    # Depth
                    elif i==3:
                        colors = [self.parent.old_color_dict[r] for r in self.parent.depths]
                        temp_ax.scatter(x_vals[:3], y_vals_odd[2:5], color=colors)
                        for x, y, label in zip(x_vals, y_vals_odd[2:5], self.parent.depths):
                            temp_ax.text(x=x + 0.1, y=y, s=f'{label}m', verticalalignment='center', fontstyle='italic')
                    # Season
                    elif i==4:
                        colors = [self.parent.old_color_dict[r] for r in self.parent.seasons]
                        temp_ax.scatter(x_vals[:2], y_vals_even[3:5], color=colors)
                        for x, y, label in zip(x_vals, y_vals_even[3:5], self.parent.seasons):
                            temp_ax.text(x=x + 0.1, y=y, s=label, verticalalignment='center', fontstyle='italic')


                apples = 'asdf'


            def plot_PCOA(self):
                self._plot_per_clade_ordinations()

                self._plot_clade_proportion_ordinations()

                plt.savefig(os.path.join(self.parent.figure_dir, 'ordination_figure.png'), dpi=1200)
                plt.savefig(os.path.join(self.parent.figure_dir, 'ordination_figure.svg'), dpi=1200)

            def _plot_clade_proportion_ordinations(self):
                # now plot up the clade_proportion ordination
                prop_explained_tot = sum(self.parent.clade_prop_pcoa_coords.loc['proportion_explained'])
                for i in range(len(self.meta_info_categories)):
                    pc_one_var = self.parent.clade_prop_pcoa_coords['PC1'].iat[-1] / prop_explained_tot
                    pc_two_var = self.parent.clade_prop_pcoa_coords['PC2'].iat[-1] / prop_explained_tot
                    color_list = []

                    uid_list = self.parent.clade_prop_pcoa_coords.index.values.tolist()[:-1]
                    for smp_uid in uid_list:
                        meta_value = self.parent.experimental_metadata_info_df.loc[smp_uid, self.meta_info_categories[i]]
                        color_list.append(self.parent.old_color_dict[meta_value])

                    self.ax_arr[i][3].scatter(x=self.parent.clade_prop_pcoa_coords['PC1'][:-1],
                                              y=self.parent.clade_prop_pcoa_coords['PC2'][:-1], marker='.',
                                              c=color_list, s=40, alpha=0.7, edgecolors='none')
                    self._write_var_explained(i, self.ax_arr[i][3], pc_one_var, pc_two_var)

            def _plot_per_clade_ordinations(self):
                for j in range(len(self.parent.clades)):  # for each clade
                    # We need to compute the pcoa coords for each clade. These will be the points plotted in the
                    # scatter for each of the different meta info for each clade
                    sample_clade_dist_df = self.parent.between_sample_clade_dist_df_dict[self.parent.clades[j]]
                    pcoa_output = pcoa(sample_clade_dist_df)
                    eig_tots = sum(pcoa_output.eigvals)
                    pc_one_var = pcoa_output.eigvals[0] / eig_tots
                    pc_two_var = pcoa_output.eigvals[1] / eig_tots

                    for i in range(len(self.meta_info_categories)):

                        color_list = []
                        # if i in [0,1]:
                        #     for smp_uid in list(sample_clade_dist_df):
                        #         r,g,b = self.new_color_dict[self.parent.metadata_info_df.loc[smp_uid, self.meta_info_categories[i]]]
                        #         color_list.append("#{0:02x}{1:02x}{2:02x}".format(r, g, b))
                        # else:
                        for smp_uid in list(sample_clade_dist_df):
                            color_list.append(self.parent.old_color_dict[self.parent.experimental_metadata_info_df.loc[
                                smp_uid, self.meta_info_categories[i]]])

                        self.ax_arr[i][j].scatter(x=pcoa_output.samples['PC1']*100, y=pcoa_output.samples['PC2']*100,
                                                  marker='.', c=color_list, s=40, alpha=0.7, edgecolors='none')
                        self._write_var_explained(i, self.ax_arr[i][j], pc_one_var, pc_two_var)

            def _write_var_explained(self, i, ax, pc_one_var, pc_two_var):
                if i == 0:
                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    height = y1-y0
                    width = x1-x0
                    text_x = x0 + 0.48 * width
                    text_y = y0 - 0.1 * height
                    ax.text(x=text_x, y=text_y, s=f'({pc_one_var:.2f}, {pc_two_var:.2f})', fontsize='x-small')
        pbc = PCOAByClade(parent=self)
        pbc.plot_PCOA()

    def plot_ternary_clade_proportions(self):
        class TernaryPlot:
            def __init__(self, parent):
                import ternary
                self.parent = parent
                self.fig = plt.figure(figsize=(5, 5))

                self.gs = gridspec.GridSpec(1, 1)
                point_ax = plt.subplot(self.gs[0,0])
                self.fig, self.tax_point = ternary.figure(ax=point_ax, scale=1)
                self._setup_tax_point()
                apples = 'asdf'

            def _setup_tax_point(self):
                self.tax_point.boundary(linewidth=1)
                self.tax_point.gridlines(color='black', multiple=0.1)
                # self.tax_point.gridlines(color='blue', multiple=0.025, linewidth=0.5)
                fontsize = 20
                # self.tax.set_title("Genera proportion", fontsize=fontsize)
                self.tax_point.left_corner_label("Symbiodinium", fontsize='small', fontstyle='italic')
                self.tax_point.right_corner_label("Durisdinium", fontsize='small', fontstyle='italic')
                self.tax_point.top_corner_label("Cladocopium", fontsize='small', fontstyle='italic')
                # self.tax_point.ticks(axis='lbr', linewidth=1, multiple=0.1, offset=0.025, tick_formats="%.2f")
                self.tax_point.get_axes().axis('off')
                self.tax_point.clear_matplotlib_ticks()

            def _plot_ternary_points(self):
                for sample_uid in self.parent.clade_proportion_df.index.values.tolist():
                    vals = [self.parent.clade_proportion_df.at[sample_uid, 'D'], self.parent.clade_proportion_df.at[sample_uid, 'C'], self.parent.clade_proportion_df.at[sample_uid, 'A']]
                    tot = sum(vals)
                    if tot != 0:
                        prop_tup = tuple([val/tot for val in vals])
                        self.tax_point.scatter([prop_tup], marker='o', color='black', s=20, alpha=0.5)
                plt.savefig(os.path.join(self.parent.figure_dir, 'ternary_figure.png'), dpi=1200)
                plt.savefig(os.path.join(self.parent.figure_dir, 'ternary_figure.svg'), dpi=1200)

        tp = TernaryPlot(parent=self)
        tp._plot_ternary_points()
        foo = 'bar'

    def make_dendrogram_with_meta_all_clades_sample_dists(self):
        """ We will produce a similar figure to the one that we have already produced for the types."""
        raise NotImplementedError

    def make_sample_balance_figure(self):
        class BalanceFigureMap:
            def __init__(self, parent):
                self.fig = plt.figure(figsize=(12, 12))
                self.parent = parent
                self.outer_gs = gridspec.GridSpec(2,2)
                self.depths = [1, 15, 30]
                self.seasons = ['Winter', 'Summer']
                self.reefs = ['Fsar', 'Tahla', 'Qita al Kirsh', 'Al Fahal', 'Shib Nazar', 'Abu Madafi']
                self.reef_types = ['Inshore', 'Midshelf', 'Offshore']

                self.species_hatch_dict = {
                    'G': '+', 'GX': 'x', 'M': '\\', 'P': 'o',
                    'PC': '.', 'SE': '/', 'ST': 'O'}

                self.species_category_list = ['M', 'G', 'GX', 'P', 'PC', 'SE', 'ST']
                self.species_category_labels = ['M. dichotoma', 'G. planulata', 'G. fascicularis', 'Porites spp.', 'P. verrucosa',
                                   'S. hystrix', 'S. pistillata']
                self.depth_height = 1
                self.depth_width = 6
                self.reef_height = 3 * self.depth_height
                self.reef_width = 1
                self.reef_type_height = 2 * self.reef_height
                self.reef_type_width = 1
                self.inner_rows = len(self.depths) * int(len(self.reefs) * 1 / 3) * self.depth_height
                self.inner_cols = self.reef_width + self.reef_type_width + (len(self.seasons) * self.depth_width)
                self.site_labels = ['Abu Madafi', "Shi'b Nazar", 'Al Fahal', 'Qita al-Kirsh', 'Tahla', 'Fsar']
                self.outer_gs = gridspec.GridSpec(2, 2)
                # axes that we will be plotting on
                self.large_map_ax = None
                self.small_map_ax = None


            def setup_plotting_space(self):
                """ We played with trying to get basemap to work but it is not worth the effort. There is too much
                wrong with it and it is constantly crashing and not finding modules or environmental variable. I have
                given up on trying to get it to work.
                I also tried working with geopandas. This was also more hassle than it was worth and I decided to 
                give up on this to. Instead, cartopy seems to be doing a great job. It was a little trickly to get
                working. After doing its pip install I had to down grade the shapely module to 1.5.17. See:
                https://github.com/conda-forge/cartopy-feedstock/issues/36
                NB When I tried making a new environment I found that cartopy was broken again. The trick was to install
                it using the conda install rather than pip as the pip install mas not installing some of the required
                libraries. conda install cartopy.
                """

                max_width = 0
                for i in range(4): # for each of the major gridspaces
                    if i == 0: # then plot map

                        self._draw_big_map()

                        self._draw_small_map()

                        self._reposition_small_map()
                        
                    # now draw the sampling balance boxes for each of the reef_types
                    else:
                        # then we are working on the nearshore

                        inner_gs = self._steup_inner_grid_spec_obj(i)

                        # setup the axes of the inner gs objects
                        self._setup_reef_type_axis(i, inner_gs)

                        self._setup_reef_axarr(i, inner_gs)

                        # season and then reef index. each inner list will have three ax
                        self._do_depth_plotting(i, inner_gs)

                print('saving fig')
                plt.savefig(os.path.join(self.parent.figure_dir, 'map_balance.png'), dpi=1200)
                plt.savefig(os.path.join(self.parent.figure_dir, 'map_balance.svg'), dpi=1200)

                apples = 'asdf'

            def _do_depth_plotting(self, i, inner_gs):
                max_width = 0
                depth_winter_ax_arr = [[[] for _ in range(int(len(self.reefs) * 1 / 3))] for _ in
                                       range(len(self.seasons))]
                for season_index, season_name in enumerate(self.seasons):  # for each season
                    for reef_index, reef_name in enumerate(self.reefs[(i - 1) * 2:((i - 1) * 2) + 2]):  # for each reef
                        for depth_index, depth_name in enumerate(self.depths):  # for each depth
                            col_end, col_start, row_end, row_start = self._get_inner_gs_indices(depth_index,
                                                                                                reef_index,
                                                                                                season_index)
                            temp_depth_ax = self._format_temp_depth_axes(
                                col_end, col_start, depth_index, inner_gs, reef_index, row_end,
                                row_start, season_index)
                            depth_winter_ax_arr[season_index][reef_index].append(temp_depth_ax)

                            # here we can do the actual plotting if we want. This will save us
                            # having to iter back through the axarr structure
                            # we should be sure to plot this in the same order as the dendrogram meta info
                            bar_widths, bar_ys, species_c_list = self._get_bar_plot_info(
                                depth_name, i, max_width, reef_name, season_name)
                            self._plot_bars(bar_widths, bar_ys, species_c_list, temp_depth_ax)

            def _plot_bars(self, bar_widths, bar_ys, species_c_list, temp_depth_ax):
                bars = []
                if sum(bar_widths) != 0:
                    for y, w, c in zip(bar_ys, bar_widths, species_c_list):
                        bars.append(temp_depth_ax.barh(y=y, width=w,
                                                       height=1 / len(self.species_category_list), align='edge',
                                                       fill=True, edgecolor='black', color=c))
                else:
                    temp_depth_ax.set_fc('white')
                    temp_depth_ax.text(2.5, 0.5, 'Not Available', horizontalalignment='center',
                                       verticalalignment='center')

            def _get_bar_plot_info(self, depth_name, i, max_width, reef_name, season_name):
                bar_ys = [_ * (1 / len(self.species_category_list)) for _ in range(len(self.species_category_list))]
                bar_width_dict = defaultdict(int)
                for spec_cat in self.species_category_list:
                    rows_of_interest = self.parent.experimental_metadata_info_df.loc[
                        (self.parent.experimental_metadata_info_df['species'] == spec_cat) &
                        (self.parent.experimental_metadata_info_df['reef'] == reef_name) &
                        (self.parent.experimental_metadata_info_df['season'] == season_name) &
                        (self.parent.experimental_metadata_info_df['depth'] == depth_name) &
                        (self.parent.experimental_metadata_info_df['reef_type'] == self.reef_types[i - 1])]
                    bar_width_dict[spec_cat] += len(rows_of_interest.index.values.tolist())
                bar_widths = [bar_width_dict[spec_cat] for spec_cat in self.species_category_list]
                species_c_list = [
                    self.parent.old_color_dict[spec_cat] for spec_cat in self.species_category_list]
                species_h_list = [
                    self.species_hatch_dict[spec_cat] for spec_cat in self.species_category_list]
                if max(bar_widths) > max_width:
                    max_width = max(bar_widths)
                return bar_widths, bar_ys, species_c_list

            def _format_temp_depth_axes(self, col_end, col_start, depth_index, inner_gs, reef_index, row_end, row_start,
                                        season_index):
                temp_depth_ax = plt.subplot(inner_gs[row_start:row_end, col_start:col_end])
                if season_index == 0:
                    self._format_label_ax(temp_depth_ax, label=f'{self.depths[depth_index]}m')
                else:
                    self._format_label_ax(temp_depth_ax)
                temp_depth_ax.set_xlim(0, 5.1)
                if depth_index == 0 and reef_index == 0:
                    temp_depth_ax.spines['top'].set_visible(True)
                    ax2 = temp_depth_ax.twiny()
                    ax2.set_xlim(0, 5.1)
                    ax2.set_xticks([5])
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['bottom'].set_visible(False)
                    if season_index == 0:
                        ax2.set_xlabel('winter', labelpad=-10, fontweight='bold')
                    else:
                        ax2.set_xlabel('summer', labelpad=-10, fontweight='bold')
                temp_depth_ax.set_fc('lightgray')
                return temp_depth_ax

            def _get_inner_gs_indices(self, depth_index, reef_index, season_index):
                row_start = reef_index * len(self.depths) + depth_index
                row_end = row_start + self.depth_height
                col_start = self.reef_type_width + self.reef_width + season_index * self.depth_width
                col_end = col_start + self.depth_width
                return col_end, col_start, row_end, row_start

            def _setup_reef_axarr(self, i, inner_gs):
                reef_axarr = []
                for j in range(2):
                    temp_reef_ax = plt.subplot(inner_gs[j * self.reef_height:(j * self.reef_height) + self.reef_height,
                                               self.reef_type_width:self.reef_type_width + self.reef_width])
                    self._format_label_ax(temp_reef_ax, label=self.reefs[((i - 1) * 2) + j])
                    reef_axarr.append(temp_reef_ax)

            def _setup_reef_type_axis(self, i, inner_gs):
                reef_type_ax = plt.subplot(inner_gs[0:self.reef_type_height, 0:self.reef_type_width])
                self._format_label_ax(reef_type_ax, label=self.reef_types[i - 1])

            def _steup_inner_grid_spec_obj(self, i):
                # set up the inner gs objects
                if i == 1:
                    inner_gs = self.inner_gs_map = gridspec.GridSpecFromSubplotSpec(
                        self.inner_rows, self.inner_cols, subplot_spec=self.outer_gs[0:1, 1:2])
                elif i == 2:
                    inner_gs = self.inner_gs_map = gridspec.GridSpecFromSubplotSpec(
                        self.inner_rows, self.inner_cols, subplot_spec=self.outer_gs[1:2, 0:1])
                else:  # i == 3
                    inner_gs = self.inner_gs_map = gridspec.GridSpecFromSubplotSpec(
                        self.inner_rows, self.inner_cols, subplot_spec=self.outer_gs[1:2, 1:2])
                return inner_gs

            def _format_label_ax(self, ax, label=None):
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                if label is not None:
                    ax.set_ylabel(label, rotation='vertical', fontweight='bold')

            def _reposition_small_map(self):
                # now readjust position
                plt.draw()
                large_map_bbox = self.large_map_ax.get_position()
                small_map_bbox = self.small_map_ax.get_position()
                self.small_map_ax.set_position([
                    large_map_bbox.x1 - small_map_bbox.width,
                    large_map_bbox.y0,
                    small_map_bbox.width,
                    small_map_bbox.height])

            def _draw_small_map(self):
                self._calc_pos_of_inset_ax_and_init()
                self.small_map_ax.set_extent(extents=(31, 60, 5, 33))
                land_110m, ocean_110m, boundary_110m = self._get_naural_earth_features_small_map()
                self._draw_natural_earth_features_small_map(land_110m, ocean_110m, boundary_110m)
                self._annotate_small_map()

            def _draw_big_map(self):
                self.large_map_ax = plt.subplot(self.outer_gs[0:1, 0:1], projection=ccrs.PlateCarree(), zorder=1)
                x0_extent, x1_extent, y0_extent, y1_extent = 38.7, 39.3, 22.0, 22.6

                from cartopy.io.shapereader import Reader
                from cartopy.feature import ShapelyFeature
                reefs_shape_path_py = '/Users/humebc/Downloads/14_001_WCMC008_CoralReefs2018_v4/01_Data/WCMC008_CoralReef2018_Py_v4.shp'
                if os.path.exists(os.path.join(self.parent.cache_dir, 'reef_reader_trimmed')):
                    reader = pickle.load(open(os.path.join(self.parent.cache_dir, 'reef_reader_trimmed'), 'rb'))
                elif os.path.exists(os.path.join(self.parent.cache_dir, 'reef_reader')):
                    reader = pickle.load(open(os.path.join(self.parent.cache_dir, 'reef_reader'), 'rb'))
                    reader = self._trim_reader(reader)
                    pickle.dump(reader, open(os.path.join(self.parent.cache_dir, 'reef_reader_trimmed'), 'wb'))
                else:
                    reader = Reader(reefs_shape_path_py)
                    pickle.dump(reader, open(os.path.join(self.parent.cache_dir, 'reef_reader'), 'wb'))
                    reader = self._trim_reader(reader)
                    pickle.dump(reader, open(os.path.join(self.parent.cache_dir, 'reef_reader_trimmed'), 'wb'))

                geom = reader.geometries()
                # for g in geom:
                #     look = 'this'
                # for g in geom:
                #     look = 'this'
                self.large_map_ax.set_extent(extents=(x0_extent, x1_extent, y0_extent, y1_extent))
                shap_f = ShapelyFeature(geom, ccrs.PlateCarree(), edgecolor='black', linewidth=0.5)

                # shape_feature = ShapelyFeature(Reader(reefs_shape_path_py).geometries(),
                #                                ccrs.PlateCarree(), edgecolor='black')
                self.large_map_ax.add_feature(shap_f, facecolor='#C1DD79', zorder=2)
                print('saving')
                # self.fig.savefig(os.path.join(self.parent.figure_dir, 'map_testing.png'), dpi=1200)
                # self.fig.savefig(os.path.join(self.parent.figure_dir, 'map_testing.svg'), dpi=1200)
                # sys.exit()

                # land_10m, ocean_10m = self._get_naural_earth_features_big_map()
                # self._draw_natural_earth_features_big_map(land_10m, ocean_10m)
                self._add_land_and_sea_to_inset(self.large_map_ax, x0_extent, x1_extent, y0_extent, y1_extent)
                self._put_gridlines_on_large_map_ax()
                self._annotate_big_map()
                # self._draw_reefs_on_map(self.large_map_ax)

            def _trim_reader(self, reader):
                new_data = []
                for r in reader._data:
                    if 'SAU' in r['ISO3']:
                        new_data.append(r)
                reader._data = new_data
                return reader

            def _add_land_and_sea_to_inset(self, map_ax, x0_extent, x1_extent, y0_extent, y1_extent):
                x_s, y_s = self._add_kml_file_to_ax(ax=map_ax,
                                                    kml_path=os.path.join(self.parent.gis_input_base_path, 'restrepo_coastline.kml'))
                poly_xy = [[x, y] for x, y in zip(x_s, y_s)]
                # add top right and bottom right
                poly_xy.extend([[x1_extent, y1_extent], [x1_extent, y0_extent]])
                land_poly = Polygon(poly_xy, closed=True, fill=True, color=(238 / 255, 239 / 255, 219 / 255), zorder=3)
                map_ax.add_patch(land_poly)
                # now do the seq poly
                poly_xy = [[x, y] for x, y in zip(x_s, y_s)]
                # add top left and bottom left
                poly_xy.extend([[x0_extent, y1_extent], [x0_extent, y0_extent]])
                sea_poly = Polygon(poly_xy, closed=True, fill=True, color=(136 / 255, 182 / 255, 224 / 255), zorder=1)
                map_ax.add_patch(sea_poly)

            def _add_kml_file_to_ax(self, ax, kml_path, linewidth=0.8, linestyle='-', color='black', ):
                with open(kml_path, 'r') as f:
                    file = [line.rstrip().lstrip() for line in f]
                for i, line in enumerate(file):
                    if '<coordinates>' in line:
                        coords = file[i + 1]
                        break
                coords_tup_list_str = coords.split(' ')
                x_y_tups_of_feature = []
                for tup in coords_tup_list_str:
                    x_y_tups_of_feature.append([float(_) for _ in tup.split(',')[:-1]])
                x_s = [_[0] for _ in x_y_tups_of_feature]
                y_s = [_[1] for _ in x_y_tups_of_feature]
                ax.plot(x_s, y_s, linewidth=linewidth, linestyle=linestyle, color=color, zorder=3)
                return x_s, y_s

            def _draw_reefs_on_map(self, map_ax):
                for i in range(1, 33, 1):
                    kml_path = os.path.join(self.parent.gis_input_base_path, f'reef_{i}.kml')
                    with open(kml_path, 'r') as f:
                        file = [line.rstrip().lstrip() for line in f]
                    for i, line in enumerate(file):
                        if '<coordinates>' in line:
                            coords = file[i + 1]
                            break
                    coords_tup_list_str = coords.split(' ')
                    x_y_tups_of_feature = []
                    for tup in coords_tup_list_str:
                        x_y_tups_of_feature.append([float(_) for _ in tup.split(',')[:-1]])
                    x_s = [_[0] for _ in x_y_tups_of_feature]
                    y_s = [_[1] for _ in x_y_tups_of_feature]
                    poly_xy = [[x, y] for x, y in zip(x_s, y_s)]
                    reef_poly = Polygon(poly_xy, closed=True, fill=True, edgecolor='None', color='red', alpha=0.2, zorder=4)
                    map_ax.add_patch(reef_poly)

            def _annotate_small_map(self):
                recticle_x = (38.7 + 39.3) / 2
                recticle_Y = (22.0 + 22.6) / 2
                self.small_map_ax.plot(recticle_x, recticle_Y, 'k+', markersize=10)

            def _draw_natural_earth_features_small_map(self, land_110m, ocean_110m, boundary_110m):
                """NB the RGB must be a tuple in a list and the R, G, B must be given as a value between 0 and 1"""
                self.small_map_ax.add_feature(land_110m, facecolor=[(238 / 255, 239 / 255, 219 / 255)],
                                              edgecolor='black', linewidth=0.2)
                self.small_map_ax.add_feature(ocean_110m, facecolor=[(136 / 255, 182 / 255, 224 / 255)],
                                              edgecolor='black', linewidth=0.2)
                self.small_map_ax.add_feature(boundary_110m, edgecolor='gray', linewidth=0.2, facecolor='None')

            def _get_naural_earth_features_small_map(self):
                land_110m = cartopy.feature.NaturalEarthFeature(category='physical', name='land',
                                                                scale='110m')
                ocean_110m = cartopy.feature.NaturalEarthFeature(category='physical', name='ocean',
                                                                 scale='110m')
                boundary_110m = cartopy.feature.NaturalEarthFeature(category='cultural',
                                                                    name='admin_0_boundary_lines_land', scale='110m')
                return land_110m, ocean_110m, boundary_110m

            def _calc_pos_of_inset_ax_and_init(self):
                """Creating an inset axis is quite difficult (maybe impossible to do) using the inset tools
                or gridspec. The best way that I have found to do it is to use the fig.add_axes method that allows
                the custom positioning of the axes in figure coordinates (that run from 0,0 to 1,1).
                NB I was calculating the figure points where I wanted the plot to go using the below:
                    To do this
                    we need to convert from the axdata units to display units and then back into figure data units.
                    We can then finally place the axes using these units.
                However, when we apply the projection and extents the axes move slightly. In the end we can use this
                method:
                https://stackoverflow.com/a/45538400/5516420
                And simply use the .get_position() method from the axes object to place our small map relatively.
                This will be done after we have drawn all the feature of the small map. Please note though, that
                the getting of the coordinates is largely redundant now as we could just have put this plot anywhere
                in the figure and then repositioned later. But I will leave the code as it is as it will take longer
                to change and it is useful for reference.
                """
                # now work towards the sub map plot within the main map window
                dis_data = self.large_map_ax.transData.transform([(39.0, 22.0), (39.3, 22.2)])
                inv = self.fig.transFigure.inverted()
                fig_data = inv.transform(dis_data)
                width = fig_data[1][0] - fig_data[0][0]
                height = fig_data[1][1] - fig_data[0][1]
                self.small_map_ax = self.fig.add_axes([fig_data[0][0], fig_data[0][1], width, height], zorder=2,
                                                      projection=ccrs.PlateCarree())

            def _annotate_big_map(self, zorder=4):
                x_site_coords = [38.778333, 38.854283, 38.960533, 38.992800, 39.055275, 39.030267, ]
                y_site_coords = [22.109143, 22.322533, 22.306233, 22.430717, 22.308564, 22.232617, ]
                self.large_map_ax.plot(x_site_coords[:2], y_site_coords[:2], 'ko', zorder=zorder)
                self.large_map_ax.plot(x_site_coords[2:4], y_site_coords[2:4], 'ks', zorder=zorder)
                self.large_map_ax.plot(x_site_coords[4:6], y_site_coords[4:6], 'k^', zorder=zorder)
                # Abu Madafi
                self.large_map_ax.text(x_site_coords[0] + 0.01, y_site_coords[0] + 0.01, self.site_labels[0], fontsize='medium', zorder=zorder)
                # Shi'b Nazar
                self.large_map_ax.text(x_site_coords[1] - 0.08, y_site_coords[1] + 0.02, self.site_labels[1], fontsize='medium', zorder=zorder)
                # Al Fahal
                self.large_map_ax.text(x_site_coords[2] - 0.06, y_site_coords[2] - 0.04, self.site_labels[2], fontsize='medium', zorder=zorder)
                # Qita al-Kirsh
                self.large_map_ax.text(x_site_coords[3] - 0.1, y_site_coords[3] + 0.02, self.site_labels[3], fontsize='medium', zorder=zorder)
                # Tahla
                self.large_map_ax.text(x_site_coords[4] - 0.04, y_site_coords[4] + 0.02, self.site_labels[4], fontsize='medium', zorder=zorder)
                # Fsar
                self.large_map_ax.text(x_site_coords[5] - 0.06, y_site_coords[5] - 0.03, self.site_labels[5], fontsize='medium', zorder=zorder)
                self.large_map_ax.plot(39.14, 22.57, 'k^', zorder=zorder)
                self.large_map_ax.text(39.16, 22.57, 'Inshore', verticalalignment='center')
                self.large_map_ax.plot(39.14, 22.51, 'ks', zorder=zorder)
                self.large_map_ax.text(39.16, 22.51, 'Midshore', verticalalignment='center')
                self.large_map_ax.plot(39.14, 22.45, 'ko', zorder=zorder)
                self.large_map_ax.text(39.16, 22.45, 'Offshore', verticalalignment='center')
                r1 = patches.Rectangle(
                    xy=(39.10, 22.4), width=0.2, height=0.2, fill=True, facecolor='white', edgecolor='black', linewidth=1, zorder=3, alpha=0.4)
                self.large_map_ax.add_patch(r1)
                apples = 'asdf'


            def _draw_natural_earth_features_big_map(self, land_10m, ocean_10m):
                """NB the RGB must be a tuple in a list and the R, G, B must be given as a value between 0 and 1"""
                self.large_map_ax.add_feature(land_10m, facecolor=[(238 / 255, 239 / 255, 219 / 255)],
                                              edgecolor='black')
                self.large_map_ax.add_feature(ocean_10m, facecolor=[(136 / 255, 182 / 255, 224 / 255)],
                                              edgecolor='black')

            def _get_naural_earth_features_big_map(self):
                land_10m = cartopy.feature.NaturalEarthFeature(category='physical', name='land',
                                                               scale='10m')
                ocean_10m = cartopy.feature.NaturalEarthFeature(category='physical', name='ocean',
                                                                scale='10m')
                return land_10m, ocean_10m

            def _put_gridlines_on_large_map_ax(self):
                """ Although there is a GeoAxis.gridlines() method, this method does not yet allow a lot of
                bespoke options. If we want to only put the labels on the top and left then we have to
                generate a Gridliner object (normally returned by GeoAxis.gridlines() ourselves. We then need
                to manually change the xlabels_bottom and ylabels_right attributes of this Gridliner object.
                We then draw it by adding it to the GeoAxis._gridliners list."""
                xlocs = [38.6, 38.8, 39.0, 39.2, 39.4]
                ylocs = [22.0, 22.2, 22.4, 22.6]

                if xlocs is not None and not isinstance(xlocs, mticker.Locator):
                    xlocs = mticker.FixedLocator(xlocs)
                if ylocs is not None and not isinstance(ylocs, mticker.Locator):
                    ylocs = mticker.FixedLocator(ylocs)
                g1 = Gridliner(
                    axes=self.large_map_ax, crs=ccrs.PlateCarree(), draw_labels=True,
                    xlocator=xlocs, ylocator=ylocs)
                g1.xlabels_bottom = False
                g1.ylabels_right = False
                self.large_map_ax._gridliners.append(g1)

        blfm = BalanceFigureMap(parent=self)
        blfm.setup_plotting_space()



        apples = 'asdf'

    def make_dendrogram_with_meta_all_clades(self, high_low=None):
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
        which has its units in display units. You then have to  this back to data units.
        This link for getting the bbox from the annotation:
        https://matplotlib.org/api/text_api.html#matplotlib.text.Annotation.get_window_extent
        This link for doing the transofrmations into the correct coordinates space:
        https://matplotlib.org/users/transforms_tutorial.html
        """
        if high_low == 'background':
            fig = plt.figure(figsize=(7, 14))
        else:
            fig = plt.figure(figsize=(7, 12))
        # required for getting the bbox of the text annotations
        fig.canvas.draw()

        apples = 'asdf'
        # order: dendro, label, species, depth, reef_type, season
        widths = [12,32,7,6,3,3,2]
        axarr = self._setup_grid_spec_and_axes_for_dendro_and_meta_fig_all_clades(widths, high_low)
        for i in range(len(self.clades)):
            self._make_dendrogram_with_meta_fig_for_all_clades(i, axarr, high_low)
        print('Saving image')
        if not high_low:
            plt.savefig(os.path.join(self.figure_dir, f'dendro_figure_{self.profile_distance_method}_{self.cutoff_abund}.png'), dpi=1200)
            plt.savefig(os.path.join(self.figure_dir, f'dendro_figure_{self.profile_distance_method}_{self.cutoff_abund}.svg'), dpi=1200)
        else:
            plt.savefig(
                os.path.join(self.figure_dir, f'dendro_figure_{self.profile_distance_method}_{high_low}.png'),
                dpi=1200)
            plt.savefig(
                os.path.join(self.figure_dir, f'dendro_figure_{self.profile_distance_method}_{high_low}.svg'),
                dpi=1200)

    def _make_dendrogram_with_meta_fig_for_all_clades(self, clade_index, axarr, high_low):
        if not high_low:
            profile_abund_cutoff = self.profile_abundance_df_cutoff
            between_profile_clade_dist_cct_specific_df_dict = self.between_profile_clade_dist_cct_specific_df_dict
            prof_uid_to_local_abund_dict_post_cutoff = self.prof_uid_to_local_abund_dict_post_cutoff
        elif high_low == 'high':
            profile_abund_cutoff = self.profile_abundance_df_cutoff_high
            between_profile_clade_dist_cct_specific_df_dict = self.profile_distance_df_dict_cutoff_high
            prof_uid_to_local_abund_dict_post_cutoff = self.prof_uid_to_local_abund_dict_cutoff_high
        elif high_low == 'background':
            profile_abund_cutoff = self.profile_abundance_df_cutoff_background
            between_profile_clade_dist_cct_specific_df_dict = self.profile_distance_df_dict_cutoff_background
            prof_uid_to_local_abund_dict_post_cutoff = self.prof_uid_to_local_abund_dict_cutoff_background
        else: # abund cuttoff == low
            profile_abund_cutoff = self.profile_abundance_df_cutoff_low
            between_profile_clade_dist_cct_specific_df_dict = self.profile_distance_df_dict_cutoff_low
            prof_uid_to_local_abund_dict_post_cutoff = self.prof_uid_to_local_abund_dict_cutoff_low


        clade = self.clades[clade_index]
        # Plot the dendrogram in first axes
        dendro_info = self._make_dendrogram_figure(
            clade=clade, ax=axarr[clade_index + 1][0], dist_df=between_profile_clade_dist_cct_specific_df_dict[clade],
            local_abundance_dict=prof_uid_to_local_abund_dict_post_cutoff, plot_labels=False)
        if clade_index == 0:
            axarr[clade_index + 1][0].set_yticks([0.0, 1.0])
        else:
            axarr[clade_index + 1][0].set_yticks([])

        title_list = ['Symbiodinium', 'Cladocopium', 'Durisdinium']
        axarr[clade_index + 1][0].set_ylabel(ylabel=title_list[clade_index], fontweight='bold', fontstyle='italic', fontsize='small')
        self._remove_spines_from_dendro(axarr[clade_index + 1], clade_index=clade_index)

        # get the uids in order for the profiles in the dendrogram
        ordered_prof_uid_list = []
        prof_uid_to_y_loc_dict = {}
        for y_loc, lab_str in dendro_info['tick_to_profile_name_dict'].items():
            temp_uid = self.prof_name_to_uid_dict[lab_str.split(' ')[0]]
            ordered_prof_uid_list.append(temp_uid)
            prof_uid_to_y_loc_dict[temp_uid] = y_loc

        # Plot labels in second axes
        self._plot_labels_plot_for_dendro_and_meta_fig(axarr[clade_index + 1][0], dendro_info, axarr[clade_index + 1][1])
        if clade_index == 0:
            axarr[clade_index + 1][1].set_title('ITS2 type profile name', fontsize='x-small', fontweight='bold')


        # for each ITS2 type profile we will need to get the samples that the profile was found in
        # then we need to look up each of the samples and see which of the parameters it refers to.
        # as such that first look up of which samples the profiles were found in can be put into a dict
        # for use in each of the meta plots.
        # How to represent the mixed states is a little tricky. I think perhaps we should just use an eveness
        # index, where a very uneven distribution is light grey (i.e. almost one of the categories and
        # the more even distribution is closer to black (i.e. more of a mix).
        # to make the grey code its probably easiest to make an RGB tupple scaling from 255,255,255 which is
        # white, to 0,0,0 which is black. This would be scaled against the eveness.

        profile_uid_to_sample_uid_list_dict = self._generate_profile_uid_to_sample_uid_list_dict(
            profile_abund_cutoff_df=profile_abund_cutoff)

        # we will work with a class for doing the meta plotting as it will be quite involved
        mip = MetaInfoPlotter(parent_analysis=self, ordered_uid_list=ordered_prof_uid_list, meta_axarr=axarr[clade_index + 1][2:],
                              prof_uid_to_smpl_uid_list_dict=profile_uid_to_sample_uid_list_dict,
                              prof_uid_to_y_loc_dict=prof_uid_to_y_loc_dict, dend_ax=axarr[clade_index + 1][0], sub_cat_axarr=axarr[clade_index][2:], clade_index=clade_index)
        mip.plot_species_meta()
        mip.plot_reef_meta()
        mip.plot_depth_meta()
        mip.plot_reef_type()
        mip.plot_season()

    def _generate_profile_uid_to_sample_uid_list_dict(self, profile_abund_cutoff_df, clade=None):

        profile_uid_to_sample_uid_list_dict = defaultdict(list)
        if clade is None:
            for prof_uid in list(profile_abund_cutoff_df):
                self._pop_prof_uid_to_smp_name_dd_list(prof_uid, profile_uid_to_sample_uid_list_dict, profile_abund_cutoff_df)
        else:
            for prof_uid in [uid for uid in list(profile_abund_cutoff_df) if clade.upper() in self.prof_uid_to_name_dict[uid]]:
                self._pop_prof_uid_to_smp_name_dd_list(prof_uid, profile_uid_to_sample_uid_list_dict, profile_abund_cutoff_df)
        return profile_uid_to_sample_uid_list_dict

    def _pop_prof_uid_to_smp_name_dd_list(self, prof_uid, profile_uid_to_sample_uid_list_dict, profile_abund_cutoff_df):
        temp_series = profile_abund_cutoff_df[prof_uid]
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
            clade=clade, ax=axarr[1][0], dist_df=self.between_profile_clade_dist_cct_specific_df_dict[clade],
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
                              prof_uid_to_y_loc_dict=prof_uid_to_x_loc_dict, dend_ax=axarr[1][0], sub_cat_axarr=axarr[0][2:])
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
                axarr[0].set_title('UniFrac distance', fontsize='x-small', fontweight='bold')
        else:
            axarr[0].set_ylabel('UniFrac distance', fontsize='x-small', fontweight='bold')

        axarr[0].spines['top'].set_visible(False)
        axarr[0].spines['bottom'].set_visible(False)
        axarr[0].spines['left'].set_visible(False)
        axarr[0].set_xticks([])

    def _plot_labels_plot_for_dendro_and_meta_fig(self, dend_ax, dendro_info, labels_ax):
        # make the x axis limits of the labels plot exactly the same as the dendrogram plot
        # then we can use the dendrogram plot x coordinates to plot the labels in the labels plot.
        labels_ax.set_ylim(dend_ax.get_ylim())

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
        x_val_buffer = 0.02  # the distance to leave between the line and the label
        for ann in annotation_list:
            bbox = ann.get_window_extent()
            inv = labels_ax.transData.inverted()
            bbox_data = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
            # Getting the center line of the label
            line_y = (bbox_data[1][1] + bbox_data[0][1]) / 2

            if bbox_data[0][0] > min_gap_to_plot and bbox_data[0][
                0] > x_val_buffer:  # then we should draw the connecting lines
                # the left connecting line
                lines.append(
                    hierarchy_sp.LineInfo([(min_gap_to_plot, line_y), (bbox_data[0][0] - x_val_buffer, line_y)],
                                          thickness=0.5, color='black'))
                # the right connecting line
                lines.append(
                    hierarchy_sp.LineInfo([(bbox_data[1][0] + x_val_buffer, line_y), (1 - min_gap_to_plot, line_y)],
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
        for y_loc, lab_str in dendro_info['tick_to_profile_name_dict'].items():
            # fig.canvas.draw()
            annotation_list.append(
                labels_ax.annotate(s=lab_str, xy=(0.5, y_loc), horizontalalignment='center',
                                   verticalalignment='center', fontsize='xx-small', fontweight='bold'))
        return annotation_list

    def _setup_grid_spec_and_axes_for_dendro_and_meta_fig_all_clades(self, list_of_widths, high_low):
        # in order (sub-cat, clade A, clade C, clade D)
        # We will set the hiehgts of the genera plots in proportion to the number of types for each genus
        # in the cutoff df
        if not high_low:
            profile_abund_cutoff = self.profile_abundance_df_cutoff
        elif high_low == 'high':
            profile_abund_cutoff = self.profile_abundance_df_cutoff_high
        elif high_low == 'background':
            profile_abund_cutoff = self.profile_abundance_df_cutoff_background
        else: # abund cuttoff == low
            profile_abund_cutoff = self.profile_abundance_df_cutoff_low

        dd_clade_counter = defaultdict(int)

        for uid in list(profile_abund_cutoff):
            dd_clade_counter[self.profile_meta_info_df.loc[uid]['Clade']] += 1

        # this is set from the original hardcoded numbers we can always adjust if needs be
        total = 25+42+24
        total_number_of_cutoff_profiles = len(list(profile_abund_cutoff))
        A_height = int((dd_clade_counter['A'] / total_number_of_cutoff_profiles) * total)
        C_height = int((dd_clade_counter['C'] / total_number_of_cutoff_profiles) * total)
        D_height = int((dd_clade_counter['D'] / total_number_of_cutoff_profiles) * total)

        plot_height_list = [6,A_height,C_height,D_height]

        gs = gridspec.GridSpec( sum(plot_height_list), sum(list_of_widths))
        # 2d list where each list is a column contining multiple axes

        axarr = []
        # first set of axes that will be used to put the subcategory labels for the metainfo
        sub_cat_label_ax_list = []
        for i in range(len(list_of_widths)):
            temp_ax = plt.subplot(gs[:plot_height_list[0],sum(list_of_widths[:i]):sum(list_of_widths[: i + 1])])
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
            for i in range(len(list_of_widths)):
                if clade == 'A':
                    col_index_start = plot_height_list[0]
                    col_index_end = sum(plot_height_list[:2])
                elif clade == 'C':
                    col_index_start = sum(plot_height_list[:2])
                    col_index_end = sum(plot_height_list[:3])
                else:
                    col_index_start = sum(plot_height_list[:3])
                    col_index_end = sum(plot_height_list[:4])

                clade_ax_list.append(plt.subplot(gs[col_index_start:col_index_end, sum(list_of_widths[:i]):sum(list_of_widths[: i + 1])]))
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
                    clade=clade, ax=axarr[0], dist_df=self.between_profile_clade_dist_df_dict[clade],
                    local_abundance_dict = self.prof_uid_to_local_abund_dict)

                # then the cct specific
                self._make_dendrogram_figure(
                    clade=clade, ax=axarr[1], dist_df=self.between_profile_clade_dist_cct_specific_df_dict[clade],
                    local_abundance_dict=self.prof_uid_to_local_abund_dict_post_cutoff)

                plt.tight_layout()
                plt.savefig(os.path.join(self.figure_dir, f'paired_dendogram_{clade}.png'))
                plt.savefig(os.path.join(self.figure_dir, f'paired_dendogram_{clade}.svg'))


        else:
            # draw a single dendrogram per clade
            for clade in self.clades:
                fig, ax = plt.subplots(figsize=(8, 12))
                self._make_dendrogram_figure(clade=clade, ax=ax, dist_df=self.between_profile_clade_dist_df_dict[clade],
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
                                         default_line_thickness=0.5, leaf_rotation=90, no_labels=not plot_labels, link_color_func=lambda k: '#000000', orientation='left')

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
        for index, row in self.profile_abundance_df.iterrows():
            values.extend(row.iloc[row.nonzero()].values.tolist())
        temp_series = pd.Series(values)
        print(len(temp_series))
        ax_zero_second = ax_arr[0].twinx()
        sns.distplot(temp_series, hist=False, kde=True,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=ax_zero_second, norm_hist=False)
        sns.distplot(temp_series, hist=True, kde=False,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=ax_arr[0], norm_hist=False)


        # hist = temp_series.hist(bins=50, ax=ax_arr[0])

        # Now do the same plot with the 0.06 cutoff applied
        cutoff = 0.05
        cut_off_values = [a for a in values if a > cutoff]
        temp_series = pd.Series(cut_off_values)
        ax_one_second = ax_arr[1].twinx()
        sns.distplot(temp_series, hist=False, kde=True,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=ax_one_second, norm_hist=False)
        sns.distplot(temp_series, hist=True, kde=False,
                     bins=50, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=ax_arr[1], norm_hist=False)
        # hist = temp_series.hist(bins=100, ax=ax_arr[1])



        # f.suptitle('Relative abundance of ITS2 type profile in sample', fontsize=14, x=0.5, y=0.05)
        ax_arr[0].set_ylabel('Frequency of observation', fontsize=10)
        ax_arr[1].set_ylabel('Frequency of observation', fontsize=10)
        ax_arr[0].set_title('Before 0.05 cutoff')
        ax_arr[1].set_title('After 0.05 cutoff')
        ax_zero_second.set_ylabel('Density', fontsize=10, rotation=270)
        ax_one_second.set_ylabel('Density', fontsize=10, rotation=270)
        ax_zero_second.set_ylim((0,5))
        ax_one_second.set_ylim((0, 5))
        f.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, 'hist.png'), dpi=1200)
        plt.savefig(os.path.join(self.figure_dir, 'hist.svg'), dpi=1200)

    def create_profile_df_with_cutoff(self):
        """Creates a new df from the old df that has all of the values below the cutoff_abundance threshold
        made to 0. We will also calculate a new prof_uid_to_local_abund_dict_post_cutoff dictionary.
        """
        cutoff_abund_str = str(self.cutoff_abund)
        if os.path.exists(os.path.join(self.cache_dir, f'prof_df_cutoff_{cutoff_abund_str}.p')):
            self.profile_abundance_df_cutoff = pickle.load(open(os.path.join(self.cache_dir, f'prof_df_cutoff_{cutoff_abund_str}.p'), 'rb'))
            self.prof_uid_to_local_abund_dict_post_cutoff = pickle.load(open(os.path.join(self.cache_dir, f'prof_uid_to_local_abund_dict_post_cutoff_{cutoff_abund_str}.p'), 'rb'))
        else:
            num_profs_pre_cutoff = self._report_profiles_before_cutoff()
            # make new df from copy of old df
            self.profile_abundance_df_cutoff = self.profile_abundance_df.copy()
            # change values below cutoff to 0
            self.profile_abundance_df_cutoff = self.profile_abundance_df_cutoff.mask(cond=self.profile_abundance_df_cutoff < self.cutoff_abund, other=0)
            # now drop columns with 0

            # now check to see if there are any type profiles that no longer have associations
            # https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
            self.profile_abundance_df_cutoff = self.profile_abundance_df_cutoff.loc[:, (self.profile_abundance_df_cutoff != 0).any(axis=0)]
            self._report_profiles_after_cutoff(num_profs_pre_cutoff)

            # now populate the new prof_uid_to_local_abund_dict_post_cutoff dictionary
            for i in list(self.profile_abundance_df_cutoff):  # for each column of the df
                temp_series = self.profile_abundance_df_cutoff[i]
                local_count = len(temp_series[temp_series > 0].index.values.tolist())
                self.prof_uid_to_local_abund_dict_post_cutoff[i] = local_count

            #dump
            pickle.dump(self.profile_abundance_df_cutoff, open(os.path.join(self.cache_dir, f'prof_df_cutoff_{cutoff_abund_str}.p'), 'wb'))
            pickle.dump(self.prof_uid_to_local_abund_dict_post_cutoff, open(os.path.join(self.cache_dir, f'prof_uid_to_local_abund_dict_post_cutoff_{cutoff_abund_str}.p'), 'wb'))

    def _create_profile_df_with_cutoff_high_low(self, cutoff_low, cutoff_high=None):
        """Creates a new df from the old df that has all of the values below the cutoff_abundance threshold
        made to 0. We will also calculate a new prof_uid_to_local_abund_dict_post_cutoff dictionary.
        """
        if cutoff_high == 0.05: # we are making the background associations set
            if os.path.exists(os.path.join(self.cache_dir, 'profile_df_cutoff_background.p')):
                self.profile_abundance_df_cutoff_background = pickle.load(open(os.path.join(self.cache_dir, 'profile_df_cutoff_background.p'), 'rb'))
                self.prof_uid_to_local_abund_dict_cutoff_background = pickle.load(open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict_cutoff_background.p'), 'rb'))
            else:
                self._create_prof_abund_high_low_from_scratch_and_pickle_out(cutoff_high, cutoff_low)
        elif cutoff_high == 0.40:# we are making the lower associations set
            if os.path.exists(os.path.join(self.cache_dir, 'profile_df_cutoff_low.p')):
                self.profile_abundance_df_cutoff_low = pickle.load(open(os.path.join(self.cache_dir, 'profile_df_cutoff_low.p'), 'rb'))
                self.prof_uid_to_local_abund_dict_cutoff_low = pickle.load(open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict_cutoff_low.p'), 'rb'))
            else:
                self._create_prof_abund_high_low_from_scratch_and_pickle_out(cutoff_high, cutoff_low)
        else: # we are making the higher associations set
            if os.path.exists(os.path.join(self.cache_dir, 'profile_df_cutoff_high.p')):
                self.profile_abundance_df_cutoff_high = pickle.load(open(os.path.join(self.cache_dir, 'profile_df_cutoff_high.p'), 'rb'))
                self.prof_uid_to_local_abund_dict_cutoff_high = pickle.load(open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict_cutoff_high.p'), 'rb'))
            else:
                self._create_prof_abund_high_low_from_scratch_and_pickle_out(cutoff_high, cutoff_low)

    def _create_prof_abund_high_low_from_scratch_and_pickle_out(self, cutoff_high, cutoff_low):
        num_profs_pre_cutoff = self._report_profiles_before_cutoff()
        # make new df from copy of old df
        profile_abundance_df_cutoff = self.profile_abundance_df.copy()
        profile_abundance_df_cutoff = self._mask_values_and_remove_0_cols(cutoff_high, cutoff_low, profile_abundance_df_cutoff)
        self._report_profiles_after_cutoff_high_low(num_profs_pre_cutoff, profile_abundance_df_cutoff)
        # now populate the new prof_uid_to_local_abund_dict_post_cutoff dictionary
        for i in list(profile_abundance_df_cutoff):  # for each column of the df
            temp_series = profile_abundance_df_cutoff[i]
            local_count = len(temp_series[temp_series > 0].index.values.tolist())
            if cutoff_high == 0.05:
                self.prof_uid_to_local_abund_dict_cutoff_background[i] = local_count
            elif cutoff_high == 0.40:
                self.prof_uid_to_local_abund_dict_cutoff_low[i] = local_count
            else:
                self.prof_uid_to_local_abund_dict_cutoff_high[i] = local_count
        # dump
        if cutoff_high == 0.05:
            self.profile_abundance_df_cutoff_background = profile_abundance_df_cutoff
            pickle.dump(self.profile_abundance_df_cutoff_background,
                        open(os.path.join(self.cache_dir, 'profile_df_cutoff_background.p'), 'wb'))
            pickle.dump(self.prof_uid_to_local_abund_dict_cutoff_background,
                        open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict_cutoff_background.p'), 'wb'))
        elif cutoff_high == 0.40:
            self.profile_abundance_df_cutoff_low = profile_abundance_df_cutoff
            pickle.dump(self.profile_abundance_df_cutoff_low,
                        open(os.path.join(self.cache_dir, 'profile_df_cutoff_low.p'), 'wb'))
            pickle.dump(self.prof_uid_to_local_abund_dict_cutoff_low,
                        open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict_cutoff_low.p'), 'wb'))
        else:
            self.profile_abundance_df_cutoff_high = profile_abundance_df_cutoff
            pickle.dump(self.profile_abundance_df_cutoff_high,
                        open(os.path.join(self.cache_dir, 'profile_df_cutoff_high.p'), 'wb'))
            pickle.dump(self.prof_uid_to_local_abund_dict_cutoff_high,
                        open(os.path.join(self.cache_dir, 'prof_uid_to_local_abund_dict_cutoff_high.p'), 'wb'))

    def _report_profiles_after_cutoff(self, num_profs_pre_cutoff):
        num_profs_post_cutoff = len(list(self.profile_abundance_df_cutoff))
        print(f'There are {num_profs_post_cutoff} after.')
        num_profs_removed = num_profs_pre_cutoff - num_profs_post_cutoff
        print(f'{num_profs_removed} ITS2 type profiles have been removed from the dataframe.')
        # get list of names of profiles removed due to cutoff
        profs_removed = [self.prof_uid_to_name_dict[uid] for uid in
                         list(self.profile_abundance_df) if
                         uid not in list(self.profile_abundance_df_cutoff)]
        print('These profiles were:')
        for prof in profs_removed:
            print(prof)
        # calculate how many unique DataSetSample to ITS2 type profile associations there are.
        num_associations_pre_cutoff = len(list(self.profile_abundance_df[self.profile_abundance_df > 0].stack().index))
        num_associations_post_cutoff = len(
            list(self.profile_abundance_df_cutoff[self.profile_abundance_df_cutoff > 0].stack().index))
        print(
            f'The number of unique DataSetSample to ITS2 type profile associations was {num_associations_pre_cutoff}.')
        print(f'The number of unique DataSetSample to ITS2 type profile associations '
              f'after cutoff is {num_associations_post_cutoff}')

    def _report_profiles_after_cutoff_high_low(self, num_profs_pre_cutoff, new_abund_df):
        num_profs_post_cutoff = len(list(new_abund_df))
        print(f'There are {num_profs_post_cutoff} after.')
        num_profs_removed = num_profs_pre_cutoff - num_profs_post_cutoff
        print(f'{num_profs_removed} ITS2 type profiles have been removed from the dataframe.')
        # get list of names of profiles removed due to cutoff
        profs_removed = [self.prof_uid_to_name_dict[uid] for uid in
                         list(self.profile_abundance_df) if
                         uid not in list(new_abund_df)]
        print('These profiles were:')
        for prof in profs_removed:
            print(prof)
        # calculate how many unique DataSetSample to ITS2 type profile associations there are.
        num_associations_pre_cutoff = len(
            list(self.profile_abundance_df[self.profile_abundance_df > 0].stack().index))
        num_associations_post_cutoff = len(
            list(new_abund_df[new_abund_df > 0].stack().index))
        print(
            f'The number of unique DataSetSample to ITS2 type profile associations was {num_associations_pre_cutoff}.')
        print(f'The number of unique DataSetSample to ITS2 type profile associations '
              f'after cutoff is {num_associations_post_cutoff}')

    def _report_profiles_before_cutoff(self):
        num_profs_pre_cutoff = len(list(self.profile_abundance_df))
        print(f'There are {num_profs_pre_cutoff} ITS2 type profiles before applying cutoff(s).')
        return num_profs_pre_cutoff

    def _mask_values_and_remove_0_cols(self, cutoff_high, cutoff_low, profile_abundance_df_cutoff):
        # change values below cutoff to 0
        if cutoff_low != 0:
            profile_abundance_df_cutoff = profile_abundance_df_cutoff.mask(
                cond=profile_abundance_df_cutoff <= cutoff_low, other=0)
        if cutoff_high:
            profile_abundance_df_cutoff = profile_abundance_df_cutoff.mask(
                cond=profile_abundance_df_cutoff > cutoff_high, other=0)
        # now drop columns with 0
        # now check to see if there are any type profiles that no longer have associations
        # https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
        return profile_abundance_df_cutoff.loc[:,
                                      (profile_abundance_df_cutoff != 0).any(axis=0)]

    def get_list_of_clade_col_type_uids_for_unifrac(self, high_low=None):
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
        # with open('cct_uids_005', 'w') as f:
        #    for uid in cct_id_list:
        #        f.write(f'{str(uid)}\n')

        # from the code here we can get a list that contains tuples of DataSetSample uids to AnalysisType uid for the
        # sample type pairings that we are interested in (i.e. the non-zeros in the cutoff df). We can then use these
        # ids to look up the CladeCollectionTypes we are interested in, get the uids of these, and pass these
        # into the distance functions of SymPortal that we are going to make.
        # we should make seperate outputs for bray vs unifrac, unifrac sqrt trans formed and not.

        if high_low == 'background':
            ### Then we are getting the tups specifically for the set of profiles that are below 0.05 in abundance
            index_column_tups = list(
                self.profile_abundance_df_cutoff_background[self.profile_abundance_df_cutoff_background > 0].stack().index)

            uid_pairs_for_ccts_path = os.path.join(self.outputs_dir, f'dss_at_tups_background')
        elif high_low == 'low':
            ### Then we are getting the tups specifically for the set of profiles that are within 0.05 and 0.40 abundance
            index_column_tups = list(
                self.profile_abundance_df_cutoff_low[self.profile_abundance_df_cutoff_low > 0].stack().index)

            uid_pairs_for_ccts_path = os.path.join(self.outputs_dir, f'dss_at_tups_low')

        else:
            # Then we are doing this for the older 0.05 or 0.40 cutoffs.
            # https://stackoverflow.com/questions/26854091/getting-index-column-pairs-for-true-elements-of-a-boolean-dataframe-in-pandas
            index_column_tups = list(self.profile_abundance_df_cutoff[self.profile_abundance_df_cutoff > 0].stack().index)
            if self.cutoff_abund == 0.05:
                uid_pairs_for_ccts_path = os.path.join(self.outputs_dir, f'dss_at_tups_005')
            else: #  0.40
                uid_pairs_for_ccts_path = os.path.join(self.outputs_dir, f'dss_at_tups_040')

        with open(uid_pairs_for_ccts_path, 'w') as f:
            for tup in index_column_tups:
                f.write(f'{tup[0]}\t{tup[1]}\n')
        print(f'A list of tuples has been output to {uid_pairs_for_ccts_path} that represent the UID of paired '
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
            prof_dist_df = self.between_profile_clade_dist_cct_specific_df_dict[clade]
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

            meta_info_df_for_clade = self.experimental_metadata_info_df.loc[meta_info_indices, :]

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

    def permute_sample_permanova(self):
        meta_df = self.experimental_metadata_info_df
        for clade in self.clades:

            clade_sample_dist_df = self.between_sample_clade_dist_df_dict[clade]
            if self.seq_distance_method == 'braycurtis':

                if self.remove_se and clade == 'D' and not self.maj_only:
                    clade_sample_dist_df = self._remove_se_from_df(clade_sample_dist_df)
                    output_path_dist_matrix = os.path.join(self.outputs_dir, f'dists_permanova_samples_{clade}_{self.seq_distance_method}_no_se.csv')
                elif self.maj_only:
                    clade_sample_dist_df = self._remove_non_maj_from_df(clade_sample_dist_df=clade_sample_dist_df, clade=clade)
                    output_path_dist_matrix = os.path.join(self.outputs_dir,
                                                           f'dists_permanova_samples_{clade}_{self.seq_distance_method}_only_maj.csv')
                else:
                    output_path_dist_matrix = os.path.join(self.outputs_dir, f'dists_permanova_samples_{clade}_{self.seq_distance_method}.csv')


            else: # dist_method == 'unifrac'
                if self.remove_se and clade == 'D' and not self.maj_only:
                    clade_sample_dist_df = self._remove_se_from_df(clade_sample_dist_df)
                    output_path_dist_matrix = os.path.join(self.outputs_dir, f'dists_permanova_samples_{clade}_{self.seq_distance_method}_no_se.csv')
                elif self.maj_only:
                    clade_sample_dist_df = self._remove_non_maj_from_df(clade_sample_dist_df=clade_sample_dist_df, clade=clade)
                    output_path_dist_matrix = os.path.join(self.outputs_dir,
                                                           f'dists_permanova_samples_{clade}_{self.seq_distance_method}_only_maj.csv')
                else:
                    output_path_dist_matrix = os.path.join(self.outputs_dir, f'dists_permanova_samples_{clade}_{self.seq_distance_method}.csv')

            clade_sample_dist_df.to_csv(path_or_buf=output_path_dist_matrix, sep=',', header=False, index=False, line_terminator='\n')

            meta_info_df_for_clade = meta_df.loc[clade_sample_dist_df.index.values.tolist(), :]

            if self.remove_se and clade == 'D' and not self.maj_only:
                output_path_meta_info = os.path.join(self.outputs_dir, f'sample_meta_info_{clade}_{self.seq_distance_method}_no_se.csv')
            elif self.maj_only:
                output_path_meta_info = os.path.join(self.outputs_dir, f'sample_meta_info_{clade}_{self.seq_distance_method}_only_maj.csv')
            else:
                output_path_meta_info = os.path.join(self.outputs_dir, f'sample_meta_info_{clade}_{self.seq_distance_method}.csv')


            # # If you want to append shuffled versions of the meta then you can use the code below
            # output_path_meta_info = os.path.join(self.outputs_dir, f'sample_meta_info_{clade}_shuffled.csv')
            # # append on some columns that are shuffled versions of the info
            # for col_name in list(meta_info_df_for_clade):
            #     meta_info_df_for_clade[f'{col_name}_shuffled'] = meta_info_df_for_clade[f'{col_name}'].sample(frac=1, random_state=2).values.tolist()
            # # now I want to append a column for that doubles the number of species and see what it does to the variance apportioned
            # binary = 1
            # double_species_list = []
            # for ind_val in meta_info_df_for_clade.index.values.tolist():
            #     current_species = meta_info_df_for_clade.loc[ind_val, 'species']
            #     if binary % 2 == 1:
            #         double_species_list.append(f'{current_species}_1')
            #     else:
            #         double_species_list.append(current_species)
            #     binary += 1
            # meta_info_df_for_clade['species_double'] = double_species_list


            meta_info_df_for_clade.to_csv(
                path_or_buf=output_path_meta_info, sep=',', header=True, index=False, line_terminator='\n')

            # # It looks as though we can compute permdisp directly in python.
            # # Although the permanova is still sadly lacking
            # dist_obj = skbio.stats.distance.DistanceMatrix(clade_sample_dist_df, ids=clade_sample_dist_df.index.values.tolist())
            # condensed_dist = scipy.spatial.distance.squareform(clade_sample_dist_df)
            # this = skbio.stats.distance.permdisp(distance_matrix=dist_obj, grouping=meta_info_df_for_clade['season'])
            # apples = 'asdf'

    def _remove_non_maj_from_df(self, clade_sample_dist_df, clade):
        # remove the samples uids from the dist matrix that are S. hystrix species
        maj_clade_ser = self.clade_proportion_df_non_normalised.idxmax(axis=1)
        clade_maj_only_uids = maj_clade_ser.index[maj_clade_ser != clade].tolist()
        # now remove the uids both from the cols and the rows
        ind_list_of_df = clade_sample_dist_df.index.values.tolist()
        # this = set(clade_maj_only_uids) & set(ind_list_of_df)
        # some of the UIDs will not be in the df as they did not have any of this clade.
        dropped_df = clade_sample_dist_df.drop(index=clade_maj_only_uids, columns=clade_maj_only_uids, errors='ignore')
        return dropped_df

    def _remove_se_from_df(self, clade_sample_dist_df):
        # remove the samples uids from the dist matrix that are of the species S. hystrix
        uids_to_remove = []
        for uid in clade_sample_dist_df.index:
            if self.experimental_metadata_info_df.at[uid, 'species'] == "SE":
                uids_to_remove.append(uid)
        # now remove the uids both from the cols and the rows
        dropped_df = clade_sample_dist_df.drop(index=uids_to_remove, columns=uids_to_remove)
        return dropped_df

    def output_seq_analysis_overview_outputs(self):
        self._report_sequencing_overview()

        self._report_predicted_profiles_overview()

        self._report_on_eveness_of_profile_distributions_per_clade()

        # TODO set up a very quick bar plot of this data
        self._report_on_profile_abundances_per_sample()

        self._report_on_within_sample_clade_profile_rank_and_abundance()

        apples = 'asdf'

    def _report_on_within_sample_clade_profile_rank_and_abundance(self):
        # now interesting to find out what the average abundance of the profiles from the different clades were
        # best to do this through averaging from the output df for the profiles.
        # again we can do this on a type by type basis so that we can get an idea of spread within each of the clades
        # TODO set up a bar plot of this data too.
        # first we do this using the df that doesn't have the <0.05 abundance profile instances removed
        print('\n\nusing the non-cutoff profile df')
        self._claclulate_av_abundance_of_clade_profiles(df_to_calculate_from=self.profile_abundance_df)
        # now we do it using the df that does have the 0.05 abundance profile instances removed
        print('\n\nusing the cutoff profile df high')
        self._claclulate_av_abundance_of_clade_profiles(df_to_calculate_from=self.profile_abundance_df_cutoff_high)
        print('\n\nusing the cutoff profile df low')
        self._claclulate_av_abundance_of_clade_profiles(df_to_calculate_from=self.profile_abundance_df_cutoff_low)
        print('\n\nusing the cutoff profile df 0.05')
        self._claclulate_av_abundance_of_clade_profiles(df_to_calculate_from=self.profile_abundance_df_cutoff)
        print('\n\nusing the non-cutoff profile df')
        self._calc_av_rank_of_clade_profile(df_to_calculate_from=self.profile_abundance_df)
        print('\n\nusing the high cutoff profile df')
        self._calc_av_rank_of_clade_profile(df_to_calculate_from=self.profile_abundance_df_cutoff_high)
        print('\n\nusing the low cutoff profile df')
        self._calc_av_rank_of_clade_profile(df_to_calculate_from=self.profile_abundance_df_cutoff_low)
        print('\n\nusing the 0.05 cutoff profile df')
        self._calc_av_rank_of_clade_profile(df_to_calculate_from=self.profile_abundance_df_cutoff)
        foo = 'bar'
    def _calc_av_rank_of_clade_profile(self, df_to_calculate_from):
        # we can also ask the question of what the average rank of the type was. For example, on average, Cladocopium its2 type profile was the 1.4th most abundant profile
        # we could also plot this as we can have an average value for each profile and average these in turn with a n value. the stdv will then
        print('\n\n')
        dd_dict_profile_av_rankings = defaultdict(list)
        for profile_uid in list(df_to_calculate_from):  # for every column/profile
            # ignore the one profile that was only found in the problematic sample we had to remove
            # profile uid 2989
            if profile_uid == 2989:
                continue
            temp_rank_list = []
            indexers_non_zero_int = list(df_to_calculate_from[profile_uid].to_numpy().nonzero()[0])
            if not indexers_non_zero_int:
                continue
            # for each sample that the profile was found in
            # work out what rank the type was
            for i in indexers_non_zero_int:
                # get the non_zero values
                indexers_non_zero = list(df_to_calculate_from.iloc[i,].to_numpy().nonzero()[0])
                non_zero_series = df_to_calculate_from.iloc[i, indexers_non_zero]
                non_zero_series_ordered = non_zero_series.sort_values(ascending=False)
                sorted_vals = non_zero_series_ordered.values.tolist()
                # now see what rank the value in question is
                value_of_profile_in_question = df_to_calculate_from.iloc[i][profile_uid]
                for i, val in enumerate(sorted_vals):
                    if val == value_of_profile_in_question:
                        temp_rank_list.append(i + 1)
                        break
            try:
                dd_dict_profile_av_rankings[self.profile_meta_info_df.loc[profile_uid]['Clade']].append(
                    sum(temp_rank_list) / len(temp_rank_list))
            except:
                foo = 'asdf'
        for clade in list('ACD'):
            av_rank = sum(dd_dict_profile_av_rankings[clade]) / len(dd_dict_profile_av_rankings[clade])
            std = statistics.pstdev(dd_dict_profile_av_rankings[clade])
            print(
                f'The average rank of a clade {clade} profile was {av_rank} with stdv of {std} from {len(dd_dict_profile_av_rankings[clade])} profiles.')

    def _claclulate_av_abundance_of_clade_profiles(self, df_to_calculate_from):

        dd_dict_profile_av_rel_abunds = defaultdict(list)
        for profile_uid in list(df_to_calculate_from):  # for every column/profile
            indexers_non_zero = list(df_to_calculate_from[profile_uid].to_numpy().nonzero()[0])
            non_zero_series = df_to_calculate_from[profile_uid].iloc[indexers_non_zero]
            dd_dict_profile_av_rel_abunds[self.profile_meta_info_df.loc[profile_uid]['Clade']].append(
                non_zero_series.mean())
        for clade in list('ACD'):
            av_abund = sum(dd_dict_profile_av_rel_abunds[clade]) / len(dd_dict_profile_av_rel_abunds[clade])
            std = statistics.pstdev(dd_dict_profile_av_rel_abunds[clade])
            print(
                f'The average abund of a clade {clade} profile was {av_abund} with stdv of {std} from {len(dd_dict_profile_av_rel_abunds[clade])} profiles.')

    def _report_on_profile_abundances_per_sample(self):
        # number of types harbored on average by sample
        # then the average abundances of the 1st, 2nd 3rd etc most abundant ITS2 type profiles within each sample
        number_of_profiles = []
        for i in range(len(self.profile_abundance_df.index.values.tolist())):  # for each row of the df
            indexers_non_zero = list(self.profile_abundance_df.iloc[i,].nonzero()[0])
            non_zero_series = self.profile_abundance_df.iloc[i, indexers_non_zero]
            non_zero_series_ordered = non_zero_series.sort_values(ascending=False)
            number_of_profiles.append(non_zero_series_ordered.values.tolist())
        # calculate the number of samples containing each number of profiles
        print('\n\n')
        dd_num_profiles = defaultdict(int)
        for l in number_of_profiles:
            dd_num_profiles[len(l)] += 1
        for i in range(10):
            if i in dd_num_profiles:
                print(f'{dd_num_profiles[i]} samples contained {i} profiles')
        av_number_of_profiles = sum([len(l) for l in number_of_profiles]) / len(number_of_profiles)
        # dictionary that will hold the values of the abundances
        dd_dict_profile_abunds = defaultdict(list)
        for l in number_of_profiles:
            for i, val in enumerate(l):
                dd_dict_profile_abunds[i].append(val)
        for i in range(10):
            if i in dd_dict_profile_abunds:
                av_abund = sum(dd_dict_profile_abunds[i]) / len(dd_dict_profile_abunds[i])
                std = statistics.pstdev(dd_dict_profile_abunds[i])
                print(
                    f'Profile {i + 1} has an average abundance of {av_abund} and a stdv of {std} calculated from {len(dd_dict_profile_abunds[i])} samples.')

    def _report_on_eveness_of_profile_distributions_per_clade(self):
        for clade in self.clades:
            self._calc_cummulative_abund_of_top_half_abund_profiles(clade=clade)

    def _report_predicted_profiles_overview(self):
        total_unique_number_of_type_profiles = len(self.profile_meta_info_df.index.tolist())
        total_absolute_number_of_type_profile_instances_local = sum(self.profile_meta_info_df['ITS2 type abundance local'].astype('float'))
        total_absolute_number_of_type_profile_instances_global = sum(self.profile_meta_info_df['ITS2 type abundance DB'])
        print(f'Across all clades {total_absolute_number_of_type_profile_instances_local} ITS2 type profile instances were '
              f'predicted in this study representing {total_unique_number_of_type_profiles} different profiles.')

        # The cutoff dfs are self.profile_abundance_df_cutoff_high and self.profile_abundance_df_cutoff_low
        number_of_types_from_df_pre_cutoff = len(self.profile_abundance_df.columns.values.tolist())  # 111
        number_of_types_from_df_post_cutoff_high = len(self.profile_abundance_df_cutoff_high.columns.values.tolist())
        number_of_types_from_df_post_cutoff_low = len(self.profile_abundance_df_cutoff_low.columns.values.tolist())

        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
        number_of_instances_after_cutoff_high = [item for sublist in (self.profile_abundance_df_cutoff_high > 0).values.tolist() for item in sublist].count(True)
        number_of_instances_after_cutoff_low = [item for sublist in
                                                 (self.profile_abundance_df_cutoff_low > 0).values.tolist() for item in
                                                 sublist].count(True)
        print(f'Number of different profiles pre 0.05 cutoff: {number_of_types_from_df_pre_cutoff}')
        print(f'Number of different profiles in high cutoff: {number_of_types_from_df_post_cutoff_high}')
        print(f'Number of different profiles in low cutoff: {number_of_types_from_df_post_cutoff_low}')

        print(f'Number of profile instances in high cutoff: {number_of_instances_after_cutoff_high}')
        print(f'Number of profile instances in low cutoff: {number_of_instances_after_cutoff_low}')

    def _report_sequencing_overview(self):
        # total number of sequences pre-QC
        sum_of_contigs = sum(self.seq_meta_data_df['raw_contigs'])
        # total number of successfully sequenced samples
        num_samples = len(self.seq_meta_data_df.index.values.tolist())
        # absolute and unique number of sequences after the SymPortal quality control but before MED
        average_num_symbiodinium_seqs_absolute_post_qc_before_med = int(
            sum(self.seq_meta_data_df['post_taxa_id_absolute_symbiodinium_seqs']) / num_samples)
        average_num_symbiodinium_seqs_uniue_post_qc_before_med = int(
            sum(self.seq_meta_data_df['post_taxa_id_unique_symbiodinium_seqs']) / num_samples)
        # absolute and unique number of sequences after the SymPortal quality control and MED
        average_num_symbiodinium_seqs_absolute_post_med = int(
            sum(self.seq_meta_data_df['post_med_absolute']) / num_samples)
        average_num_symbiodinium_seqs_unique_post_med = int(sum(self.seq_meta_data_df['post_med_unique']) / num_samples)
        print(f'In total, {sum_of_contigs} contigs were produced from {num_samples} samples.')
        print(f'This translates to an average read depth of {sum_of_contigs/num_samples}.')
        print(
            f'Before undergoing minimum entropy decomposition, an average of {average_num_symbiodinium_seqs_absolute_post_qc_before_med} Symbiodiniaceae sequences were returned per sample representing, on average, {average_num_symbiodinium_seqs_uniue_post_qc_before_med} distinct sequences per sample.')
        print(
            f'After MED, these values were {average_num_symbiodinium_seqs_absolute_post_med} and {average_num_symbiodinium_seqs_unique_post_med}, respectively.')

    def _calc_cummulative_abund_of_top_half_abund_profiles(self, clade):
        df_clade_specific = self.profile_meta_info_df[self.profile_meta_info_df['Clade'] == clade]
        sorted_df = df_clade_specific.sort_values(by='ITS2 type abundance local', axis=0, ascending=False)
        num_profiles = len(df_clade_specific.index.values.tolist())
        print(f'{num_profiles} clade {clade} profiles were returned.')
        sum_of_first_half_most_abundant_profiles = sum(
            sorted_df.iloc[:int(num_profiles / 2), ]['ITS2 type abundance local'])
        total_num_profile_instances_for_clade = sum(sorted_df['ITS2 type abundance local'])
        percent_rep_by_most_abund_profiles = sum_of_first_half_most_abundant_profiles / total_num_profile_instances_for_clade
        print(f'The 50% most abundant profiles represented {percent_rep_by_most_abund_profiles} of the {total_num_profile_instances_for_clade} total profile-sample occurences for this genus.')

    def assess_balance_and_dispersions_of_distance_matrix(self):
        """This method will be used to investigate the pairwise comparisons that have shown significant
        PERMDISP test results. It will look at how balanced the factors are and what the dispersion looks like"""

        # self._assess_balance_and_variance_clade_prop()

        # we will strip the between sample matrices of non-maj samples if self.maj_only is true before assessing
        # variances as below
        if self.maj_only:
            for clade in self.clades:
                self.between_sample_clade_dist_df_dict[clade] = self._remove_non_maj_from_df(clade_sample_dist_df=self.between_sample_clade_dist_df_dict[clade], clade=clade)
        # First for each clade and for each factor check to see what the ratios for numbers are
        for clade in self.clades:
            print(f'\n\nExamining balance for {clade}')
            for factor in list(self.experimental_metadata_info_df):
                print(f'\nExaminging balanace for factor {factor}')
                factor_counter_dd, max_val = self._populate_factor_counter_and_get_max_val(clade, factor)
                output_str = ''
                for dd_k, dd_v in factor_counter_dd.items():
                    output_str += f'{dd_k}: {dd_v/max_val}; '
                print(f'{factor} balance ratios are: {output_str}')
        foo = 'asdf'


        # now we look at the variance ratios between the pairwise group comparisons
        # for every clade and factor combo
        for clade in self.clades:
            print(f'\n\nExamining variance ratios for {clade}')
            for factor in list(self.experimental_metadata_info_df):

                print(f'\nExaminging variance ratios for factor {factor}')
                factor_counter_dd, max_val = self._populate_factor_counter_and_get_max_val(clade, factor)

                variance_per_group_of_factor_dict = self._make_group_variance_dict(clade, factor, factor_counter_dd)

                pw_var_ratio_dict = self._make_pw_variance_ratio_dict(factor_counter_dd,
                                                                      variance_per_group_of_factor_dict)

                self._report_bad_values_and_create_pw_df(clade, factor, factor_counter_dd, pw_var_ratio_dict,
                                                         variance_per_group_of_factor_dict)

                foo = 'bar'
            foo = 'bar'

    def _assess_balance_and_variance_clade_prop(self):
        # now assess the balance of the clade proportions
        print('\n\nNow assessing balance of clade proportion matrix')
        for factor in list(self.experimental_metadata_info_df):
            print(f'\nExaminging balanace for factor {factor}')
            factor_counter_dd, max_val = self._populate_factor_counter_and_get_max_val_clade_prop(factor)
            output_str = ''
            for dd_k, dd_v in factor_counter_dd.items():
                output_str += f'{dd_k}: {dd_v / max_val}; '
            print(f'{factor} balance ratios are: {output_str}')
        # now look at variance ratios for the clade proportion
        for factor in list(self.experimental_metadata_info_df):
            print(f'\nExaminging variance ratios for factor {factor}')
            factor_counter_dd, max_val = self._populate_factor_counter_and_get_max_val_clade_prop(factor)

            variance_per_group_of_factor_dict = self._make_group_variance_dict_clade_prop(factor, factor_counter_dd)

            pw_var_ratio_dict = self._make_pw_variance_ratio_dict(factor_counter_dd,
                                                                  variance_per_group_of_factor_dict)

            self._report_bad_values_and_create_pw_df_clade_prop(factor, factor_counter_dd, pw_var_ratio_dict,
                                                                variance_per_group_of_factor_dict)

    def _report_bad_values_and_create_pw_df(self, clade, factor, factor_counter_dd, pw_var_ratio_dict,
                                            variance_per_group_of_factor_dict):
        # now we can create a dataframe of the values
        fixed_list_groups = list(variance_per_group_of_factor_dict.keys())
        df = pd.DataFrame(columns=fixed_list_groups, index=fixed_list_groups)
        for outer_group in fixed_list_groups:
            for inner_group in fixed_list_groups:
                if outer_group == inner_group:
                    df.at[outer_group, inner_group] = 1
                else:
                    var_ratio = pw_var_ratio_dict[frozenset({outer_group, inner_group})]
                    abund_ratio = factor_counter_dd[outer_group] / factor_counter_dd[inner_group]
                    bad_abund = False
                    if abund_ratio > 2 or abund_ratio < 0.5:
                        bad_abund = True
                    df.at[outer_group, inner_group] = var_ratio
                    if var_ratio > 1 and bad_abund:
                        print(
                            f'Clade {clade}, factor {factor}, combo {outer_group}_{inner_group} is bad. Var ratio was {var_ratio}. Abund ratio was {abund_ratio}.')

    def _report_bad_values_and_create_pw_df_clade_prop(self, factor, factor_counter_dd, pw_var_ratio_dict,
                                            variance_per_group_of_factor_dict):
        # now we can create a dataframe of the values
        fixed_list_groups = list(variance_per_group_of_factor_dict.keys())
        df = pd.DataFrame(columns=fixed_list_groups, index=fixed_list_groups)
        for outer_group in fixed_list_groups:
            for inner_group in fixed_list_groups:
                if outer_group == inner_group:
                    df.at[outer_group, inner_group] = 1
                else:
                    var_ratio = pw_var_ratio_dict[frozenset({outer_group, inner_group})]
                    abund_ratio = factor_counter_dd[outer_group] / factor_counter_dd[inner_group]
                    bad_abund = False
                    if abund_ratio > 2 or abund_ratio < 0.5:
                        bad_abund = True
                    df.at[outer_group, inner_group] = var_ratio
                    if var_ratio > 1 and bad_abund:
                        print(
                            f'Factor {factor}, combo {outer_group}_{inner_group} is bad. Var ratio was {var_ratio}. Abund ratio was {abund_ratio}.')

    def _make_pw_variance_ratio_dict(self, factor_counter_dd, variance_per_group_of_factor_dict):
        # now that we have the interpoint distances for each of the groups for the factor
        # we can now do the pairwise comparisons between the groups of the factors to workout variance ratios
        # For each of the ratio calculations we will always make sure to but the less abundant
        # group as the numerator and the more abundant as the denominator.
        # This way we can look for positive values of the variance ratios as a sign that something is not good
        # I.e. this will show us that despite being less abundant a group has a bigger dispersion than a more
        # abundant group.
        pw_var_ratio_dict = {}
        for group_1, group_2 in itertools.combinations(variance_per_group_of_factor_dict, 2):
            if factor_counter_dd[group_1] > factor_counter_dd[group_2]:
                pw_var_ratio_dict[frozenset({group_1, group_2})] = variance_per_group_of_factor_dict[group_2] / \
                                                                   variance_per_group_of_factor_dict[group_1]
            else:
                pw_var_ratio_dict[frozenset({group_1, group_2})] = variance_per_group_of_factor_dict[group_1] / \
                                                                   variance_per_group_of_factor_dict[group_2]
        return pw_var_ratio_dict

    def _make_group_variance_dict(self, clade, factor, factor_counter_dd):
        # for each group that is in factor get a list of the distances and hold it in dd
        # this dict will have a key for every group in the factor that is present in the clade distance matrix
        # for each key there will be a value that is a list of the interpoint distances for only the samples
        # that are of the given group. We will calculate the variance of these points
        inter_point_dist_per_group_of_factor_dd = defaultdict(list)
        # This dictionary will hold the results of the variance calculation from the dd above.
        # As such it will be group as key and variance as value
        variance_per_group_of_factor_dict = dict()
        for group in factor_counter_dd:
            # for each sample uid in the btwn sample dist matrix
            # if the sample is of the group
            # then add this uid to list
            smp_uids_of_group = []
            for smp_uid in self.between_sample_clade_dist_df_dict[clade].index:
                if self.experimental_metadata_info_df.at[smp_uid, factor] == group:
                    smp_uids_of_group.append(smp_uid)
            # now go pairwise for each uid in the list and populate
            for uid_1, uid_2 in itertools.combinations(smp_uids_of_group, 2):
                inter_point_dist_per_group_of_factor_dd[group].append(
                    self.between_sample_clade_dist_df_dict[clade].at[uid_1, uid_2])

            # Now to the variance calculation
            try:
                variance_per_group_of_factor_dict[group] = variance(inter_point_dist_per_group_of_factor_dd[group])
            except statistics.StatisticsError:
                variance_per_group_of_factor_dict[group] = 0
        return variance_per_group_of_factor_dict

    def _make_group_variance_dict_clade_prop(self, factor, factor_counter_dd):
        # for each group that is in factor get a list of the distances and hold it in dd
        # this dict will have a key for every group in the factor that is present in the clade distance matrix
        # for each key there will be a value that is a list of the interpoint distances for only the samples
        # that are of the given group. We will calculate the variance of these points
        inter_point_dist_per_group_of_factor_dd = defaultdict(list)
        # This dictionary will hold the results of the variance calculation from the dd above.
        # As such it will be group as key and variance as value
        variance_per_group_of_factor_dict = dict()
        for group in factor_counter_dd:
            # for each sample uid in the btwn sample dist matrix
            # if the sample is of the group
            # then add this uid to list
            smp_uids_of_group = []
            for smp_uid in self.between_sample_clade_proportion_distances_df.index:
                if self.experimental_metadata_info_df.at[smp_uid, factor] == group:
                    smp_uids_of_group.append(smp_uid)
            # now go pairwise for each uid in the list and populate
            for uid_1, uid_2 in itertools.combinations(smp_uids_of_group, 2):
                inter_point_dist_per_group_of_factor_dd[group].append(
                    self.between_sample_clade_proportion_distances_df.at[uid_1, uid_2])

            # Now to the variance calculation
            try:
                variance_per_group_of_factor_dict[group] = variance(inter_point_dist_per_group_of_factor_dd[group])
            except statistics.StatisticsError:
                variance_per_group_of_factor_dict[group] = 0
        return variance_per_group_of_factor_dict

    def _populate_factor_counter_and_get_max_val(self, clade, factor):
        factor_counter_dd = defaultdict(int)
        for smp_ind in self.between_sample_clade_dist_df_dict[clade].index:
            factor_counter_dd[self.experimental_metadata_info_df.at[smp_ind, factor]] += 1
        # here we have the factor counter populated
        max_val = max(factor_counter_dd.values())
        return factor_counter_dd, max_val

    def _populate_factor_counter_and_get_max_val_clade_prop(self, factor):
        factor_counter_dd = defaultdict(int)
        for smp_ind in self.between_sample_clade_proportion_distances_df.index:
            factor_counter_dd[self.experimental_metadata_info_df.at[smp_ind, factor]] += 1
        # here we have the factor counter populated
        max_val = max(factor_counter_dd.values())
        return factor_counter_dd, max_val


class MetaInfoPlotter:
    def __init__(self, parent_analysis, ordered_uid_list, meta_axarr, prof_uid_to_smpl_uid_list_dict, prof_uid_to_y_loc_dict, dend_ax, sub_cat_axarr, clade_index):
        self.parent = parent_analysis
        self.ordered_prof_uid_list = ordered_uid_list
        self.clade_index = clade_index
        # these are the axes that will display the actual data
        self.meta_axarr = meta_axarr
        # these are the axes that will hold the subcategory labels
        self.sub_cat_axarr = sub_cat_axarr
        # set the x axis lims to match the dend_ax
        for ax, cat_ax, label, labpad in zip(self.meta_axarr, self.sub_cat_axarr, ['Species', 'Reef','Reef\nType', 'Depth', 'Season'], [0,10,0,10,0]):
            ax.set_ylim(dend_ax.get_ylim())
            # ax.spines['top'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim((0, 1))
            # if clade_index ==0:
            #     cat_ax.set_ylabel(label, rotation='vertical', fontweight='bold', fontsize='x-small',
            #                        verticalalignment='center', labelpad=labpad)


        self.prof_uid_to_smpl_uid_list_dict = prof_uid_to_smpl_uid_list_dict
        self.prof_uid_to_y_loc_dict = prof_uid_to_y_loc_dict
        self.smpl_meta_df = self.parent.experimental_metadata_info_df
        # the space left between the info boxes of the plot
        # this should be set dynmaically at some point rather than hard coded
        self.meta_box_buffer = 1
        self.meta_box_height = 0.8

        # species
        self.species_plotter = None
        self.depth_plotter = None
        self.reef_type_plotter = None
        self.season = None
        self.reef_plotter = None

    def plot_species_meta(self):
        # Plot species, season, depth, reef type
        color_dict = {
            'G': '#98FB98', 'GX': '#F0E68C', 'M': '#DDA0DD', 'P': '#8B008B',
            'PC': '#00BFFF', 'SE': '#0000CD', 'ST': '#D2691E'}
        category_list = ['M', 'G', 'GX',  'P', 'PC', 'SE', 'ST']
        category_labels = ['M. dichotoma', 'G. planulata', 'G. fascicularis', 'Porites spp.', 'P. verrucosa', 'S. hystrix', 'S. pistillata']
        self.species_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[0], cat_ax=self.sub_cat_axarr[0], color_dict=self.parent.old_color_dict,
                                               category_list=category_list, category_df_header='species', category_labels=category_labels)
        self.species_plotter.plot()

    def plot_reef_meta(self):
        # Plot species, season, depth, reef type
        category_list = ['Fsar', 'Tahla', 'Qita al Kirsh', 'Al Fahal', 'Shib Nazar', 'Abu Madafi']


        category_labels = ['Fsar', 'Tahla', 'Qita al Kirsh', 'Al Fahal', 'Shib Nazar', 'Abu Madafi']
        self.reef_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[1], cat_ax=self.sub_cat_axarr[1], color_dict=self.parent.old_color_dict,
                                               category_list=category_list, category_df_header='reef', category_labels=category_labels)
        self.reef_plotter.plot()

    def plot_depth_meta(self):
        color_dict = {
            1:'#CAE1FF', 15: '#2E37FE', 30: '#000080'}
        category_list = [1, 15, 30]
        category_labels = ['1 m', '15 m', '30 m']
        self.depth_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[3], cat_ax=self.sub_cat_axarr[3],color_dict=self.parent.old_color_dict,
                                               category_list=category_list, category_df_header='depth', category_labels=category_labels)
        self.depth_plotter.plot()

    def plot_reef_type(self):
        color_dict = {
            'Inshore': '#FF0000', 'Midshelf': '#FFFF00', 'Offshore': '#008000'}
        category_list = ['Inshore', 'Midshelf', 'Offshore']
        category_labels = ['Inshore', 'Midshelf', 'Offshore']
        self.depth_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[2], cat_ax=self.sub_cat_axarr[2],color_dict=self.parent.old_color_dict,
                                             category_list=category_list, category_df_header='reef_type', category_labels=category_labels)
        self.depth_plotter.plot()

    def plot_season(self):
        color_dict = {
            'Summer': '#FF0000', 'Winter': '#00BFFF'}
        category_list = ['Summer', 'Winter']
        category_labels = ['Summer', 'Winter']
        self.depth_plotter = self.CatPlotter(parent_meta_plotter=self, ax=self.meta_axarr[4], cat_ax=self.sub_cat_axarr[4],color_dict=self.parent.old_color_dict,
                                             category_list=category_list, category_df_header='season', category_labels=category_labels)
        self.depth_plotter.plot()

    class CatPlotter:
        def __init__(self, parent_meta_plotter, ax, cat_ax, color_dict, category_list, category_df_header, category_labels):
            self.parent_meta_plotter = parent_meta_plotter
            self.prof_uid_list = self.parent_meta_plotter.ordered_prof_uid_list
            self.prof_uid_to_smpl_uid_list_dict = self.parent_meta_plotter.prof_uid_to_smpl_uid_list_dict
            self.prof_y_loc_dict = self.parent_meta_plotter.prof_uid_to_y_loc_dict
            self.meta_df = self.parent_meta_plotter.smpl_meta_df
            self.ax = ax
            self.cat_ax = cat_ax
            y_loc_one = self.prof_y_loc_dict[self.prof_uid_list[0]]
            y_loc_two = self.prof_y_loc_dict[self.prof_uid_list[1]]
            self.dist_betwee_y_locs = y_loc_two - y_loc_one
            # the space left between the info boxes of the plot
            # this should be set dynmaically at some point rather than hard coded
            self.meta_box_buffer = self.parent_meta_plotter.meta_box_buffer
            self.meta_box_height = self.parent_meta_plotter.meta_box_height
            self.color_dict = color_dict
            self.category_list = category_list
            self.category_df_header = category_df_header
            self.cat_labels = category_labels

        def plot(self):
            x0_list, width_list = self._plot_data_ax()

            if self.parent_meta_plotter.clade_index == 0:
                # only have to make the sub category plot once.
                self._make_sub_category_plot(width_list, x0_list)

        def _make_sub_category_plot(self, width_list, x0_list):
            # now populate the category axis with the sub category labels
            # y values will be the y0list + half the height
            bar_width = width_list[0]
            for cat_lab, x0_val in zip(self.cat_labels, x0_list):
                if self.category_df_header == 'species':  # italics
                    self.cat_ax.annotate(s=cat_lab, xy=(x0_val + (0.5 * bar_width), 0), horizontalalignment='center',
                                         verticalalignment='bottom', fontsize='xx-small', fontstyle='italic', rotation='vertical')
                else:
                    self.cat_ax.annotate(s=cat_lab, xy=(x0_val + (0.5 * bar_width), 0), horizontalalignment='center',
                                         verticalalignment='bottom', fontsize='xx-small', rotation='vertical')

        def _plot_data_ax(self):
            """We will plot a horizontal bar plot using rectangle patches"""
            for prof_uid in self.prof_uid_list:
                list_of_sample_uids = self.prof_uid_to_smpl_uid_list_dict[prof_uid]
                list_of_cat_instances = [self.meta_df.at[smpl_uid, self.category_df_header] for smpl_uid in
                                         list_of_sample_uids]
                # calculate eveness
                counter = Counter(list_of_cat_instances)

                # Then this only contains the one species and it should simply be the species color
                y0_list, x0_list, height_list, width_list,  = self._get_rect_attributes(prof_uid, counter)

                for x, y, w, h, s in zip(x0_list, y0_list, width_list, height_list, self.category_list):
                    if h > 0:
                        rect_p = patches.Rectangle(
                            xy=(x, y), width=w, height=h, facecolor=self.color_dict[s], edgecolor='none')
                        self.ax.add_patch(rect_p)
            return x0_list, width_list

        def _get_rect_attributes(self, prof_uid, counter):

            num_categories = len(self.category_list)

            bar_width = (1/(num_categories))
            x0_list = [i * bar_width for i in range(num_categories)]
            width_list = [bar_width for _ in range(num_categories)]

            y_loc_of_prof = self.prof_y_loc_dict[prof_uid]
            data_y0 = (y_loc_of_prof - (self.dist_betwee_y_locs / 2)) + self.meta_box_buffer
            data_y1 = (y_loc_of_prof + (self.dist_betwee_y_locs / 2)) - self.meta_box_buffer
            rect_height = data_y1 - data_y0

            heights_list = []
            num_samples = sum(counter.values())
            for cat in self.category_list:
                if cat in counter:
                    heights_list.append((counter[cat]/num_samples)*rect_height)
                else:
                    heights_list.append(0)
            y0_list = [data_y0 for _ in range(num_categories)]
            return y0_list, x0_list, heights_list, width_list



if __name__ == "__main__":
    rest_analysis = RestrepoAnalysis(cutoff_abund=0.05, remove_se=False, maj_only=False, remove_se_clade_props=True, seq_distance_method='unifrac')
    # rest_analysis.populate_data_sheet()
    # When we ran the analysis of variance ratios within the context of the between sample distance permanova analysis
    # we saw that one of the problematic groups was SE (S. hystrix) in the clade D matrix.
    # We are therefore going to allow an option to remove the samples that are this species from the clade C matrix

    # run this to generate the dss and at id tuples that we can use in the SymPortal shell to get the specific
    # clade collection types that we can then generate distances from to make the dendrogram figure
    # NB I have saved the cct uid commar sep string used to output the distances in the outputs folder as
    # cct_uid_string_005 and cct_uid_string_006
    # rest_analysis.get_list_of_clade_col_type_uids_for_unifrac()
    # code to make the dendrogram figure. The high_low option will take either 'high' or 'low'.
    # If high is provided the 0.40 cutoff will be used. If low is passed the 0.05-0.40 cutoff range will be used
    # rest_analysis.make_dendrogram_with_meta_all_clades(high_low='background')
    # rest_analysis.report_on_fidelity_proxies_for_profile_associations()
    # rest_analysis.report_on_reef_type_effect_metrics()
    # rest_analysis.make_networks()
    # rest_analysis.assess_balance_and_dispersions_of_distance_matrix()
    # rest_analysis.output_seq_analysis_overview_outputs()
    # rest_analysis.plot_pcoa_of_cladal()
    rest_analysis._plot_temperature()
    # rest_analysis._quaternary_plot()
    # rest_analysis.make_sample_balance_figure()
    # run this to write out the distance files for running permanova in R
    # rest_analysis.permute_sample_permanova()
    # rest_analysis.make_sample_balance_figure()
    # rest_analysis.permute_profile_permanova()
    # rest_analysis.histogram_of_all_abundance_values()
    # rest_analysis.investigate_background()
    # rest_analysis.get_list_of_clade_col_type_uids_for_unifrac()






