library(vegan)
# output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
# setwd(output_path)
# dis_matrix = as.dist(data.matrix(read.table(file="dists_A.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
# meta_info = read.table(file='meta_info_A.csv', header=TRUE, sep=',')
# adonis(formula = dis_matrix ~ reef*reef_type*depth*season*species, data=meta_info)

# SAMPLE PERMANOVAs and tests of heterogeneity of dispersion using PERMDISP2
# Clade A
output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
setwd(output_path)
dis_matrix = as.dist(data.matrix(read.table(file="dists_permanova_samples_A_braycurtis.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
# meta_info = read.table(file='sample_meta_info_A.csv', header=TRUE, sep=',')
meta_info = read.table(file='sample_meta_info_A_braycurtis.csv', header=TRUE, sep=',')
adonis(formula = dis_matrix ~ reef_type*reef*species*depth*season, data=meta_info)
# adonis(formula = dis_matrix ~ reef_type_shuffled*reef_shuffled*species_shuffled*depth_shuffled*season_shuffled, data=meta_info)
# adonis(formula = dis_matrix ~ reef_type+reef+species+depth+season, data=meta_info)

# Tests of heterogeneity of dispersion using PERMDISP2
# Reef_type
mod_reef_type = betadisper(dis_matrix, meta_info$reef_type)
anova(mod_reef_type)
(mod_reef_type.HSD = TukeyHSD(mod_reef_type))
permutest(mod_reef_type, pairwise=TRUE)
plot(mod_reef_type, segments = FALSE, col = c('cyan', 'blue', 'blue4'), ellipse=TRUE)

# Reef
mod_reef = betadisper(dis_matrix, meta_info$reef)
anova(mod_reef)
(mod_reef.HSD = TukeyHSD(mod_reef))
permutest(mod_reef, pairwise=TRUE)
plot(mod_reef, segments = FALSE, ellipse=TRUE)

# Species
mod_species = betadisper(dis_matrix, meta_info$species)
anova(mod_species)
(mod_species.HSD = TukeyHSD(mod_species))
plot(mod_species, segments = FALSE, ellipse=TRUE)

# Depth
mod_depth = betadisper(dis_matrix, meta_info$depth)
anova(mod_depth)
(mod_depth.HSD = TukeyHSD(mod_depth))
permutest(mod_depth, pairwise=TRUE)
plot(mod_depth, segments = FALSE, col = c('cyan', 'blue', 'blue4'), ellipse=TRUE)

# Turns out that of all the P<0.01 are comparisons ST. As such ST may be worth leaving out.
# Season, non significant
mod_season = betadisper(dis_matrix, meta_info$season)
anova(mod_season)
(mod_season.HSD = TukeyHSD(mod_season))
plot(mod_season, segments = FALSE, ellipse=TRUE)


# Clade C
output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
setwd(output_path)
dis_matrix = as.dist(data.matrix(read.table(file="dists_permanova_samples_C_unifrac_only_maj.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
meta_info = read.table(file='sample_meta_info_C_unifrac_only_maj.csv', header=TRUE, sep=',')
adonis(formula = dis_matrix ~ reef_type*reef*species*depth*season, data=meta_info)
# SAMPLE PERMANOVAs and tests of heterogeneity of dispersion using PERMDISP2
# Tests of heterogeneity of dispersion using PERMDISP2

# Reef_type
mod_reef_type = betadisper(dis_matrix, meta_info$reef_type)
anova(mod_reef_type)
(mod_reef_type.HSD = TukeyHSD(mod_reef_type))
permutest(mod_reef_type, pairwise=TRUE)
plot(mod_reef_type, segments = FALSE, ellipse=TRUE)

# Reef
mod_reef = betadisper(dis_matrix, meta_info$reef)
anova(mod_reef)
(mod_reef.HSD = TukeyHSD(mod_reef))
permutest(mod_reef, pairwise=TRUE)
plot(mod_reef, segments = FALSE, ellipse=TRUE)

# Species
mod_species = betadisper(dis_matrix, meta_info$species)
anova(mod_species)
(mod_species.HSD = TukeyHSD(mod_species))
plot(mod_species, segments = FALSE, ellipse=TRUE)

# Depth
mod_depth = betadisper(dis_matrix, meta_info$depth)
anova(mod_depth)
(mod_depth.HSD = TukeyHSD(mod_depth))
permutest(mod_depth, pairwise=TRUE)
plot(mod_depth, segments = FALSE, col = c('cyan', 'blue', 'blue4'), ellipse=TRUE)

# Season, non significant
mod_season = betadisper(dis_matrix, meta_info$season)
anova(mod_season)
(mod_season.HSD = TukeyHSD(mod_season))
plot(mod_season, segments = FALSE, ellipse=TRUE)

# Clade D
output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
setwd(output_path)
dis_matrix = as.dist(data.matrix(read.table(file="dists_permanova_samples_D_unifrac_no_se.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
meta_info = read.table(file='sample_meta_info_D_unifrac_no_se.csv', header=TRUE, sep=',')
adonis(formula = dis_matrix ~ reef_type*reef*species*depth*season, data=meta_info)

# Reef_type
mod_reef_type = betadisper(dis_matrix, meta_info$reef_type)
anova(mod_reef_type)
(mod_reef_type.HSD = TukeyHSD(mod_reef_type))
#permutest(mod_reef_type, pairwise=TRUE)
plot(mod_reef_type, segments = FALSE, ellipse=TRUE)

# Reef
mod_reef = betadisper(dis_matrix, meta_info$reef)
anova(mod_reef)
(mod_reef.HSD = TukeyHSD(mod_reef))
#permutest(mod_reef, pairwise=TRUE)
plot(mod_reef, segments = FALSE, ellipse=TRUE)

# Tests of heterogeneity of dispersion using PERMDISP2
# Species
mod_species = betadisper(dis_matrix, meta_info$species)
anova(mod_species)
(mod_species.HSD = TukeyHSD(mod_species))
plot(mod_species, segments = FALSE, ellipse=TRUE)

# Depth
mod_depth = betadisper(dis_matrix, meta_info$depth)
anova(mod_depth)
(mod_depth.HSD = TukeyHSD(mod_depth))
# permutest(mod_depth, pairwise=TRUE)
plot(mod_depth, segments = FALSE, col = c('cyan', 'blue', 'blue4'), ellipse=TRUE)

# Season
mod_season = betadisper(dis_matrix, meta_info$season)
anova(mod_season)
(mod_season.HSD = TukeyHSD(mod_season))
plot(mod_season, segments = FALSE, ellipse=TRUE)


# CLADE PROPORTIONS PERMANOVA
output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
setwd(output_path)
dis_matrix = as.dist(data.matrix(read.table(file="between_sample_clade_proportion_distances_no_se.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
meta_info = read.table(file='sample_meta_for_clade_proportion_permanova_no_se.csv', header=TRUE, sep=',')
adonis(formula = dis_matrix ~ reef_type*reef*species*depth*season, data=meta_info)

# Reef_type
mod_reef_type = betadisper(dis_matrix, meta_info$reef_type)
anova(mod_reef_type)
(mod_reef_type.HSD = TukeyHSD(mod_reef_type))
permutest(mod_reef_type, pairwise=TRUE)
plot(mod_reef_type, segments = FALSE, ellipse=TRUE)

# Reef
mod_reef = betadisper(dis_matrix, meta_info$reef)
anova(mod_reef)
(mod_reef.HSD = TukeyHSD(mod_reef))
permutest(mod_reef, pairwise=TRUE)
plot(mod_reef, segments = FALSE, ellipse=TRUE)

# Tests of heterogeneity of dispersion using PERMDISP2
# Species
mod_species = betadisper(dis_matrix, meta_info$species)
anova(mod_species)
(mod_species.HSD = TukeyHSD(mod_species))
plot(mod_species, segments = FALSE, ellipse=TRUE)

# Season
mod_season = betadisper(dis_matrix, meta_info$season)
anova(mod_season)
(mod_season.HSD = TukeyHSD(mod_season))
plot(mod_season, segments = FALSE, ellipse=TRUE)

# Depth
mod_depth = betadisper(dis_matrix, meta_info$depth)
anova(mod_depth)
(mod_depth.HSD = TukeyHSD(mod_depth))
permutest(mod_depth, pairwise=TRUE)
plot(mod_depth, segments = FALSE, col = c('cyan', 'blue', 'blue4'), ellipse=TRUE)


