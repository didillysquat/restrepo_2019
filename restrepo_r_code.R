# This R file is conserned with the Restrepo 2019 publication

# This part of the code was used to run the linear models related to the temperature profiles.

# TESTING of Tahla compared to the other sites
# We are testing for signficance of slope coefficient at the 1m depth and 15m depth
# 1m depth
output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
setwd(output_path)
library(lsmeans)
local_temp_df <- read.table(file="local_temp_df.csv", header=TRUE, sep=',', stringsAsFactors = TRUE)
# Important to change the depth to a factor rather than an int
local_temp_df$depth = as.factor(local_temp_df$depth)
m.interaction <- lm(temp ~ minutes_from_first_record*site, data=local_temp_df, subset=(depth==1))
anova(m.interaction)
m.interaction$coefficients
m.lst <- lstrends(m.interaction, "site", var="minutes_from_first_record")
pairs(m.lst)

# contrast                    estimate      SE  df t.ratio p.value
# al_fahal - qita_al_kirsh   -0.002136 0.00156 540 -1.369  0.5191 
# al_fahal - shib_nazar      -0.000235 0.00156 540 -0.151  0.9988 
# al_fahal - tahla           -0.006851 0.00156 540 -4.393  0.0001 
# qita_al_kirsh - shib_nazar  0.001901 0.00156 540  1.219  0.6151 
# qita_al_kirsh - tahla      -0.004716 0.00156 540 -3.024  0.0139 
# shib_nazar - tahla         -0.006617 0.00156 540 -4.243  0.0002 

# Testing at 15m
m.interaction <- lm(temp ~ minutes_from_first_record*site, data=local_temp_df, subset=(depth==15))
anova(m.interaction)
m.interaction$coefficients
m.lst <- lstrends(m.interaction, "site", var="minutes_from_first_record")
pairs(m.lst)

# contrast                    estimate      SE  df t.ratio p.value
# abu_madafi - al_fahal       0.002151 0.00163 675  1.320  0.6786 
# abu_madafi - qita_al_kirsh -0.001892 0.00163 675 -1.161  0.7735 
# abu_madafi - shib_nazar    -0.001544 0.00163 675 -0.948  0.8780 
# abu_madafi - tahla         -0.003475 0.00163 675 -2.133  0.2073 # DO ANCOVA
# al_fahal - qita_al_kirsh   -0.004043 0.00163 675 -2.482  0.0960 
# al_fahal - shib_nazar      -0.003695 0.00163 675 -2.268  0.1566 
# al_fahal - tahla           -0.005626 0.00163 675 -3.453  0.0053
# qita_al_kirsh - shib_nazar  0.000348 0.00163 675  0.213  0.9995 
# qita_al_kirsh - tahla      -0.001583 0.00163 675 -0.972  0.8678 # DO ANCOVA
# shib_nazar - tahla         -0.001931 0.00163 675 -1.185  0.7599 # DO ANCOVA

# Now do the required ancovas
m.interaction <- aov(temp ~ minutes_from_first_record*site, data=local_temp_df, subset=(depth==15 & (site=='al_fahal' | site=='tahla')))
summary(m.interaction)

# abu_madafi - tahla 15m
m.interaction <- aov(temp ~ minutes_from_first_record*site, data=local_temp_df, subset=(depth==15 & (site=='abu_madafi' | site=='tahla')))
summary(m.interaction)
m.interaction <- aov(temp ~ minutes_from_first_record+site, data=local_temp_df, subset=(depth==15 & (site=='abu_madafi' | site=='tahla')))
summary(m.interaction)

#                            Df Sum Sq Mean Sq F value Pr(>F)    
#  minutes_from_first_record   1  878.7   878.7  2604.8 <2e-16 ***
#  site                        1   39.5    39.5   117.2 <2e-16 ***
#  Residuals                 271   91.4     0.3                   
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# qita_al_kirsh - tahla 15m
m.interaction <- aov(temp ~ minutes_from_first_record*site, data=local_temp_df, subset=(depth==15 & (site=='qita_al_kirsh' | site=='tahla')))
summary(m.interaction)
m.interaction <- aov(temp ~ minutes_from_first_record+site, data=local_temp_df, subset=(depth==15 & (site=='qita_al_kirsh' | site=='tahla')))
summary(m.interaction)

#                           Df Sum Sq Mean Sq F value Pr(>F)    
#  minutes_from_first_record   1  915.8   915.8  2456.4 <2e-16 ***
#  site                        1   38.0    38.0   102.1 <2e-16 ***
#  Residuals                 271  101.0     0.4                   
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# shib_nazar - tahla 15m
m.interaction <- aov(temp ~ minutes_from_first_record+site, data=local_temp_df, subset=(depth==15 & (site=='shib_nazar' | site=='tahla')))
summary(m.interaction)

#                            Df Sum Sq Mean Sq F value Pr(>F)    
#  minutes_from_first_record   1  908.9   908.9  2804.8 <2e-16 ***
#  site                        1   38.7    38.7   119.3 <2e-16 ***
#  Residuals                 271   87.8     0.3                   
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# TESTING of the depths at each site
# Tahla
m.interaction <- aov(temp ~ minutes_from_first_record*depth, data=local_temp_df, subset=(site=='tahla'))
summary(m.interaction)
m.interaction$coefficients
m.lst <- lstrends(m.interaction, "depth", var="minutes_from_first_record")
pairs(m.lst)

# contrast estimate      SE  df t.ratio p.value
# 1 - 15    0.00961 0.00183 270 5.243   <.0001

# al_fahal
m.interaction <- lm(temp ~ minutes_from_first_record*depth, data=local_temp_df, subset=(site=='al_fahal'))
anova(m.interaction)
m.interaction$coefficients
m.lst <- lstrends(m.interaction, "depth", var="minutes_from_first_record")
pairs(m.lst)

# contrast estimate      SE  df t.ratio p.value
# 1 - 15    0.00838 0.00156 270 5.373   <.0001

# qita_al_kirsh
m.interaction <- lm(temp ~ minutes_from_first_record*depth, data=local_temp_df, subset=(site=='qita_al_kirsh'))
anova(m.interaction)
m.interaction$coefficients
m.lst <- lstrends(m.interaction, "depth", var="minutes_from_first_record")
pairs(m.lst)

# contrast estimate      SE  df t.ratio p.value
# 1 - 15    0.00648 0.00167 405  3.883  0.0004 
# 1 - 30    0.01990 0.00167 405 11.929  <.0001 
# 15 - 30   0.01342 0.00167 405  8.046  <.0001

# shib_nazar
m.interaction <- lm(temp ~ minutes_from_first_record*depth, data=local_temp_df, subset=(site=='shib_nazar'))
anova(m.interaction)
m.interaction$coefficients
m.lst <- lstrends(m.interaction, "depth", var="minutes_from_first_record")
pairs(m.lst)

# contrast estimate      SE  df t.ratio p.value
# 1 - 15    0.00492 0.00146 405  3.381  0.0023 
# 1 - 30    0.01790 0.00146 405 12.295  <.0001 
# 15 - 30   0.01298 0.00146 405  8.913  <.0001

# additionally test the 1-15 to see if there is a sig diff of the intercept
m.interaction <- aov(temp ~ minutes_from_first_record*depth, data=local_temp_df, subset=(site=='shib_nazar' & (depth==1 | depth==15)))
summary(m.interaction)
#                                  Df Sum Sq Mean Sq F value   Pr(>F)    
# minutes_from_first_record         1  968.9   968.9 4349.22  < 2e-16 ***
#  depth                             1    5.0     5.0   22.65 3.17e-06 ***
#  minutes_from_first_record:depth   1    2.6     2.6   11.66 0.000738 ***
#  Residuals                       270   60.1     0.2                     
# This showed us a significant effect of the depth effect at 0.0007 so we will consider the slope coefficients to be sig diff

# TESTING the intra day 1m, vs 15m vs 30m.
local_temp_intraday_df <- read.table(file="local_temp_intraday_df.csv", header=TRUE, sep=',', stringsAsFactors = TRUE)
local_temp_intraday_df$depth = as.factor(local_temp_intraday_df$depth)
sapply(local_temp_intraday_df, class)

m.interaction <- lm(temp ~ minutes_from_first_record*depth, data=local_temp_intraday_df)
anova(m.interaction)
summary(m.interaction)
m.interaction$coefficients
m.lst <- lstrends(m.interaction, "depth", var="minutes_from_first_record")
pairs(m.lst)
# contrast estimate       SE   df t.ratio p.value
# 1 - 15   -0.00708 0.000497 1501 -14.247 <.0001 
# 1 - 30   -0.00476 0.000642 1501  -7.410 <.0001 
# 15 - 30   0.00233 0.000620 1501   3.753 0.0005

# We will also test the means via standard approach
# http://www.sthda.com/english/wiki/kruskal-wallis-test-in-r#multiple-pairwise-comparison-between-groups
# http://www.sthda.com/english/wiki/one-way-anova-test-in-r
# Plot up the data
ggboxplot(local_temp_intraday_df, x="depth", y="temp", color="depth",
          palette= c("#00AFBB", "#E7B800", "#FC4E07"), order = c(1, 15, 30),
          ylab = "Temp", xlab = "Depth")
# Anova
res.aov = aov(temp ~ depth, data=local_temp_intraday_df)
summary(res.aov)
# Pairwise check and correct
TukeyHSD(res.aov)
# Test equal variance between samples
plot(res.aov, 1)
library("ggpubr")
library(car)
leveneTest(temp ~ depth, data = local_temp_intraday_df)
# Equal vairance assumption is FAILED
# Test normality
plot(res.aov, 2)
# Extract the residuals
aov_residuals <- residuals(object = res.aov )
# Run Shapiro-Wilk test
shapiro.test(x = aov_residuals )
#Normality assumptions are failed
# Thus perform kruskal-wallis test
res.kruskal <- kruskal.test(temp ~ depth, data = local_temp_intraday_df)
# For pairwise inferences wilcox
pairwise.wilcox.test(local_temp_intraday_df$temp, local_temp_intraday_df$depth,
                     p.adjust.method = "BH")


# This part of the code was used to run the PERMANOVA analyses, run the beradisper analyses, and to test significance of pairwise tests.
library(vegan)

# SAMPLE PERMANOVAs and tests of heterogeneity of dispersion using PERMDISP2
# Clade A
output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
setwd(output_path)
# For each of these clades there are vavious options that can be used for the dis_matrix file argument
# there is the unifrac version, the braycurtis version, the unifrac without s hystrix (clade D only) and the unifrac with only the majority samples
# these have the endings:
# dists_permanova_samples_D_braycurtis.csv
# dists_permanova_samples_D_unifrac.csv
# dists_permanova_samples_D_unifrac_no_se.csv
# dists_permanova_samples_D_unifrac_only_maj.csv

# then for the clade proportion matrices there is a no_se version that is without s. hystrix
# between_sample_clade_proportion_distances.csv
# between_sample_clade_proportion_distances_no_se.csv
dis_matrix = as.dist(data.matrix(read.table(file="dists_permanova_samples_A_unifrac_no_sqrt.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))

# Then for the meta info files there are also different versions that correspond to the distance matrices
# including one where I have shuffled up the samples to make sure that the results we were seeing are real.
# sample_meta_info_D_braycurtis.csv
# sample_meta_info_D_unifrac_no_se.csv
# sample_meta_info_D_unifrac_only_maj.csv
# sample_meta_info_D_shuffled.csv

meta_info = read.table(file='sample_meta_info_A_unifrac_no_sqrt.csv', header=TRUE, sep=',')
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
dis_matrix = as.dist(data.matrix(read.table(file="dists_permanova_samples_C_unifrac_no_sqrt.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
meta_info = read.table(file='sample_meta_info_C_unifrac_no_sqrt.csv', header=TRUE, sep=',')
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
dis_matrix = as.dist(data.matrix(read.table(file="dists_permanova_samples_D_unifrac_no_sqrt.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
meta_info = read.table(file='sample_meta_info_D_unifrac_no_sqrt.csv', header=TRUE, sep=',')
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


