library(vegan)
output_path = '/Users/humebc/Google_Drive/projects/alejandro_et_al_2018/restrepo_git_repo/outputs/'
setwd(output_path)
dis_matrix = as.dist(data.matrix(read.table(file="dists_A.csv", header=FALSE, sep=',', stringsAsFactors = FALSE)))
meta_info = read.table(file='meta_info_A.csv', header=TRUE, sep=',')
adonis(formula = dists ~ reef*reef_type*depth*species*season, data=meta_info)