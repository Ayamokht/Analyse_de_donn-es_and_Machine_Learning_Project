library(FactoMineR)
library(factoextra)
library(cluster)


#Exercice 2 
voitures <- read.table("voitures",header=T,row.names=1)

#ACP
voitures.pca <- PCA(voitures,scale.unit = TRUE,graph=F)

plot(voitures.pca,choix="var")
plot(voitures.pca,choix="ind")

#valeurs propres
voitures.pca$eig 

#cos² 
voitures.pca$ind$cos2
voitures.pca$var$cos2



#Exercice 3 
voitures <- read.table("voitures",header=T,row.names=1)

#Somme du carrés des distances intra-cluster, clusters de 1 à k_max.
k_max <- 8
wss <- sapply(1:k_max, function(k){kmeans(voitures, centers = k)$tot.withinss})

# Méthode du coude
plot(1:k_max, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Nombre de clusters (k)", ylab = "Somme des carrés des distances intra-cluster",
     main = "Méthode du coude pour choisir k")

#K-means
voitures.cluster1 <- kmeans(voitures,3,nstart=100)
voitures.cluster1

voitures.cluster <- kmeans(voitures,4,nstart=100)
voitures.cluster

#CAH 
distances <- dist(voitures)

hierarchical_cluster <- hclust(distances, method = "ward.D")
plot(hierarchical_cluster, main = "Dendrogramme")



#Exercice 4

chiens <- read.table("chiens", header = TRUE, row.names = 1)

#déclaration des variables catégorielles 
chiens$TAI <- as.factor(chiens$TAI)
chiens$POI <- as.factor(chiens$POI)
chiens$VEL <- as.factor(chiens$VEL)
chiens$INT <- as.factor(chiens$INT)
chiens$AFF <- as.factor(chiens$AFF)
chiens$AGR <- as.factor(chiens$AGR)
chiens$FON <- as.factor(chiens$FON)

#ACM
chiens.mca <- MCA(chiens, quali.sup = 7)

plot(chiens.mca,choix="ind")

#valeurs propres
chiens.mca$eig

#contributions
chiens.mca$var$contrib[,1:2]
chiens.mca$ind$contrib[,1:2]

#cos²
chiens.mca$var$cos2
chiens.mca$ind$cos2

