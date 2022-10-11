rm(list=ls())
cat("\014")
library(readxl)
library(rpart)
library(caret)
library(rpart.plot)
#install.packages('e1071')
library(e1071)
library(corrplot)
#install.packages('mltools')
library('mltools')
#install.packages('data.table')
library('data.table')
#install.packages("neuralnet")
library('neuralnet')
#install.packages('EMCluster')
library(EMCluster)
library(factoextra)
#install.packages("RandPro")
library(RandPro)
#install.packages("ica")
library(ica)

setwd("~/Documents/1.MSBA/Summer'22/Machine Learning/Assignments/1")
data.org <- read_excel("SeoulBikeData.xls")
data<- data.org
summary(data)
data$Seasons <- as.factor(data$Seasons)
data$Holiday <- as.factor(data$Holiday)
data$`Functioning Day` <- as.factor(data$`Functioning Day`)
data <- one_hot(as.data.table(data))

#removing reference group
data <- subset(data,select = -c(`Holiday_Holiday`,`Functioning Day_No`, Seasons_Autumn))

#Y variable
data$`Rented Bike Count`<- ifelse(data$`Rented Bike Count` > mean(data$`Rented Bike Count`)
                                    ,1,0)

data <- subset(data, select=-c(Date, month, year,`Dew point temperature(∞C)`))


######################Clustering##############################
set.seed(123)
fviz_nbclust(data, kmeans, nstart = 10, k.max = 10, method = "silhouette")

#######Kmeans
Kmeans <- kmeans(data, centers = 3, nstart = 10)
class <- as.data.frame(Kmeans$cluster)
data1<- cbind(data,class)
colnames(data1)[15] <- "k.label"


agg<- aggregate(. ~ k.label, data=data1, mean )

agg
  g <- ggplot(data1, aes(x=data1$`Visibility (10m)`, y=data1$`Temperature(∞C)`)) 
  g + geom_text(aes(label=k.label, color=k.label),hjust=0, vjust=0)  # label dot using the cluster name


##########EMCluster
scale.data <- as.data.frame(scale(data))
set.seed(12)
emobj <- simple.init(scale.data, nclass = 3)
emobj <- shortemcluster(scale.data, emobj)
summary(emobj)

ret <- emcluster(scale.data, emobj, assign.class = TRUE)
summary(ret)

class2 <- as.data.frame(ret$class)
data1<- cbind(data1,class2)
colnames(data1)[16] <- "em.label"


g <- ggplot(data1, aes(x=data1$`Temperature(∞C)`, y=data$`Functioning Day_Yes`))
g + geom_text(aes(label=em.label, color=em.label),hjust=0, vjust=0)  # label dot using the cluster name

help(em.lab)



######################Dimension reduction algorithms##############################

##########################Decission tree
fit.big <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
                 data=data, # dataframe used
                 method="class",  # treat churn as a categorical variable, default
                 control=rpart.control(xval=10, minsplit=1000), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
                 parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

prp(fit.big, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    
#Important features are temperature, hour, humidity

##########################PCA
pca.out = prcomp(data[,-4]) #excluding y
summary(pca.out)
pca.scores <- as.data.frame(pca.out$x)
pca.scores <- cbind(data$`Rented Bike Count`, pca.scores)
colnames(pca.scores)[1] <- "Rented Bike Count"


######################ICA
ica.out <- ica(data[,-1], nc=13)#excluding y
summary(ica.out)
ica.scores <-as.data.frame(ica.out$S)
ica.scores <- cbind(ica.scores,data$`Rented Bike Count`)
colnames(ica.scores)[14] <- "Rented Bike Count"


fit.big <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
                 data=ica.scores, # dataframe used
                 method="class",  # treat churn as a categorical variable, default
                 control=rpart.control(xval=10, minsplit=1000), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
                 parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

prp(fit.big, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    
#V4, V8, V3, features are selected.

######################RP
library('RandPro')
RP.scores <- form_matrix(8760,13, F)
data.matrix <- as.matrix(data[,-1])
RP.scores <- RP.scores * data.matrix
RP.scores <- as.data.frame(cbind(data$`Rented Bike Count`, RP.scores))
colnames(RP.scores)[1] <- "Rented Bike Count"

fit.big <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
                 data=RP.scores, # dataframe used
                 method="class",  # treat churn as a categorical variable, default
                 control=rpart.control(xval=10, minsplit=1000), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
                 parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

prp(fit.big, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    
#Seasons_winter, Solar_rad, Hour are selected.


#########################Kmeans + Dimension reduction###########
###########Kmeans + PCA


Kmeans <- kmeans(pca.scores[,1:3],  centers = 3, nstart = 10)
pca.scores$cluster <- as.character(Kmeans$cluster)
g <- ggplot(pca.scores, aes(PC1, PC2)) 
g + geom_text(aes(label=cluster, color=cluster),hjust=0, vjust=0)  # label dot using the cluster name



###########Kmeans + ICA
#V4, V8, V3, V9 features are selected.

Kmeans <- kmeans(ica.scores[,c(4,8,3,9)],  centers = 3, nstart = 10)
ica.scores$cluster <- as.character(Kmeans$cluster)
g <- ggplot(ica.scores, aes(V4,V8)) 
g + geom_text(aes(label=cluster, color=cluster),hjust=0, vjust=0)  # label dot using the cluster name



###########Kmeans + RP
#Seasons_winter, Solar_rad, Hour are selected.

Kmeans <- kmeans(RP.scores[,c(12,7,2)],  centers = 3, nstart = 10)
RP.scores$cluster <- as.character(Kmeans$cluster)
g <- ggplot(RP.scores, aes(Seasons_Winter, Hour)) 
g + geom_text(aes(label=cluster, color=cluster),hjust=0, vjust=0)  # label dot using the cluster name


#########################EM + Dimension reduction###########
############EM + PCA
set.seed(12)
emobj <- simple.init(pca.scores[,1:3], nclass = 3)
emobj <- shortemcluster(pca.scores[,1:3], emobj)
summary(emobj)

ret <- emcluster(pca.scores[,1:3], emobj, assign.class = TRUE)
summary(ret)

class2 <- as.data.frame(ret$class)
pca.scores <- cbind(pca.scores, class2)
colnames(pca.scores)[16] <- "cluster2"


g <- ggplot(pca.scores, aes(PC1, PC2))
g + geom_text(aes(label=cluster2, color=cluster2),hjust=0, vjust=0)  # label dot using the cluster name

############EM + ICA
set.seed(12)
emobj <- simple.init(ica.scores[,c(4,8,3,9)], nclass = 3)
emobj <- shortemcluster(ica.scores[,c(4,8,3,9)], emobj)
summary(emobj)

ret <- emcluster(ica.scores[,c(4,8,3,9)], emobj, assign.class = TRUE)
summary(ret)

class2 <- as.data.frame(ret$class)
ica.scores <- cbind(ica.scores, class2)
colnames(ica.scores)[16] <- "cluster2"


g <- ggplot(ica.scores, aes(V4, V8))
g + geom_text(aes(label=cluster2, color=cluster2),hjust=0, vjust=0)  # label dot using the cluster name

############EM + RP
set.seed(12)
emobj <- simple.init(RP.scores[,c(12,7,2)], nclass = 3)
emobj <- shortemcluster(RP.scores[,c(12,7,2)], emobj)
summary(emobj)

ret <- emcluster(RP.scores[,c(12,7,2)], emobj, assign.class = TRUE)
summary(ret)

class2 <- as.data.frame(ret$class)
RP.scores <- cbind(RP.scores, class2)
colnames(RP.scores)[16] <- "cluster2"


g <- ggplot(RP.scores, aes(Seasons_Winter, `Solar Radiation (MJ/m2)`))
g + geom_text(aes(label=cluster2, color=cluster2),hjust=0, vjust=0)  # label dot using the cluster name




#########################Dimension reduction + neural network###########
#Decission tree
#creating reduced data sets
#Important features are temperature, hour, humidity
dt.df <-as.data.frame(cbind(data$`Rented Bike Count`, data$`Temperature(∞C)`, data$Hour, data$`Humidity(%)`)) 


set.seed(13)
sample <- sample(1:nrow(dt.df), (0.7)*nrow(dt.df))
dt.df.train <- dt.df[sample,]
dt.df.test <- dt.df[-sample,]

nn <- neuralnet(V1 ~.,
                data=dt.df.train, hidden = 3)

plot(nn)
#Train accuracy
actual <- dt.df.train$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,dt.df.train)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))

#Test accuracy
actual <- dt.df.test$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,dt.df.test)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))


#PCA
#creating reduced data sets
#Important features
pca.df <- pca.scores[,1:3]
set.seed(13)
sample <- sample(1:nrow(pca.df), (0.7)*nrow(pca.df))
pca.df.train <- pca.df[sample,]
pca.df.test <- pca.df[-sample,]

nn <- neuralnet(`Rented Bike Count` ~.,
                data=pca.df.train, hidden = 3, act.fct = "tanh" )

plot(nn)
actual <- pca.df.train$`Rented Bike Count`
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,pca.df.train)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))

help("neuralnet")


#ICA
#creating reduced data sets
#Important features 
#V4, V8, V3,V9 features are selected.
ica.df <-as.data.frame(cbind(ica.scores$`Rented Bike Count`, ica.scores$V4, ica.scores$V8,ica.scores$V3,ica.scores$V9)) 

set.seed(13)
sample <- sample(1:nrow(ica.df), (0.7)*nrow(ica.df))
ica.df.train <- ica.df[sample,]
ica.df.test <- ica.df[-sample,]

nn <- neuralnet(V1 ~.,
                data=ica.df.train, hidden = 3)

plot(nn)
#Train
actual <- ica.df.train$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,ica.df.train)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))

#Test
actual <- ica.df.test$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,ica.df.test)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))

#RP
#Seasons_winter, Solar_rad, Hour are selected.
rp.df <-as.data.frame(cbind(RP.scores$`Rented Bike Count`,
                            RP.scores$Seasons_Winter, 
                            RP.scores$`Solar Radiation (MJ/m2)`,
                            RP.scores$Hour))

set.seed(13)
sample <- sample(1:nrow(rp.df), (0.7)*nrow(rp.df))
rp.df.train <- rp.df[sample,]
rp.df.test <- rp.df[-sample,]

nn <- neuralnet(V1 ~.,
                data=rp.df.train, hidden = 3,)

plot(nn)
#Train
actual <- rp.df.train$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,rp.df.train)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))

#Test
actual <- rp.df.test$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,rp.df.test)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))



#########################ANN on cluster lables
cl.df <-as.data.frame(cbind(data1$`Rented Bike Count`,
                            data1$k.label,
                            data1$em.label))


set.seed(13)
sample <- sample(1:nrow(cl.df), (0.7)*nrow(cl.df))
cl.df.train <- cl.df[sample,]
cl.df.test <- cl.df[-sample,]

nn <- neuralnet(V1 ~.,
                data=cl.df.train, hidden = 3,)

plot(nn)
#Train
actual <- cl.df.train$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,cl.df.train)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))

#Test
actual <- cl.df.test$V1
actual <- as.character(actual)
actual<- as.factor(actual)
predict <- compute(nn,cl.df.test)
prob <- predict$net.result
predict <- ifelse(prob>0.5, 1, 0)
predict <- as.factor(predict)
levels(predict) <- c(0,1)
confusionMatrix(table(predict,actual))







