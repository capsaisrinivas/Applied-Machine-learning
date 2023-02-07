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

####################Data Preprocessing#############################
setwd("/Users/saisrinivas/Documents/1.MSBA/Courses/6.Machine Learning/Assignments/1 Linear regression")
data.org <- read_excel("SeoulBikeData.xls")
data<- data.org
summary(data)

#converting to factor for hot encoding(Only factor features are hot encoded)
data$Hour <- as.factor(data$Hour)
data$Seasons <- as.factor(data$Seasons)
data$Holiday <- as.factor(data$Holiday)
data$`Functioning Day` <- as.factor(data$`Functioning Day` )

data <- one_hot(as.data.table(data))

#Removing reference group for holiday and functioing day and Hour but not for seasons as later delete based on correlation
data <- subset(data,select = -c(Hour_0, `Holiday_No Holiday`,`Functioning Day_No`))





#Correlation plot(correlatoin doesnt work on factors)
cor1<- cor(data[,-c(5:27)])#Excluding Hours
corrplot(cor1,        # Correlation matrix
         method = "shade", # Correlation plot method
         type = "full",    # Correlation plot style (also "upper" and "lower")
         diag = TRUE,      # If TRUE (default), adds the diagonal
         tl.col = "black", # Labels color
         bg = "white",     # Background color
         title = "",       # Main title
         col = NULL)       # Color palette


cor2<- cor(data[,-c(5:27,32,38)])#exluding  few correlated columns columns(dew point and summer)
corrplot(cor2,        # Correlation matrix
         method = "shade", # Correlation plot method
         type = "full",    # Correlation plot style (also "upper" and "lower")
         diag = TRUE,      # If TRUE (default), adds the diagonal
         tl.col = "black", # Labels color
         bg = "white",     # Background color
         title = "",       # Main title
         col = NULL)       # Color palette


#Removing columns 
data <- subset(data, select =-c(`Dew point temperature(∞C)`, Seasons_Summer, Seasons_Winter))


#Binary Labeling the output variable
data$`Rented Bike Count`<- ifelse(data$`Rented Bike Count` > mean(data$`Rented Bike Count`)
                                  ,"High","Low")
data$`Rented Bike Count` <- as.factor(data$`Rented Bike Count`)



#dividing data into train and test
set.seed(123)
sample <- sample(1:nrow(data), (0.7)*nrow(data))
train <- data[sample,]
test <- data[-sample,]





######################Decission tree -1(Gini); learning curve on size###################################
start<-Sys.time()
fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
             data=train, # dataframe used
             method="class",  # treat churn as a categorical variable, default
             control=rpart.control(xval=0, minsplit=0), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
             parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
end <- Sys.time()
duration <- end - start;duration
#decision tree plot
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    

#confusion matrix
predict <- predict(fit,train, type="class")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

predict <- predict(fit, test, type="class")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')




#learning curve
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
               data=train[1:i,], # dataframe used
               method="class",  # treat churn as a categorical variable, default
               control=rpart.control(xval=0, minsplit=0), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
               parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
  predict.tr <- predict(fit,train[1:i,], type="class")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve(Gini)",ylim = c(0,25) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('bottomright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))


######################Decission tree -2(Entropy); learning curve on size################################
start<-Sys.time()
fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
             data=train, # dataframe used
             method="class",  # treat churn as a categorical variable, default
             control=rpart.control(xval=0, minsplit=0), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
             parms=list(split="information"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
end <-Sys.time()
duration<- end - start;duration
#decision tree plot
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    

#confusion matrix
predict <- predict(fit,train, type="class")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

predict <- predict(fit, test, type="class")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')


#Learning curve
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
               data=train[1:i,], # dataframe used
               method="class",  # treat churn as a categorical variable, default
               control=rpart.control(xval=0, minsplit=0), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
               parms=list(split="information"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
  predict.tr <- predict(fit,train[1:i,], type="class")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve(Entropy)",ylim = c(0,25) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('bottomright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))

###################Decission tree -learning curve on depth; 9 is best inline with above####################
#Learning curve(Gini Index)
learnCurve <- data.frame(m = integer(20),
                         train.error = integer(20),
                         test.error = integer(20))

for(i in 1:20) {
  learnCurve$m[i] <- i
  fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
               data=train, # dataframe used
               method="class",  # treat churn as a categorical variable, default
               control=rpart.control(xval=0, minsplit=0, maxdepth = i), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
               parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
  predict.tr <- predict(fit,train, type="class")
  actual.tr <- train$`Rented Bike Count`
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Depth of tree",
     ylab = "Error%", main = "learning Curve",ylim = c(0,30) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))





################### Decission tree - learning curve on pre pruning######
#Learning curve
learnCurve <- data.frame(m = integer(10),
                         train.error = integer(10),
                         test.error = integer(10))

for(i in c(50,100,150,200,200,250,300,350,400,450,500)) {
  learnCurve$m[i/50] <- i
  fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
               data=train, # dataframe used
               method="class",  # treat churn as a categorical variable, default
               control=rpart.control(xval=0, minsplit=i), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
               parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
  predict.tr <- predict(fit,train, type="class")
  actual.tr <- train$`Rented Bike Count`
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/50] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/50] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Minsplit",
     ylab = "Error%", main = "learning Curve",ylim = c(10,20) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('bottomright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))






####################Decissin tree -3(Pre pruning - 200)  & learning curve#########################
fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
             data=train, # dataframe used
             method="class",  # treat churn as a categorical variable, default
             control=rpart.control(xval=0, minsplit=200), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
             parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

#decision tree plot
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    

#confusion matrix
predict <- predict(fit,train, type="class")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

predict <- predict(fit, test, type="class")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Learning curve
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
               data=train[1:i,], # dataframe used
               method="class",  # treat churn as a categorical variable, default
               control=rpart.control(xval=0, minsplit=200), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
               parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
  predict.tr <- predict(fit,train[1:i,], type="class")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve (pre pruning @ Minsplit=200)",ylim = c(0,25) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))






####################Decission tree -5(Post pruning with cross validation)######################################
fit.big <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
             data=train, # dataframe used
             method="class",  # treat churn as a categorical variable, default
             control=rpart.control(xval=10, minsplit=0, cp=0), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
             parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")

#decision tree plot - taking too long to run given the size of dataset
#prp(fit.big, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    


bestcp <- fit.big$cptable[which.min(fit.big$cptable[,"xerror"]),"CP"]
fit.post <- prune.rpart(fit.big, cp=bestcp)
prp(fit.post, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Rental Prediction")    

fit.big$cptable

#confusion matrix
predict <- predict(fit.post,train, type="class")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

predict <- predict(fit.big, test, type="class")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')


#Learning curve
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- rpart(`Rented Bike Count` ~., # formula, all predictors will be considered in splitting
               data=train[1:i,], # dataframe used
               method="class",  # treat churn as a categorical variable, default
               control=rpart.control(xval=10, minsplit=0, cp=bestcp), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
               parms=list(split="gini"))  # criterial for splitting: gini default, entropy if set parms=list(split="information")
  predict.tr <- predict(fit,train[1:i,], type="class")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning Curve(post pruning at best CP)",ylim = c(0,25) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))




















################ SVM - C-classification, Linear Model ############################
fit = svm(`Rented Bike Count` ~ .,
data = train,
type = 'C-classification',
kernel = 'linear', scale = T)

#Train error
predict <- predict(fit,train, type="C-classification")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Test error
predict <- predict(fit,test, type="C-classification")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Lerning curve
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train[1:i,],
             type = 'C-classification',
             kernel = 'linear', scale = T)  
  predict.tr <- predict(fit,train[1:i,], type="C-classification")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="C-classification")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve(SVM Linear - C classification)",ylim = c(8,12) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))


################ SVM - Nu-classification, Linear Model(nu=0.3 is best) ############################

#Lerning curve for nu
learnCurve <- data.frame(m = integer(7),
                         train.error = integer(7),
                         test.error = integer(7))

for(i in c(0.1,0.2,0.3,0.4,0.5,0.6,0.7)) {
  learnCurve$m[i*10] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train,
             type = 'nu-classification',
             kernel = 'linear', scale = T,
             nu = i)
  predict.tr <- predict(fit,train, type="nu-classification")
  actual.tr <- train$`Rented Bike Count`
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="nu-classification")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i*10] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i*10] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Nu parameter",
     ylab = "Error%", main = "Learning curve",ylim = c(9,34) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))

#0.3 is best
#Accuracy
fit = svm(`Rented Bike Count` ~ .,
          data = train,
          type = 'nu-classification',
          kernel = 'linear', scale = T,
          nu=0.3)

#Train error
predict <- predict(fit,train, type="C-classification")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Test error
predict <- predict(fit,test, type="C-classification")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')



#Learning
#Lerning curve on training data set @ nu=0.3
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train[1:i,],
             type = 'nu-classification',
             kernel = 'linear', scale = T,
             nu=0.3)
  predict.tr <- predict(fit,train[1:i,], type="nu-classification")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="nu-classification")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve(SVM Linear - nu classification; nu=0.3)",ylim = c(8,12) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('bottomright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))



################ SVM - Kernal-polynomial(degree=2 is best) ############################

#Lerning curve on degree(ex: if we have 1 feature, 3rd degree projection means features are increased with X^3,X^2,X ie increasing dimensions)
learnCurve <- data.frame(m = integer(10),
                         train.error = integer(10),
                         test.error = integer(10))

for(i in c(1,2,3,4,5,6,7,8,9,10)) {
  learnCurve$m[i] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train,
             type = 'C-classification',
             kernel = 'polynomial', scale = T,
             degree = i)
  predict.tr <- predict(fit,train, type="C-classification")
  actual.tr <- train$`Rented Bike Count`
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="C-classification")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Degree of polynomial",
     ylab = "Error%", main = "Learning curve",ylim = c(9,13) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('bottomright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))

#2 degrees is optimum
#Accuracy
fit = svm(`Rented Bike Count` ~ .,
          data = train,
          type = 'C-classification',
          kernel = 'polynomial', scale = T,
          degree = 2)

#Train error
predict <- predict(fit,train, type="nu-classification")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Test error
predict <- predict(fit,test, type="C-classification")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')


#Learning curve at 2nd degree on training data set
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train[1:i,],
             type = 'C-classification',
             kernel = 'polynomial', scale = T,
             degree =2)
  predict.tr <- predict(fit,train[1:i,], type="C-classification")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve(SVM Kernal @2nd degree)",ylim = c(5,18) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))



################ SVM - Kernal-radial(no effect on gama parameter) ############################
fit = svm(`Rented Bike Count` ~ .,
          data = train,
          type = 'C-classification',
          kernel = 'radial', scale = T)

#Train error
predict <- predict(fit,train, type="nu-classification")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Test error
predict <- predict(fit,test, type="C-classification")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Lerning curve on training data set
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train[1:i,],
             type = 'C-classification',
             kernel = 'radial', scale = T)
  predict.tr <- predict(fit,train[1:i,], type="C-classification")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve(SVM Kernal: radial)",ylim = c(5,18) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))

###################Experiment-1 (nu=0.3 and degree =2)####################
fit = svm(`Rented Bike Count` ~ .,
          data = train,
          type = 'nu-classification',
          kernel = 'polynomial', scale = T,
          degree = 2,
          nu=0.3)

#Train error
predict <- predict(fit,train, type="nu-classification")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Test error
predict <- predict(fit,test, type="C-classification")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Lerning curve on training data set
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train[1:i,],
             type = 'nu-classification',
             kernel = 'polynomial', scale = T,
             degree=2,
             nu=0.3)
  predict.tr <- predict(fit,train[1:i,], type="C-classification")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve experiment(SVM nu=0.3; 2nd degree polynomial",ylim = c(5,18) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))



###################Experiment-2 (nu=0.3 and Radial)####################
fit = svm(`Rented Bike Count` ~ .,
          data = train,
          type = 'nu-classification',
          kernel = 'radial', scale = T,
          nu=0.3)

#Train error
predict <- predict(fit,train, type="nu-classification")
actual <- train$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Test error
predict <- predict(fit,test, type="C-classification")
actual <- test$`Rented Bike Count`
confusionMatrix(table(predict,actual), positive='High')

#Lerning curve on training data set
learnCurve <- data.frame(m = integer(12),
                         train.error = integer(12),
                         test.error = integer(12))

for(i in c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)) {
  learnCurve$m[i/500] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train[1:i,],
             type = 'nu-classification',
             kernel = 'polynomial', scale = T,
             degree=2,
             nu=0.3)
  predict.tr <- predict(fit,train[1:i,], type="C-classification")
  actual.tr <- head(train$`Rented Bike Count`,i)
  cm.tr <- confusionMatrix(table(predict.tr,actual.tr), positive='High')
  predict.ts <- predict(fit,test, type="class")
  actual.ts <- test$`Rented Bike Count`
  cm.ts <- confusionMatrix(table(predict.ts,actual.ts), positive='High')
  
  learnCurve$train.error[i/500] <- 100-(cm.tr$overall[["Accuracy"]]*100)
  learnCurve$test.error[i/500] <- 100-(cm.ts$overall[["Accuracy"]]*100)
}


plot(x=learnCurve$m,y=learnCurve$test.error,type = "l",col = "red", xlab = "Training set size",
     ylab = "Error%", main = "Learning curve experiment(SVM nu=0.3; radial)",ylim = c(5,18) )
lines(x=learnCurve$m, y=learnCurve$train.error, type = "l", col = "blue")
legend('topright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))




###############SVM cross validation#############
#Lerning curve on cross validation error
learnCurve <- data.frame(m = integer(10),
                         cross.error = integer(10))

for(i in 1:10) {
  learnCurve$m[i] <- i
  fit <- svm(`Rented Bike Count` ~ .,
             data = train,
             type = 'nu-classification',
             kernel = 'radial', scale = T,
             nu=0.3,
             cross = i)
  cross.error<- mean(fit$accuracies)
  
  learnCurve$cross.error[i] <- 100-cross.error
}


plot(x=learnCurve$m,y=learnCurve$cross.error,type = "l",col = "red", xlab = "No. of folds",
     ylab = "cross validation error%", main = "Cross validation error(SVM nu=0.3; radial)",ylim = c(9,10) )
legend('bottomright', c("Test error","Train error" ), lty = c(1,1), lwd = c(1, 1),
       col = c("red", "blue"))



