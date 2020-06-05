#套件----
library(caret)
library(randomForest)
library(ROCR)
library(cluster)
library(ggplot2)
library(neuralnet)
library(Ckmeans.1d.dp)
library(DiagrammeR)



#設定Data------
data =read.csv("online_shoppers_intention.csv",header = T)
data$Revenue = as.numeric(data$Revenue)
set.seed(123)
train.index=sample(1:nrow(data),0.8*nrow(data),replace=FALSE)
train= data[train.index,];test= data[-train.index,]
table(train$Revenue)



#平衡資料----
#把Y1==0 的資料壓縮
set.seed(123)
datadown <- train[train$Revenue== "0",];dim(datadown)
k.max <- 30
asw <- rep(0,30)
for(i in 2:k.max){
  asw[i] = clara(datadown,i)$silinfo$avg.width
}
k.best <- which.max(asw)
print(k.best)
plot(1:30, asw[1:30],
     type="l", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")




#Y=0----
#In "y=0" part, the optimal number of clusters is _
set.seed(12345)
clustering <- clara(datadown,k.best)
datadown_cluster <- data.frame(datadown, clustering$cluster)
head(datadown_cluster)

cluster1 <- datadown_cluster[datadown_cluster$clustering.cluster==1,][,1:18]
cluster2 <- datadown_cluster[datadown_cluster$clustering.cluster==2,][,1:18]
cluster3 <- datadown_cluster[datadown_cluster$clustering.cluster==3,][,1:18]

set.seed(123)
n1 = cluster1[sample(nrow(cluster1), 0.9*nrow(cluster1),replace = F), ]
n2 = cluster2[sample(nrow(cluster2), 0.9*nrow(cluster2),replace = F), ]
n3 = cluster3[sample(nrow(cluster3), 0.9*nrow(cluster3),replace = F), ]


only.no <- rbind(n1,n2,n3)
table(only.no$Revenue)



#Y=1----
#Clustering & Resamping the minority part in Training Data
#About "y == 1" part
set.seed(123)
dataup <- train[train$Revenue == "1",];dim(dataup)
k.max <- 30
asw <- rep(0,30)
for(i in 2:k.max){
  asw[i] = clara(dataup,i)$silinfo$avg.width
}
k.best <- which.max(asw)
print(k.best)
plot(2:30, asw[2:30],
     type="l", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

set.seed(12345)
clustering <- clara(dataup,k.best)
dataup_cluster <- data.frame(dataup, clustering$cluster)

cluster1 <- dataup_cluster[dataup_cluster$clustering.cluster==1,][,1:18]
cluster2 <- dataup_cluster[dataup_cluster$clustering.cluster==2,][,1:18]


set.seed(123)
n1 = cluster1[sample(nrow(cluster1), 5*nrow(cluster1),replace = T), ]
n2 = cluster2[sample(nrow(cluster2), 5*nrow(cluster2),replace = T), ]


train <- rbind(only.no,n1,n2)
table(train$Revenue)

###################

#5-fold Cross validation
customRF = list(type = 'Classification', library = 'randomForest', loop = NULL)
customRF$parameters = data.frame(parameter = c("mtry","ntree"), class = rep("numeric",2), label = c("mtry","ntree"))
customRF$grid = function(x,y,len=NULL, search = "grid"){}
customRF$fit = function(x,y,wts, param, lev, last, weights, classProbs,...){
  randomForest(x,y,mtry=param$mtry, ntree=param$ntree,...)
}
customRF$predict = function(modelFit,newdata,preProc=NULL, submodels=NULL)
  predict(modelFit,newdata)
customRF$prob = function(modelFit,newdata,preProc=NULL, submodels=NULL)
  predict(modelFit,newdata, type="prob")
customRF$sort = function(x) x[order(x[,1]),]
customRF$levels = function(x) x$classes

#Set metric and control
control<-trainControl(method="cv",number=5, search='grid')
metric<-"Kappa" 
tunegrid = expand.grid(mtry=c(5),ntree=3500)
set.seed(123)
rf.mod3 = train(as.factor(Revenue)~., data = train, method = customRF, tuneGrid=tunegrid, 
                trControl = control, metric = metric)

### Confusion Matrix
predTst.rf3 <- predict(rf.mod3, newdata = test, type='prob')[,2]
optCutOff <- optimalCutoff(actuals = test$Revenue , predictedScores = predTst.rf3)[1]
predFac3 = cut(predTst.rf3, breaks=c(-Inf, 0.5, Inf), labels=c("0", "1"))

confusionMatrix(table(predFac3, test$Revenue))
