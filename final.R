library(InformationValue)
online <- read.csv( "online_shoppers_intention.csv")
str(online)
online$Revenue = as.numeric(online$Revenue)
# M = cor(online[,1:10])
# library(corrplot) # for correlation plot, install if needed
# corrplot(M, method = "number" , number.cex = .7 )

table(online$Revenue)


logit  <- glm(formula = Revenue ~.,
                  data = online ,family = binomial(link = "logit"))

summary(logit)

#select variable
logit  <- glm(formula = Revenue ~ ProductRelated_Duration + ExitRates + PageValues + OperatingSystems + Browser +
                VisitorType + Month,
              data = online ,family = binomial(link = "logit"))
summary(logit)

#VIF
library(car)
vif(logit)

#--------under sampling --------
#match it
library(MatchIt)
data_match1 <- matchit( Revenue ~ PageValues , data = online , method = "nearest" , ratio = 1)
online <- online[which(data_match1$weights==1),]
plot(data_match1 , type = "hist")

#--------------------------
# logit regression
mis <- c()
pre <- c()
recall <- c()
spe <- c()
acu <- c()
set.seed(123)
datarandom<- online[sample(nrow(online)),]
folds <- cut(seq(1,nrow(datarandom)),breaks=5,labels=FALSE)
#--------------------------------------------------------
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_data1 <- datarandom[testIndexes, ]
  train_data1 <- datarandom[-testIndexes, ]
  
  model_logit  <- glm(formula = Revenue ~ ProductRelated_Duration + ExitRates + PageValues + OperatingSystems + Browser +
                  VisitorType + Month,
                data = train_data1 ,family = binomial(link = "logit"))
  
  prob_logit <- predict(model_logit , test_data1 , type="response" )
  optCutOff <- optimalCutoff(actuals = test_data1$Revenue , predictedScores = prob_logit)[1]

  prob_logit = cut(prob_logit, breaks=c(-Inf, optCutOff, Inf), labels=c("0", "1"))
  matrix = table(prob_logit ,test_data1$Revenue)
  acu[i] <-     (matrix[1,1]+matrix[2,2])/sum(matrix)
  pre[i] <-     matrix[2,2]/(sum(matrix[2,]))
  recall[i] <-  matrix[2,2]/(sum(matrix[,2]))
  spe[i] <-     matrix[2,2]/(sum(matrix[,2]))
}
output_logit <- data.frame(
  accuracy = acu,
  precision = pre,
  sensitivity = recall,
  specificity = spe
)
avg_output_logit<- data.frame(
  accuracy = mean(acu),
  precision = mean(pre),
  sensitivity = mean(recall),
  specificity = mean(spe)
)
round(avg_output_logit,4)

plotROC(test_data1$Revenue, prob_logit)

#--------------
#RandomForest
library(randomForest)
set.seed(123)
datarandom<- online[sample(nrow(online)),]
folds <- cut(seq(1,nrow(datarandom)),breaks=5,labels=FALSE)
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_data2 <- datarandom[testIndexes, ]
  train_data2 <- datarandom[-testIndexes, ]
  
  model_rd <- randomForest( Revenue ~ ProductRelated_Duration + ExitRates + PageValues + OperatingSystems + Browser +
                             VisitorType + Month,
                           data = train_data2, n_tree = 500 )
  prob_rd <- predict(model_rd , test_data2 , type="response" )
  
  optCutOff <- optimalCutoff(actuals = test_data2$Revenue , predictedScores = prob_rd )[1]
  prob_rd = ifelse(prob_rd>optCutOff , 1 ,0)
  matrix = table(prob_rd ,test_data2$Revenue)
  acu[i] <-     (matrix[1,1]+matrix[2,2])/sum(matrix)
  pre[i] <-     matrix[2,2]/(sum(matrix[2,]))
  recall[i] <-  matrix[2,2]/(sum(matrix[,2]))
  spe[i] <-     matrix[2,2]/(sum(matrix[,2]))
}
output_rd <- data.frame(
  accuracy = acu,
  precision = pre,
  sensitivity = recall,
  specificity = spe
)
avg_output_rd <- data.frame(
  accuracy = mean(acu),
  precision = mean(pre),
  sensitivity = mean(recall),
  specificity = mean(spe)
)
round(avg_output_rd,4)

#--------------
#決策樹CART
library(rpart)
set.seed(123)
datarandom<- online[sample(nrow(online)),]
folds <- cut(seq(1,nrow(datarandom)),breaks=5,labels=FALSE)

for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_data3 <- datarandom[testIndexes, ]
  train_data3 <- datarandom[-testIndexes, ]
  model_tree <- rpart(formula =  Revenue ~ ProductRelated_Duration + ExitRates + PageValues + OperatingSystems + Browser +
                        VisitorType + Month,
                      data = train_data3, method = "class",cp = 2e-3)
  
  prob_tree <- predict(object = model_tree , newdata = test_data3, type = "class")
  matrix = table(prob_tree ,test_data3$Revenue)
  acu[i] <-     (matrix[1,1]+matrix[2,2])/sum(matrix)
  pre[i] <-     matrix[2,2]/(sum(matrix[2,]))
  recall[i] <-  matrix[2,2]/(sum(matrix[,2]))
  spe[i] <-     matrix[1,1]/(sum(matrix[,1]))
}
output_tree <- data.frame(
  accuracy = acu,
  precision = pre,
  sensitivity = recall,
  specificity = spe
)
avg_output_tree <- data.frame(
  accuracy = mean(acu),
  precision = mean(pre),
  sensitivity = mean(recall),
  specificity = mean(spe)
)
round(avg_output_tree,4)

#--------------
#Support Vector Machine(SVM)
library(e1071)
online$Revenue = as.factor(online$Revenue)
model_svm <- svm(formula =  Revenue ~ ProductRelated_Duration + ExitRates + PageValues + OperatingSystems + Browser +
                   VisitorType + Month,
                 data = online)
#-------tune parameters in SVM
library("mlbench")
tune.model = tune(svm,
                 Revenue ~ ProductRelated_Duration + ExitRates + PageValues + OperatingSystems + Browser +
                   VisitorType + Month,
                 data=online,
                 kernel="radial", # RBF kernel function
                 range=list(cost=10^(-1:2), gamma=c(.5,1,2)))


set.seed(123)
datarandom<- online[sample(nrow(online)),]
folds <- cut(seq(1,nrow(datarandom)),breaks=5,labels=FALSE)
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_data4 <- datarandom[testIndexes, ]
  train_data4 <- datarandom[-testIndexes, ]
  
  model_svm <- svm(formula =  Revenue ~ ProductRelated_Duration + ExitRates + PageValues + OperatingSystems + Browser +
                     VisitorType + Month,
                   data = train_data4)
  
  prob_svm <- predict(object = model_svm , newdata = test_data4, type = "response")
  matrix = table(prob_svm ,test_data4$Revenue)
  acu[i] <-     (matrix[1,1]+matrix[2,2])/sum(matrix)
  pre[i] <-     matrix[1,1]/(sum(matrix[1,]))
  recall[i] <-  matrix[1,1]/(sum(matrix[,1]))
  spe[i] <-     matrix[2,2]/(sum(matrix[,2]))
}
output_svm <- data.frame(
  accuracy = acu,
  precision = pre,
  sensitivity = recall,
  specificity = spe
)
avg_output_svm <- data.frame(
  accuracy = mean(acu),
  precision = mean(pre),
  sensitivity = mean(recall),
  specificity = mean(spe)
)
round(avg_output_svm,4)




library(ROCR)
pr_logit <- prediction(predictions = prob_logit, labels = test_data1$Revenue)
prf_logit <- performance(prediction.obj = pr_logit , measure = "tpr", x.measure = 'fpr')
dd_logit <- data.frame(FP = prf_logit@x.values[[1]], TP = prf_logit@y.values[[1]])

pr_rd <- prediction(predictions = prob_rd , labels = test_data2$Revenue)
prf_rd <- performance(prediction.obj = pr_rd , measure = "tpr", x.measure = 'fpr')
dd_rd <- data.frame(FP = prf_rd@x.values[[1]], TP = prf_rd@y.values[[1]])

pr_tree <- prediction(predictions = as.numeric(prob_tree) , labels = test_data3$Revenue)
prf_tree <- performance(prediction.obj = pr_tree , measure = "tpr", x.measure = 'fpr')
dd_tree <- data.frame(FP = prf_tree@x.values[[1]], TP = prf_tree@y.values[[1]])

pr_svm <- prediction(predictions = as.numeric(prob_svm) , labels = test_data4$Revenue)
prf_svm <- performance(prediction.obj = pr_svm , measure = "tpr", x.measure = 'fpr')
dd_svm <- data.frame(FP = prf_svm@x.values[[1]], TP = prf_svm@y.values[[1]])
library(ggplot2)
g <- 
  ggplot() +
  geom_line(data = dd_logit , mapping = aes(x = FP, y = TP, color = 'Logistic Regression')) +
  geom_line(data = dd_tree , mapping = aes(x = FP, y = TP, color = 'CART')) +
  geom_line(data = dd_rd , mapping = aes(x = FP, y = TP, color = 'Random Forest')) +
  geom_line(data = dd_svm,mapping = aes(x = FP, y = TP, color = 'Support Vector Machine')) +
  geom_segment(mapping = aes(x = 0, xend = 1, y = 0, yend = 1)) +
  ggtitle(label = "ROC Curve") +
  labs(x = "False Positive Rate", y = "True Positive Rate")
g +
  scale_color_manual(name = "classifier",values = c('Logistic Regression'='#E69F00', 
                                                    'CART'='#009E73',
                                                    'Random Forest'='#D55E00',
                                                    'Support Vector Machine'='#0072B2'))

