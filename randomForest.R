
setwd("~/PycharmProjects/titanic")


library(caret)
library(randomForest)
library(MLmetrics)



normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}


# read data.frame with target and paste it to explanatory variables data.frame

t <- read.csv("train.csv")
target <- t$Survived

colnames(t)
exp<- t[,c(3,5:8,10:12)]
colnames(exp)

x<- data.frame(target,exp)
x<-x[sapply(x, function(x) !any(is.na(x)))] 

str(x)

#tenemos tres variables con factores, vamos a cambiarlas a numericas

x$Sex<-ifelse(x$Sex=="female",0,1)
CABIN<- lapply(x$Cabin, substring, 1, 1)
x$Cabin<-unlist(CABIN)
table(x$Embarked)
empty <- x$Embarked == ""
x$Embarked[empty] <- "S"
x$Embarked<-ifelse(x$Embarked=="C",1,ifelse(x$Embarked=="Q",2,3))
x$Cabin<-as.factor(x$Cabin)
table(x$Cabin)
x$Cabin<-ifelse(x$Cabin=="A",1,ifelse(x$Cabin=="B",
                                      2,ifelse(x$Cabin=="C",3,
                                               ifelse(x$Cabin=="D",
                                                      4,ifelse(x$Cabin=="E",
                                                               5,ifelse(x$Cabin=="F",
                                                                        6,ifelse(x$Cabin=="G",
                                                                                 7,ifelse(x$Cabin=="T",8,9))))))))
                                                               

                                                                     
str(x)


x<-lapply(x, normalize)

x<-as.data.frame(x)

set.seed(7)

rand <- sample(1:nrow(x), nrow(x))

# train is our training sample.
train = x[rand[1:600], ]

# Create a holdout set for evaluating model performance.
# Note: cross-validation is even better than a single holdout sample.

cv = x[601:nrow(x), ]

# Review the outcome variable distribution.CLASIFICACIOON
table(Y_train, useNA = "ifany")

# Set the seed for reproducibility.
set.seed(1)

RF<-randomForest(as.factor(train$target)~.,
                 data = train[,-1],importance=TRUE,proximity=T,ntree=500)

p<-predict(RF,cv[,-1])

Accuracy(p,cv$target)

###########################

set.seed(1)

RF<-svm(as.factor(train$target)~.,data = train[,-1],
        scale = TRUE,kernel="polynomial",type="C-classification",degree=3)

p<-predict(RF,cv[,-1])

Accuracy(p,cv$target)


#################################################

model <- keras_model_sequential()

model %>%
  layer_dense(units = ncol(train[,-1]), activation = 'relu',
              input_shape = ncol(train[,-1]), ) %>%
  layer_dropout(rate = 0.99) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dropout(rate = 0.95) %>%
  layer_dense(units = 1, activation = 'sigmoid')

history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr=0.001),
  metrics = c('accuracy')
)

model %>% fit(
  as.matrix(train[,-1],dimname=NULL), train$target,
  epochs =5,
  batch_size = 20,
  validation_split = 0.0
)

#####################################################

