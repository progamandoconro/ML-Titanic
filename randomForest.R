
  setwd("~/PycharmProjects/titanic")
  
  
  library(caret)
  library(randomForest)
  library(MLmetrics)
  library(mlbench)
  library(zoo)
  library(ROSE)
  library(reticulate)
  library(MASS)
  library(qdap)
  library(stringr)
  library(e1071)
  
  
  
  normalize <- function(x) { 
    return((x - min(x)) / (max(x) - min(x)))
  }
  
  
  # read data.frame with target and paste it to explanatory variables data.frame
  
  t <- read.csv("train.csv")
  target <- t$Survived
  
  colnames(t)
  exp<- t[,c(3:8,10:12)]
  colnames(exp)
  
  x<- data.frame(target,exp)
  #x<-x[sapply(x, function(x) !any(is.na(x)))] 
  #vamos a atender el asunto de los NA 
  
  str(x)
  
  #tenemos tres variables con factores, vamos a cambiarlas a numericas
  
  x$Sex<-ifelse(x$Sex=="female",0,1)
  
  
  
  
  table(x$Embarked)
  empty <- x$Embarked == ""
  x$Embarked[empty] <- "S"
  x$Embarked<-ifelse(x$Embarked=="C",1,ifelse(x$Embarked=="Q",2,3))
  
  
  CABIN<- lapply(x$Cabin, substring, 1, 1)
  x$Cabin<-unlist(CABIN)
  table(x$Cabin)
  x$Cabin<-as.factor(x$Cabin)
  table(x$Cabin)
  x$Cabin<-ifelse(x$Cabin=="A",
                  1,ifelse(x$Cabin=="B",
                           2,ifelse(x$Cabin=="C",
                                    3,ifelse(x$Cabin=="D",
                                             4,ifelse(x$Cabin=="E",
                                                      5,ifelse(x$Cabin=="F",
                                                               6,ifelse(x$Cabin=="G",
                                                                        7,ifelse(x$Cabin=="T",8,0))))))))
  
  #table(x$Cabin)
  
  #x$Age==ifelse(x$Age<15,1,ifelse(x$Age<30,2,ifelse(x$Age<45,3,4)))
  
  #x$SibSp2=ifelse(x$SibSp+x$Parch<1,1,ifelse(x$SibSp+x$Parch<4,2,ifelse(x$SibSp+x$Parch<6,3,4)))
  
  
  
  nombres=t$Name %>%
    str_replace_all(c(".*Mrs.*" = "MRS.",".*Mr.*" = "MR.",  ".*Dr.*" = "DR.",".*Capt.*" = "CAPT",".*Col.*" = "COL",
                      ".*Miss.*" = "MISS", ".*Rev.*" = "REV", ".*Master.*" ="MASTER",".*Major.*" = "MAJOR", 
                      ".*Countess.*" = "COUNTESS", ".*Jonkheer.*" = "JONKHEER", ".*Don.*" = "DON", ".*Mlle.*" = "MLLE", 
                      ".*Ms.*" = "MS", ".*Mme.*" = "MME"))
  
  table(nombres)
  
  
  x$Fare<-replace(x$Fare,x$Fare>100,100)
  
  x$Name=ifelse(nombres=="MR.",1,ifelse(nombres=="MRS.",2,ifelse(nombres=="MISS",3, 4)))
  
  
  x<-x
  colnames(x)
  
  
  x<-na.aggregate(x)
  
  str(x)
  
  x<-lapply(x, normalize)
  
  x<-as.data.frame(x)
  
  
  
  #TT<- lapply(t$Ticket, substring, 1, 1)
  #x$Ticket<-as.factor(unlist(TT))
  pos=x[x$target==1,];neg=x[x$target==0,]
  set.seed(7)
  epos=sample(1:nrow(pos),floor(0.7*nrow(pos)))
  post=pos[epos,]
  poscv=pos[-epos,]
  
  set.seed(7)
  eneg=sample(1:nrow(neg),floor(0.7*nrow(neg)))
  negt=neg[eneg,]
  negcv=neg[-eneg,]
  train=rbind(post,negt); cv=rbind(poscv,negcv)
  
  
  
  
  
  
  
  
  
  #py_config
  #use_python("/home/ro/anaconda3/envs/r-tensorflow/bin/python")
  
  #py_config
  #use_python("/home/ro/anaconda3/envs/r-tensorflow/bin/python")
  
  
  k = import("sklearn.ensemble")
  
  rf = k$RandomForestClassifier()
  
  
  apply(train, 2, function(x) any(is.na(x)))
  
  
  
  rf$fit (train[,-1], train[,1])
  
  k$RandomForestClassifier(bootstrap=TRUE, class_weight=NULL, criterion='gini',
                           max_depth=NULL, max_features='auto', max_leaf_nodes=NULL,
                           min_impurity_split=1e-07, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=10, n_jobs=1, oob_score=FALSE, random_state=NULL,
                           verbose=0, warm_start=FALSE)
  y_pred = rf$predict(cv[,-1])
  

  
  #############
  
  
  k2 = import("sklearn.ensemble")
  
  rf2 = k2$RandomForestClassifier()
  
  
  rf2$fit (train[,-1], train[,1])
  
  k2$RandomForestClassifier(bootstrap=TRUE, class_weight=NULL, criterion='gini',
                            max_depth=NULL, max_features='auto', max_leaf_nodes=NULL,
                            min_impurity_split=1e-07, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=10, n_jobs=1, oob_score=FALSE, random_state=NULL,
                            verbose=0, warm_start=FALSE)
  y_pred = rf2$predict(cv[,-1])
  #######
  k3 = import("sklearn.ensemble")
  
  rf3 = k3$RandomForestClassifier()
  
  
  rf3$fit (train[,-1], train[,1])
  
  k3$RandomForestClassifier(bootstrap=TRUE, class_weight=NULL, criterion='gini',
                            max_depth=NULL, max_features='auto', max_leaf_nodes=NULL,
                            min_impurity_split=1e-07, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=10, n_jobs=1, oob_score=FALSE, random_state=NULL,
                            verbose=0, warm_start=FALSE)
  y_pred = rf3$predict(cv[,-1])
  
  set.seed(4)
  k6<- svm(as.factor(train[,1]) ~ . ,data = train[,-1],
           scale = TRUE,type="C-classification",kernel="polynomial",
           gamma=0.05,cost=4)
  
  
  
  
  
  
  test<-read.csv("test.csv")
  colnames(test)
  colnames(x)
  exp<- test[,c(-1,-8)]
  colnames(exp)
  
  
  
  
  x<- exp
  
  str(x)
  
  #tenemos tres variables con factores, vamos a cambiarlas a numericas
  
  x$Sex<-ifelse(x$Sex=="female",0,1)
  
  
  
  
  table(x$Embarked)
  empty <- x$Embarked == ""
  x$Embarked[empty] <- "S"
  x$Embarked<-ifelse(x$Embarked=="C",1,ifelse(x$Embarked=="Q",2,3))
  
  
  CABIN<- lapply(x$Cabin, substring, 1, 1)
  x$Cabin<-unlist(CABIN)
  table(x$Cabin)
  x$Cabin<-as.factor(x$Cabin)
  table(x$Cabin)
  x$Cabin<-ifelse(x$Cabin=="A",
                  1,ifelse(x$Cabin=="B",
                           2,ifelse(x$Cabin=="C",
                                    3,ifelse(x$Cabin=="D",
                                             4,ifelse(x$Cabin=="E",
                                                      5,ifelse(x$Cabin=="F",
                                                               6,ifelse(x$Cabin=="G",
                                                                        7,ifelse(x$Cabin=="T",8,0))))))))
  
  table(x$Cabin)
  
  #x$Age==ifelse(x$Age<15,1,ifelse(x$Age<30,2,ifelse(x$Age<45,3,4)))
  
  #x$SibSp2=ifelse(x$SibSp+x$Parch<1,1,ifelse(x$SibSp+x$Parch<4,2,ifelse(x$SibSp+x$Parch<6,3,4)))
  
  
  
  nombres=test$Name %>%
    str_replace_all(c(".*Mrs.*" = "MRS.",".*Mr.*" = "MR.",  ".*Dr.*" = "DR.",".*Capt.*" = "CAPT",".*Col.*" = "COL",
                      ".*Miss.*" = "MISS", ".*Rev.*" = "REV", ".*Master.*" ="MASTER",".*Major.*" = "MAJOR", 
                      ".*Countess.*" = "COUNTESS", ".*Jonkheer.*" = "JONKHEER", ".*Don.*" = "DON", ".*Mlle.*" = "MLLE", 
                      ".*Ms.*" = "MS", ".*Mme.*" = "MME"))
  
  table(nombres)
  
  
  
  
  x$Name=ifelse(nombres=="MR.",1,ifelse(nombres=="MRS.",2,ifelse(nombres=="MISS",3, 4)))
  
  
  colnames(x)
  
  
  x<-na.aggregate(x)
  
  str(x)
  
  x<-lapply(x, normalize)
  
  x<-as.data.frame(x)
  
  
  xtest <- x
  
  Survived1<-rf$predict(xtest)
  Survived2<-rf2$predict(xtest)
  Survived3<-rf3$predict(xtest)
  Survived4<-predict(k6,xtest)
  Survived<- (Survived1+Survived2+Survived3)/3
  Survived<-ifelse(Survived<0.5,0,1)
  
  write.csv(data.frame
            (test$PassengerId,Survived,Survived1,Survived2,Survived3,Survived4,xtest),
            "mispredicciones.csv",row.names = FALSE)


