# Titanic
library(randomForest)
library(rpart)
library(ggplot2)
library(tidyverse)
library(scales)
library(grid)
library(caret)
library(e1071)


#データの読み込み
train<-read.csv("train.csv")
test<-read.csv("test.csv")

train$data <- "train"
test$data  <- "test"
test$Survived <- NA
full <- rbind(train, test)
head(full)

#データ理解
summary(full)
str(full)
dim(full)
lapply(full, function(x) length(unique(x))) #カラムごとの値の数

##性別
hist(full$Age)
k = 20 #ヒストグラムのbin数
hist(full[full$Sex=='male', 'Age'],col ="#ff00ff40" , border = "#ff00ff", breaks = k,main ="Age by gender",xlab ="Age",cex.main=2)
hist(full[full$Sex=='female', 'Age'],col ="#0000ff40" , border = "#0000ff", breaks = k,add=T )
summary(full[full$Sex=='male', 'Age'])
summary(full[full$Sex=='female', 'Age'])


##############################
# 欠損値処理
##############################
full %>% summarize_all(funs(sum(is.na(.))))

# Age(性別ごとに予測して埋める)
# モデリング(決定木)
tmp_data<-train[is.na(full$Age)==F, ]
age_model<-rpart(Age~Pclass+Sex+SibSp+Parch+Fare+Embarked , data=tmp_data)
summary(age_model)
# 予測
pred_age<-predict(age_model, full)
full$Age_NA<-as.numeric(is.na(full$Age))
full$Age[is.na(full$Age)]<-pred_age[is.na(full$Age)]

# Fare
full$Fare[is.na(full$Fare)]<-mean(full$Fare, na.rm = T)

##############################
# 特徴量作成
##############################

#Nameからtitleを抜き出す(,と.の間の文字列)
full$Name<-as.character(full$Name)
parse_names<-strsplit(full$Name, "[,\\.]") 
name_title<-function(parse_name){
  tmp_ <- unlist(parse_name)
  title <- gsub(" ", "", tmp_[2], fixed = TRUE) 
  return(title)
}
titles<-lapply(parse_names, FUN=name_title)
titles<-unlist(titles)
full$Title<-titles
table(titles)

# SibSpを家族(Family)に
full$Family <-full$SibSp + full$Parch + 1 
full$Family[full$Family >= 5] <- 'Big' 
full$Family[full$Family < 5 & full$Family >= 2] <- 'Small' 
full$Family[full$Family == 1] <- 'Single' 
full$Family=as.factor(full$Family)

# Ticketを団体(Party)に
party_cnt <- rep(0, nrow(full))
uniq_ticket<-unique(full$Ticket)
for (i in 1:length(uniq_ticket)) {
  tmp_ticket <- uniq_ticket[i]
  party_idx <- which(full$Ticket == tmp_ticket)
  for (j in 1:length(party_idx)) {
    party_cnt[party_idx[j]] <- length(party_idx)
  }
}
full$Party <- party_cnt
full$Party[full$Party >= 5]   <- 'Big'
full$Party[full$Party < 5 & full$Party>= 2]   <- 'Small'
full$Party[full$Party == 1]   <- 'Single'

str(full)

#########################################
# 前処理
#########################################
feauter<-full[,c("Pclass","Title","Sex","Age","Fare","SibSp","Parch",
                 "Embarked","Family","Party","Survived","data","PassengerId")]
uniq_title<-unique(c(feauter$Title))
feauter$Title<-factor(feauter$Title, levels = uniq_title)
feauter$Title<-as.numeric(feauter$Title)
feauter$Sex<-as.numeric(feauter$Sex)
feauter$Embarked<-as.numeric(feauter$Embarked)
feauter$Family<-as.numeric(feauter$Family)
feauter$Party=as.factor(feauter$Party)
feauter$Party<-as.numeric(feauter$Party)
feauter$Survived=as.factor(feauter$Survived)
str(feauter)

train <- feauter[feauter$data=="train",]
test <- feauter[feauter$data=="test",]

set.seed(123)
idx=createDataPartition(train$Survived,times=1,p=0.7,list=FALSE)
X_train=train[idx,]
y_train=train[-idx,]

#########################################
# モデリング
#########################################
#Random Forest
rf<-randomForest(Survived~.-data-PassengerId, data=X_train, trees=1000, proximity=T, importance=T)
rf
plot(rf)
varImpPlot(rf)
importance(rf)

pred_rf=predict(rf,newdata = y_train)
confusionMatrix(pred_rf,y_train$Survived)

rf2<-randomForest(Survived~.-data-PassengerId-Embarked-Parch-SibSp, 
                  data=X_train, trees=1000, proximity=T, importance=T)
rf2
plot(rf2)
varImpPlot(rf2)
importance(rf2)

pred_rf=predict(rf2,newdata = y_train)
confusionMatrix(pred_rf,y_train$Survived)

#クロスバリデーション
myCrossValidation<-function(k, data){
  kfolds<-split(c(1:dim(data)[1]),f = 1:k)
  bucket_names<-names(kfolds)
  accuracy_list<-NULL
  for(i in 1:length(bucket_names)){
    test_idx <- array(unlist(kfolds[i]))
    train_subset<- data[-test_idx, ]
    test_subset <- data[test_idx, ]
    mymodel<-randomForest(Survived~.-data-PassengerId-Embarked-Parch-SibSp, 
                          data=train_subset, trees=1000, maxnodes=10, proximity=T, importance=T)
    ypred_flag<-predict(mymodel, test_subset)
    conf_mat<-table(ypred_flag, test_subset$Survived)
    accuracy<-(conf_mat[1]+conf_mat[4])/dim(test_subset)[1]
    accuracy_list<-c(accuracy_list, accuracy)
  }
  print(paste("Accuracy Mean: ", mean(accuracy_list)))
  print(paste("Accuracy SD:", sd(accuracy_list)))
}
myCrossValidation(k=5, data=train)


#予測してSubmission
mymodel<-randomForest(Survived~.-data-PassengerId-Embarked-Parch-SibSp, 
                      data=train, trees=1000, maxnodes=10, proximity=T, importance=T)
mymodel
varImpPlot(mymodel)
ypred_flag<-predict(mymodel, test)

solution <- data.frame(Survived = ypred_flag, PassengerID = test$PassengerId)
write.csv(solution, file = 'titanic.csv', row.names = F)
