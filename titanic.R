# Titanic
library(randomForest)
library(rpart)
library(caret)
library(doParallel)

#並列化演算
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

#データの読み込み
train<-read.csv("train.csv")
test<-read.csv("test.csv")

head(train)

#データ理解
summary(train)

##性別
hist(train$Age)
k = 20 #ヒストグラムのbin数
hist(train[train$Sex=='male', 'Age'],col ="#ff00ff40" , border = "#ff00ff", breaks = k,main ="Age by gender",xlab ="Age",cex.main=2)
hist(train[train$Sex=='female', 'Age'],col ="#0000ff40" , border = "#0000ff", breaks = k,add=T )
summary(train[train$Sex=='male', 'Age'])
summary(train[train$Sex=='female', 'Age'])


##############################
# Age 欠損値処理(性別ごとに予測して埋める)
##############################

# train
tmp_data<-train[is.na(train$Age)==F, ]
age_model<-rpart(Age~Pclass+Sex+SibSp+Parch+Fare+Embarked , data=tmp_data)
summary(age_model)

pred_age<-predict(age_model, train)
train$Age_NA<-as.numeric(is.na(train$Age))
train$Age[is.na(train$Age)]<-pred_age[is.na(train$Age)]
#train$Age[is.na(train$Age)]<-median(train$Age, na.rm = T)

# test
age_pred<-predict(age_model, test)
plot(test$Age, age_pred) #予測した年齢と実際の年齢をプロット
test$Age_NA<-as.numeric(is.na(test$Age))
test$Age[is.na(test$Age)]<-age_pred[is.na(test$Age)]
#test$Age[is.na(test$Age)]<-median(test$Age, na.rm = T)

##############################
# 名前から特徴量を抽出
##############################

#Nameの特徴量抽出
train$Name<-as.character(train$Name)
test$Name<-as.character(test$Name)

## 欠損ないか？
sum(is.na(train$Name))
sum(is.na(test$Name))

#真ん中ののtitleを抜き出す
parse_names_train<-strsplit(train$Name, "[,\\.]") #,と.でsplit
name_title<-function(parse_name){
  tmp_ <- unlist(parse_name)
  title <- gsub(" ", "", tmp_[2], fixed = TRUE) 
  return(title)
}

#train
titles<-lapply(parse_names_train, FUN=name_title)
titles<-unlist(titles)
train$title<-titles

#test
parse_names_test<-strsplit(test$Name, "[,\\.]")
titles<-lapply(parse_names_test, FUN=name_title)
titles<-unlist(titles)
test$title<-titles

#########################################
# その他の特徴量(ランダムフォレスト前提)
#########################################

#Titleを数値にする
uniq_title<-unique(c(train$title, test$title))
train$title<-factor(train$title, levels = uniq_title)
test$title<-factor(test$title, levels = uniq_title)
train$title<-as.numeric(train$title)
test$title<-as.numeric(test$title)

#Cabin(最初の１文字を抜き出す)
train$Cabin<-as.character(train$Cabin)
firts_cabin<-lapply(train$Cabin, FUN=function(x){return(substr(x,1,1))})
train$firts_cabin<-unlist(firts_cabin)

firts_cabin<-lapply(test$Cabin, FUN=function(x){return(substr(x,1,1))})
test$firts_cabin<-unlist(firts_cabin)

uniq_cabin<-unique(c(train$firts_cabin, test$firts_cabin))
train$firts_cabin<-factor(train$firts_cabin, levels = uniq_cabin)
test$firts_cabin<-factor(test$firts_cabin, levels = uniq_cabin)
train$firts_cabin<-as.numeric(train$firts_cabin)
test$firts_cabin<-as.numeric(test$firts_cabin)

#Fareの欠損値
sum(is.na(test$Fare))
test$Fare[is.na(test$Fare)]<-mean(test$Fare, na.rm = T)

#Embarked
train$Embarked<-as.numeric(train$Embarked)
test$Embarked<-as.numeric(test$Embarked)

#Sex
train$Sex<-as.numeric(train$Sex)
test$Sex<-as.numeric(test$Sex)

summary(test)
#########################################
# モデリング
#########################################
train$Survived<-as.factor(train$Survived)

#Random Forest
rf<-randomForest(Survived~
                   Pclass+
                   Sex+
                   Age+
                   Fare+
                   title+
                   firts_cabin+
                   Age_NA+
                   SibSp+
                   Parch+
                   Embarked, data=train, trees=1000, proximity=T, importance=T)
rf
plot(rf)
varImpPlot(rf)
predict(rf, newdata = test, type="prob")


#ランダムフォレスト(caret)
set.seed(0)
modelRF <- train(Survived~
                   Pclass+
                   Sex+
                   Age+
                   Fare+
                   title+
                   firts_cabin+
                   Age_NA+
                   SibSp+
                   Parch+
                   Embarked,
                 data = train, 
                 method = "rf", 
                 tuneLength = 4,
                 trControl = trainControl(method = "cv")
)

predRF <- predict(modelRF, test)
step(predRF)
AIC(predRF)
summary(train)

#クロスバリデーション

myCrossValidation<-function(k, data){
  kfolds<-split(c(1:dim(data)[1]),f = 1:k)
  bucket_names<-names(kfolds)
  accuracy_list<-NULL
  for(i in 1:length(bucket_names)){
    test_idx <- array(unlist(kfolds[i]))
    train_subset<- data[-test_idx, ]
    test_subset <- data[test_idx, ]
    mymodel<-randomForest(Survived~
                            Pclass+
                            Sex+
                            Age+
                            Fare+
                            title+
                            firts_cabin+
                            Age_NA+
                            SibSp+
                            Parch+
                            Embarked, data=train_subset, trees=1000, maxnodes=3, proximity=T, importance=T)
    ypred_flag<-predict(mymodel, test_subset)
    #ypred<-predict(mymodel, train_mat[test_idx, ], type="prob")
    #ypred_flag<-ifelse(ypred >= 0.3838384, 1, 0)
    conf_mat<-table(ypred_flag, test_subset$Survived)
    accuracy<-(conf_mat[1]+conf_mat[4])/dim(test_subset)[1]
    accuracy_list<-c(accuracy_list, accuracy)
  }
  print(paste("Mean: ", mean(accuracy_list)))
  print(paste("SD:", sd(accuracy_list)))
}

myCrossValidation(k=5, data=train)


#予測してSubmission
mymodel<-randomForest(Survived~
                        Pclass+
                        Sex+
                        Age+
                        Fare+
                        title+
                        firts_cabin+
                        SibSp+
                        Embarked, data=train, trees=500, maxnodes=10, proximity=T, importance=T)
mymodel
varImpPlot(mymodel)
ypred_flag<-predict(mymodel, test)
#ypred_flag<-ifelse(test_pred_proba >= 0.3838384, 1, 0)

solution <- data.frame(Survived = ypred_flag, PassengerID = test$PassengerId)
write.csv(solution, file = 'rf_model_sol4.csv', row.names = F)
