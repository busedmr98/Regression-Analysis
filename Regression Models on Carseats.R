                       #DSM 5007 FINAL
library(ISLR)
library(car)
library(psych)
library(corrplot)
library(olsrr)
library(Metrics)
library(ModelMetrics)
library(truncgof)
library(nortest)
library(lmtest)
library(caret)
library(ModelMetrics)
library(dplyr)
library(tree)
library(randomForest)

data<-Carseats
anyNA(data)
?Carseats
sapply(data,class)# 3 kategorik bagimsiz degisken 7 numeric bagimsiz degisken
summary(data)
head(data)
str(data)

###Test-Train###
smp_size <- floor(0.70 * nrow(data)) # Verilerin 0.7'si egitim 0.3'u test verisi olarak ayrildi
set.seed(2021900143) 
train_ind <- sample(nrow(data), size = smp_size, replace = FALSE)
train <- data[train_ind, ]
test <- data[-train_ind, ]

# DOGRUSAL REGRESYON MODELI

model_lm_full<- lm(Sales~CompPrice+ Income+ Advertising +Population 
                +Price +ShelveLoc +Age+ Education +Urban  +US,data=train)

summary(model_lm_full)
vif(model_lm_full) #Coklu Dogrusal Baglanti Sorunu Yoktur.

#Alternatif Modeller

a1<-ols_step_all_possible(model_lm_full)
a1
plot(a1)
a1[c(1013,968,969,848,638),]
a1[c(638),]

a2<-ols_step_both_p(model_lm_full,pent = 0.05,prem=0.1)
plot(a2) 
a2

a3<-ols_step_backward_p(model_lm_full) #modelden cikanlari soyler
a3

a4<-ols_step_forward_p(model_lm_full) 

#Uygulanan algoritmalar sonucu secilen 1013,968,969,848 ve 638. modellerin test verisi uzerindeki sonuclarina bakilmistir

model_lm_1<-lm(Sales~CompPrice+ Income+ Advertising +Price +ShelveLoc +Age+ Education +Urban+US,data=train)
summary(model_lm_1) #Education ve Urban degislkenleri anlamsiz
qf(.95,9,269) #model gecerli

model_lm_2<-lm(Sales~CompPrice+ Income+ Advertising +Price +ShelveLoc +Age+ Education+US,data=train)
summary(model_lm_2) #Education  degiskeni anlamsiz
qf(.95,8,270) #model gecerli

model_lm_3<-lm(Sales~CompPrice+ Income+ Advertising +Price +ShelveLoc +Age +Urban+US,data=train)
summary(model_lm_3) # Urban degiskeni anlamsiz
qf(.95,9,270) #model gecerli

model_lm_4<-lm(Sales~CompPrice+ Income+ Advertising +Price +ShelveLoc +Age+US,data=train)
summary(model_lm_4)#Degiskenler  anlamli
qf(.95,8,271) #model gecerli

model_lm_5<-lm(Sales~CompPrice+ Income+ Advertising +Price +ShelveLoc +Age,data=train)
summary(model_lm_5) #Tum degiskenler anlamli. Model gecerli

#Model karsilastirmasi icin modellerin test verisindeki performanslarina bakilir
predictions1=predict(model_lm_1,test)
predictions2=predict(model_lm_2,test)
predictions3=predict(model_lm_3,test)
predictions4=predict(model_lm_4,test)
predictions5=predict(model_lm_5,test)

RMSE1 = RMSE(predictions1, test$Sales) 
RMSE2 = RMSE(predictions2, test$Sales)
RMSE3 = RMSE(predictions3, test$Sales)
RMSE4=RMSE(predictions4,test$Sales)
RMSE5=RMSE(predictions5,test$Sales)#en kucuk deger
cbind(RMSE1,RMSE2,RMSE3,RMSE4,RMSE5)

#mean absolute error hesaplamak icin
mae1=mae(predictions1, test$Sales)
mae2=mae(predictions2, test$Sales) 
mae3=mae(predictions3, test$Sales)
mae4=mae(predictions4,test$Sales)
mae5=mae(predictions5,test$Sales)# en kucuk deger
cbind(mae1,mae2,mae3,mae4,mae5)

summary(model_lm_5)
vif(model_lm_5) #Coklu baglanti sorunu yok
#model_lm_5 (653.model) en iyi model secilmistir.

#Modelin iyilesmesine yonelik calismalar
par(mfrow=c(2,2))
plot(model_lm_5)
hist(model_lm_5$residuals,freq=F)
lines(density(model_lm_5$residuals),col="red")

## Normallik Testleri

v.test(model_lm_5$residuals, "pnorm",fit=list(mean=0,sd=sd(model_lm_5$residuals)))
cvm.test(model_lm_5$residuals)
ks.test(model_lm_5$residuals, "pnorm",fit=list(mean=0,sd=sd(model_lm_5$residuals)))
shapiro.test(model_lm_5$residuals)

#Ho hipotezi reddedilemedi dagılim normaldir. p>0.05

##Breusch-Pagan Test####

bptest(model_lm_5) # p>0.05 Ho reddedilemedi.Varyans sabittir

#Aykiri deger tespiti
hat <- hatvalues(model_lm_5)
res<- model_lm_5$residuals

length(which(hat>2*mean(hat)))  #6 kaldirac noktasi

mse=sum(res^2)/model_lm_5$df

st.res=res/(mse^0.5)
stud.res=res/((mse*(1-hat))^0.5)

sum(st.res-stud.res)

plot(hat,st.res,xlab = "Hat Values", ylab = "Standardized Residuals")
abline(h=c(-3,3),v=2*mean(hat))


hatvalues(model_lm_5)
hatvalues(model_lm_5)>2*mean(hatvalues(model_lm_5))
which(hatvalues(model_lm_5)>2*mean(hatvalues(model_lm_5))) #6 kaldirac degeri


par(mfrow=c(1,1))
st.res=model_lm_5$residuals/sd(model_lm_5$residuals)
plot(hatvalues(model_lm_5),st.res)
abline(h=c(-2,2),v=2*mean(hatvalues(model_lm_5)))


train[c(which(st.res<(-3)),
        which(st.res>(3))),]

c(which(st.res<(-3)),which(st.res>(3)))
length(c(which(st.res<(-3)),which(st.res>(3))))

ols_plot_cooksd_chart(model_lm_5)
summary(model_lm_5)
clean_train<-train[-c(101,358),]
summary(model_lm_5)
model_lm_cl<-lm(formula = Sales ~ CompPrice + Income + Advertising + Price + 
                  ShelveLoc + Age, data =clean_train)
summary(model_lm_cl)
vif(model_lm_cl)

# REGRESYON AGACI MODELİ
hist(data$Sales,freq=F,main="Histogram(Sales)")
lines(density(data$Sales),col="red")
ad.test(data$Sales) #normal dagılmis

model_tree_full <- tree(Sales~CompPrice+ Income+ Advertising +Population 
                        +Price +ShelveLoc +Age+ Education +Urban  +US,data =train)

plot(model_tree_full) #shelveLoc Kok dugum
text(model_tree_full)
cv_data_full <- cv.tree(model_tree_full)
cv_data_full
summary(model_tree_full)
data.train <- data[train_ind,"Sales"]
data.test<-data[-train_ind,"Sales"]
plot(cv_data_full$size,cv_data_full$dev,type='b') #17 tane terminal noktasi

## Terminal noktasi 5
prune_data_5 <- prune.tree(model_tree_full,best=5)
summary(prune_data_5) #deviance artti #3.95
plot(prune_data_5)
text(prune_data_5,pretty=0)

##Terminal noktasi 8
prune_data_8 <- prune.tree(model_tree_full,best=8)
summary(prune_data_8) #3.3
plot(prune_data_8)
text(prune_data_8,pretty=0)

#Terminal noktasi 9
prune_data_9 <- prune.tree(model_tree_full,best=9)
summary(prune_data_9) #3.1
plot(prune_data_9)
text(prune_data_9,pretty=0)

#Terminal noktasi 11
prune_data_11 <- prune.tree(model_tree_full,best=11)
summary(prune_data_11) # 2.88
plot(prune_data_11)
text(prune_data_11,pretty=0)

##test verisi performansları

yhat<-predict(model_tree_full,newdata=data[-train_ind,]) 
summary(as.factor(yhat))

plot(yhat,data.test) #tahmin degerleri ile gercek degerler arasindaki iliski
abline(0,1)
sqrt(mean((data.test-yhat)^2)) #2.26



##prune_data ile  test verisinde
yhat5<-predict(prune_data_5,newdata=data[-train_ind,])
data.test_5<-data[-train_ind,"Sales"]
plot(yhat5,data.test_5) 
abline(0,1)
sqrt(mean((data.test_5-yhat5)^2)) #2.34
summary(as.factor(yhat5))

yhat8<-predict(prune_data_8,newdata=data[-train_ind,])
data.test_8<-data[-train_ind,"Sales"]
plot(yhat8,data.test_8) 
abline(0,1)
sqrt(mean((data.test_8-yhat8)^2)) #2.28
summary(as.factor(yhat8)) 

yhat9<-predict(prune_data_9,newdata=data[-train_ind,])
data.test_9<-data[-train_ind,"Sales"]
plot(yhat9,data.test_9) 
abline(0,1)
sqrt(mean((data.test_9-yhat9)^2)) #2.298
summary(as.factor(yhat9))

yhat11<-predict(prune_data_11,newdata=data[-train_ind,])
data.test_11<-data[-train_ind,"Sales"]
plot(yhat11,data.test_11) 
abline(0,1)
sqrt(mean((data.test_11-yhat11)^2)) #2.293
summary(as.factor(yhat11))


#Bagging Regresyon Ağacı

set.seed(1)

bag.data<-randomForest(Sales~.,data=train,mtry=10,importance=TRUE)
yhat.bag<-predict(bag.data,newdata=data[train_ind,]) #train verisi
plot(yhat.bag,data.train) 
abline(0,1)
sqrt(mean((data.train-yhat.bag)^2)) #train verisi #0.64

yhat.bag.test<-predict(bag.data,newdata=data[-train_ind,]) #test verisi
plot(yhat.bag.test,data.test) 
abline(0,1)
sqrt(mean((data.test-yhat.bag.test)^2)) #test verisi #1.74

#ntree  değişitirilirse
bag.data_1<-randomForest(Sales~.,data=train,mtry=10,importance=TRUE, ntree=100)
yhat.bag.test_1<-predict(bag.data_1,newdata=data[-train_ind,])
plot(yhat.bag.test_1,data.test) 
abline(0,1)
sqrt(mean((data.test-yhat.bag.test_1)^2)) #1.80 

bag.data$importance #Onemli degiskenler ShelveLoc, Price,CompPrice
varImpPlot(bag.data)
#bagging de hep benzer agaclar elde ediliyor. Ve agacları göremiyoruz.


##Random Forest
rf.data<-randomForest(Sales~.,data=train,mtry=3,importance=TRUE)
yhat.bag_rf<-predict(rf.data,newdata=data[train_ind,]) #train verisi
plot(yhat.bag_rf,data.train) 
abline(0,1)
sqrt(mean((data.train-yhat.bag_rf)^2)) 

yhat.bag_rf_test<-predict(rf.data,newdata=data[-train_ind,]) #test verisi
plot(yhat.bag_rf_test,data.test) 
abline(0,1)
sqrt(mean((data.test-yhat.bag_rf_test)^2)) #test verisi

rf.data$importance
varImpPlot(rf.data)
#birbirinden bagimsiz agaclar elde edilir

##Model Karşılaştırması##
par(mfcol=c(2,2))
plot(yhat.bag,data.train,main="Bagging") 
abline(0,1,col="red")
plot(yhat.bag_rf_test,data.test,main="Rassal Ormanlar") 
abline(0,1,col="blue")
plot(yhat8,data.test_8,main="Regresyon Agacı") 
abline(0,1)


 
RMSE5 #Doğrusal Regresyon
sqrt(mean((data.test_8-yhat8)^2)) #Regresyon Agacı
sqrt(mean((data.test-yhat.bag.test)^2)) #Bagging
sqrt(mean((data.test-yhat.bag_rf_test)^2)) #Random Forest
