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
library(ggplot2)
library(e1071)
library(car)
library(forecast)
library(MASS)
library("devtools")
library(ggord)
library(mvnTest)

Carseats
High <- ifelse(Carseats$Sales <=mean(Carseats$Sales),"0","1 ")#Ortalamanin altindakiler 0 ustundekiler 1 degerini alir
High<-as.factor(High) 
Carseats <- data.frame(Carseats,High)
Carseats<-Carseats[c(-1)] #Sales degiskeni veriden cikarilir
summary(Carseats)
str(Carseats)

set.seed(2021900143)
train_ind <- sample (1: nrow(Carseats ), 280)
train<-Carseats[train_ind,]
test <- Carseats [-train_ind ,]
# Verilerin 0.7'si egitim 0.3'u test verisi olarak ayrildi


#Siniflandirma Agaci
tree.carseats <- tree(High~CompPrice+ Income+ Advertising +Population 
                      +Price +ShelveLoc +Age+ Education +Urban  +US, data=Carseats )
summary(tree.carseats ) #23 Kok dugum
plot(tree.carseats )
text(tree.carseats ,pretty =0)
tree.pred <- predict(tree.carseats ,Carseats ,type="class")
table <- table(tree.pred,High)
table
accuracy <- sum(diag(table))/sum(table)
accuracy #Tum veri kullanilarak olusan modelin dogrulugu %88


High.test <- High[-train_ind]
tree.carseats <- tree(High~CompPrice+ Income+ Advertising +Population 
                      +Price +ShelveLoc +Age+ Education +Urban  +US ,data=train)
summary(tree.carseats )
plot(tree.carseats )
text(tree.carseats ,pretty =0)
tree.pred <- predict(tree.carseats ,test ,type="class")
table <- table(tree.pred ,High.test)
table
accuracy <- sum(diag(table))/sum(table)
accuracy #test uzerindeki performans

set.seed(15)
cv.carseats <- cv.tree(tree.carseats ,FUN=prune.misclass )
cv.carseats
plot(cv.carseats$size ,cv.carseats$dev ,type="b") #Terminal noktalari #8 11 14


prune.carseats <- prune.misclass (tree.carseats,best=14)
plot(prune.carseats )
summary(prune.carseats) #0.90 dogru sinif
text(prune.carseats ,pretty =0)
tree.pred <- predict(prune.carseats ,test , type="class")
table <- table(tree.pred ,High.test)
table
accuracy <- sum(diag(table))/sum(table)
accuracy #test verisinde  %72

prune.carseats <- prune.misclass (tree.carseats,best=11)
plot(prune.carseats )
summary(prune.carseats) #train setinde hata orani %13 dogrulugu %86
text(prune.carseats ,pretty =0)
tree.pred <- predict(prune.carseats ,test , type="class")
table <- table(tree.pred ,High.test)
table
accuracy <- sum(diag(table))/sum(table)
accuracy #test verisinde  %72 #train verisindeki dogruluk daha yuksek

prune.carseats <- prune.misclass (tree.carseats ,best=8)
plot(prune.carseats )
text(prune.carseats ,pretty =0)
summary(prune.carseats) #train setinde hata orani %17
tree.pred <- predict(prune.carseats ,test , type="class")
table <- table(tree.pred ,High.test)
table
accuracy <- sum(diag(table))/sum(table)
accuracy #%76

##Bagging ile Siniflandirma
set.seed(15) 
bag <- randomForest(High~CompPrice + Income + Advertising + Population + 
                      Price + ShelveLoc + Age + Education + Urban + US, data=train,mtry=10,importance=TRUE)
yhat.bag <- predict(bag,newdata=test) 
table <- table(yhat.bag ,High.test)
table
accuracy <- sum(diag(table))/sum(table)
accuracy #%80 dogruluk

bag$importance
varImpPlot(bag)

#Rastgele Orman
set.seed(15) 
rf <- randomForest(High~CompPrice + Income + Advertising + Population + 
                     Price + ShelveLoc + Age + Education + Urban + US, data = train,mtry=3,importance=TRUE)
yhat.rf <- predict(rf,newdata=test) 
table <- table(yhat.rf ,High.test)
table
accuracy <- sum(diag(table))/sum(table)
accuracy
rf$importance
varImpPlot(rf)

#Lojjistik Regresyon
logistic_model<-glm(High~CompPrice + Income + Advertising + Population + 
                   Price + ShelveLoc + Age + Education + Urban + US, data=train,family=binomial) 
summary(logistic_model) #AIC kucuk olan secilir alternatif modellerde
exp(coef(logistic_model)) #Bagimsiz degiskenlerdeki 1 birim artis bagimli degiskeni nasil etkiler
pairs.panels(train)
pairs(train, pch = 19, col='red', lower.panel = NULL) 


confint.default(logistic_model) #katsayılar için güven araligi #0 i iceriyor mu?
vcov(logistic_model) #katsayılar için varyans-kovaryans matriis
sqrt(vcov(logistic_model)[2,2]) #beta1 katsayisinin standart hatasi

##katsayılar için güven aralığı (odds oranı cinsinden)
odds.confint<-exp(confint.default(logistic_model)) #odd ratio cinsinden olan araligin 1 i icermemesi gerekir
odds.confint
#CompPrice 1 birim arttirilinca verilerin sinif 1 de cikma olasiliginin cikmamama olasiligina orani(odds)
#%95 guvenle 1.11 ile 1.25 arasindadir 

p.tahmin<-fitted(logistic_model,train)
p.tahmin[p.tahmin>0.5]=1
p.tahmin[p.tahmin<=0.5]=0
tablo=table(p.tahmin>0.5,train$High)
tablo
sum(diag(tablo))/sum(tablo)*100 #accuracy
120/138 #Sensitivity
131/142 #specifity
                            
#Model gecerliligi
G=logistic_model$null.deviance-logistic_model$deviance
qchisq(0.95,10) #kritik Chisqare değeri Bagimsiz degisken sayisi 10
G>qchisq(0.95,10) ##TRUE Model gecerli
#veya
p_value=1-pchisq(G,10)
p_value #0<0.05

# Artiklar
pearson.res.chd<-residuals(logistic_model,type="pearson")
deviance.res.chd<-residuals(logistic_model,type="deviance") 
cbind(pearson.res.chd,deviance.res.chd)

a<-cbind(train,train$High,p.tahmin)

cbind(train,train$High,pearson.res.chd)
cbind(train,train$High,deviance.res.chd)

plot( train$Advertising ,pearson.res.chd)
plot( train$Income ,pearson.res.chd)
plot( train$CompPrice ,pearson.res.chd)
plot(train$Price,pearson.res.chd )
plot(train$Age,pearson.res.chd )
plot(train$Population ,pearson.res.chd )
plot(train$Education ,pearson.res.chd )

abline(h=c(-2,2))
identify(train$Price ,pearson.res.chd)


#leverage 
hvalues = influence(logistic_model)$hat
r_si=pearson.res.chd/(sqrt(1-hvalues))
d_si=deviance.res.chd/(sqrt(1-hvalues))
cbind(train$Price,r_si,d_si)
par(mfrow=c(1,2))
plot(train$Price,r_si)
plot(train$Price,d_si)

#Cook's.distance
par(mfrow=c(1,1))
plot(cooks.distance(logistic_model))
outlierTest(logistic_model) #5 deger aykiri 16,208,124,348,318
influenceIndexPlot(logistic_model)
influencePlot(logistic_model)

influence.measures(logistic_model) #hangi degerler katsayilar uzerinde etkili

#Artik grafikleri
par(mfrow=c(2,2))
plot(logistic_model)
train_new=train[-c(16,124,208,318,348),]

logistic_new<-glm(High~CompPrice + Income + Advertising + Population + 
                    Price + ShelveLoc + Age + Education + Urban + US, data=train_new,family=binomial) 
summary(logistic_new)

#new model icin 
#Sensitivity ve Specificity hes(p.tahmin.1>0.45) 
p.tahmin.chd.new<-fitted(logistic_new)  
p.tahmin.chd.new[p.tahmin.chd.new>0.5]=1
p.tahmin.chd.new[p.tahmin.chd.new<=0.5]=0


tablo_new=table(p.tahmin>0.5,train$High)
tablo_new
sum(diag(tablo_new))/sum(tablo_new)*100 #accuracy
120/138 #Sensitivity
131/142 #specifity

pseudorsq<-PseudoR2(logistic_model, which ="all")

pseudorsq_new<-PseudoR2(logistic_new, which ="all")

cbind(pseudorsq,pseudorsq_new)

#modelden anlamsiz degiskenler cikarildi
logistic2<-glm(High~CompPrice + Income + Advertising +  
                 Price + ShelveLoc + Age , data=train_new,family=binomial) 
summary(logistic2)
summary(logistic_new)

tahmin<-fitted(logistic2)  
tahmin[tahmin>0.5]=1
tahmin[tahmin<=0.5]=0


tablo_yeni=table(tahmin>0.5,train_new$High)
tablo_yeni
sum(diag(tablo_yeni))/sum(tablo_yeni)*100 #accuracy
121/138 #Sensitivity
127/139 #specifity

N=logistic2$null.deviance-logistic2$deviance
qchisq(0.95,10) #kritik Chisqare değeri Bagimsiz degisken sayisi 10
N>qchisq(0.95,10) ##TRUE Model gecerli

pseudorsq<-PseudoR2(logistic_model, which ="all")
pseudorsq_new<-PseudoR2(logistic2, which ="all")
cbind(pseudorsq,pseudorsq_new)


#LDA

model_lda<-lda(High~.,data=train)
model_lda

#histograms
tahmin_1<-predict(model_lda,train)
hist_lda1<-ldahist(data=tahmin_1$x[,1],g=train$High) #veriler hangi sinifa ait

tahmin_1$class
tahmin_1$posterior
tahmin_1$x #verilerin hangi sinifa ait oldugunun olasiliklari


#Confusion matrix-accuracy-train
tahmin_1<-predict(model_lda,train)
cfmatrix_1<-table(Tahmin=tahmin_1$class, Gercek=train$High)
cfmatrix_1
accuracy_1<-sum(diag(cfmatrix_1))/sum(cfmatrix_1)
accuracy_1

#Confusion matrix-accuracy-test
tahmin_lda<-predict(model_lda,test)
cfmatrix_lda<-table(Tahmin=tahmin_lda$class, Gercek=test$High)
cfmatrix_lda
accuracy_lda<-sum(diag(cfmatrix_lda))/sum(cfmatrix_lda)
accuracy_lda #%87

# QDA

model_qda<-qda(High~.,data=train)
model_qda
tahmin_qda_1<-predict(model_qda,train)
cfmatrix_qda_1<-table(Tahmin=tahmin_qda_1$class, Gercek=train$High)
accuracy_qda_1<-mean(tahmin_qda_1$class==train$High)
accuracy_qda_1

 ##test
tahmin_qda<-predict(model_qda,test)
cfmatrix_qda<-table(Tahmin=tahmin_qda$class, Gercek=test$High)
accuracy_qda<-mean(tahmin_qda$class==test$High)
accuracy_qda

#normallik testi

Sinif0<-Carseats[Carseats$High=="0",]
Sinif1<-Carseats[Carseats$High=="1",]

HZ.test(Sinif0[,-2])
HZ.test(Sinif1[,-1])

DH.test(Sinif1[,-5])
DH.test(Sinif0[,-5])

#Varyans Homojenligi
leveneTest(Carseats$Price ~ Carseats$High, Carseats)
leveneTest(Carseats$Age~ Carseats$High, Carseats)
leveneTest(Carseats$CompPrice ~ Carseats$High, Carseats)
leveneTest(Carseats$Population~ Carseats$High, Carseats)
leveneTest(Carseats$Education ~ Carseats$High, Carseats)
## Varyans homojenligi yoktur p>0.05

