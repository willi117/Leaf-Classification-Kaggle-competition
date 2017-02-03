require(h2o)
require(jpeg)
require(doMC)
require(e1071)
require(caret)
require(h20)
require(dplyr)
require(corrplot)


#Correlation Within column types(Thanks too Jason Liu from kaggle community: https://www.kaggle.com/jiashenliu/leaf-classification/updatedtry-5-different-classifiers-and-questions/run/369103)
#shape
correlationsS<- cor(train %>% select(contains("shape")),use="everything")
corrplot(correlationsS, method="circle", type="lower",  sig.level = 0.01, insig = "blank",main="shape Correlation")
#shape max and min
min(correlationsS)
max(correlationsS)
#margin
correlationsM<- cor(train %>% select(contains("margin")),use="everything")
corrplot(correlationsM, method="circle", type="lower",  sig.level = 0.01, insig = "blank",main="margin Correlation")
#margin max and min
min(correlationsM)
max(correlationsM)
#texture max and min
correlationsT<- cor(train %>% select(contains("texture")),use="everything")
corrplot(correlationsT, method="circle", type="lower",  sig.level = 0.01, insig = "blank",main="texture Correlation")
min(correlationsT)
max(correlationsT)

#parellel processing in mac2
registerDoMC()#mac parrallel proceessing 
options(cores=5)#5 cores 
getDoParWorkers()#check


setwd("/Users/joshuawilliams/Documents/Leaf/leafdeaplearning")
train=read.csv("train.csv", stringsAsFactors=F)  #read in the raw data
test=read.csv("test.csv", stringsAsFactors=F)  #read in the raw data
str(train) # see the data structure
sum(is.na(train)) #check for zeros
str(test) # see the data structure
sum(is.na(test)) #check for zeros

####################Merge Data / Scale Data#########################(Dr.Fulton)
species=rep("", 594)
id=test$id
newtest=cbind(id,species,test[,c(-1)])
total=rbind(train,newtest)

###################Features################## (Dr. Fulton)
#extracting features from the images.
setwd("/Users/joshuawilliams/Documents/Leaf/leafdeaplearning/images")
numrow=c(rep(0,1584)) #initialization
numcol=c(rep(0, 1584)) #iniialization
total$NROW=c(rep(0,1584))
total$NCOL=c(rep(0,1584))
for (i in 1:1584)
{
  fname=paste(i,".jpg", sep="")
  myjpeg=readJPEG(fname, native=F)
  numrow=nrow(myjpeg)
  numcol=ncol(myjpeg)
  total$NROW[i]=numrow
  total$NCOL[i]=numcol
}







#Spliting back into test and train
trainnew=total[1:990,]
testnew=total[991:1584,]


write.csv(trainnew,"trainnew.csv", row.names = F)
write.csv(testnew,"testnew.csv", row.names = F)



#SVM Model PCA
trainnew<-read.csv("trainnew.csv")
testnew<-read.csv("testnew.csv")
testnew$species<-NULL
pca <- prcomp(trainnew[,3:196], scale=T)
summary(pca) #we chose 97% of varience, so 88 features
pcaTrain<-data.frame(train$species,pca$x[,1:88])
pcaTest<-predict(pca,testnew)[,1:88]


#Lets make a model SVM
tuned = tune.svm(train.species~., data = pcaTrain.r, kernal= "radial",gamma = 10^(-3:2),cost = 10^(-3:3))
summary(tuned)
#max is gamma=.001 and cost=10
svm.model <- svm(train.species~.,pcaTrain, probability = T, kernal="radial", gamma=.001, cost=10)
pred.svm<- predict(svm.model,pcaTest, probability = T)
sub7<- data.frame(id=test[,1],attr(pred.svm,"probabilities"))
View(sub7)
write.csv(sub7,'Sub7.csv',row.names=FALSE)


#NaiveBayes .42818

#Build NaiveBayes Model
NaivB<- naiveBayes(train.species~.,pcaTrain)
pred<- predict(NaivB,newdata=pcaTest,type='raw')

#Write prediction
Pred_1<- predict(NaivB,newdata= pcaTest, type='raw')
Sub_1<- data.frame(id=test[,1],Pred_1)
write.csv(Sub_1,'Sub1.csv',row.names=FALSE)



#Now lets try the 
rm(list=ls())  #clears the workspace
            


#The winning model h2o deep learning .o8
library(h2o)
h2o.init(nthreads=-1, max_mem_size="2G")
train.hex= h2o.uploadFile(path="trainnew.csv")
test.hex=h2o.uploadFile(path="testnew.csv")
test.hex$species<-NULL
View(train.hex)

mydeep = h2o.deeplearning(x=c(3:196), y= 2, 
                          overwrite_with_best_model=T,
                          training_frame=train.hex,
                          nfolds=10,
                          single_node_mode = T,
                          activation="Rectifier",
                          hidden=c(1000),
                          epochs=200,
                          train_samples_per_iteration=-2,
                          seed=600,
                          adaptive_rate=T,
                          nesterov_accelerated_gradient=TRUE,
)

predict5<- predict(mydeep,test.hex)
write.csv(as.data.frame(predict5), "predict18.csv")
sub8<-read.csv("predict.18.csv")
sub8$X<-NULL
test<- read.csv("test.csv")
sub8$predict<- test$id
colnames(sub8)[1] <- "id"
View(sub8)
write.csv(sub8,'Sub18.csv',row.names=FALSE)

