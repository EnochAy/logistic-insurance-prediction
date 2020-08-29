#Classification Problem
#Logistic Regression(LR) Algorithm
#LR returns binary result unlike in statistics where regression implies continuous values. The
#algorithm measures the relationship between features are weighted and impact the result
#(1 and 0; in this case, Delayed or not Delayed)

#DNS ML Project using Logistic Regression
#
# Define the problem
# Load the libraries
# Acquire the data
# Ingest the data
# Set woking directory
# Explore data
# Munge the data (if necessary)
# Prepare data
#a. Scale data
#b. split data into train and test datasets
# Train model using the train data
# Run the test data through the model
# validate the model -- accuracy, precision, etc.
#
#Hypothesis (prob(y=0|x;Theta) = 1 - prob(y=1|x;Theta)
#h??(x)=P(y=1|x;??)=1???P(y=0|x;??)
#--> P(y=0|x;??)+P(y=1|x;??)=1
#
#Suppose we want to predict, from data xx about a tumor, whether it is malignant (y=1y=1)
#or benign (y=0y=0). Our logistic regression classifier outputs, for a specific tumor,
#h_\theta(x) = P(y=1\vert x;\theta) = 0.7h
#??
#
#(x)=P(y=1???x;??)=0.7, so we estimate that there is a 70% chance of this tumor being malignant.
# What should be our estimate for P(y=0\vert x;\theta)P(y=0???x;??), the probability the tumor is
# benign?
# P(y=0???x;??)=0.3
##Hypothesis interpretation:
##h(x) = Estimated probability that y(Claim) = 1, given an input value of x
##h(x) = 0.7 for example means there are 70% chance that there is Claim.
##This means that there are 30% chance that there is no Claim
##
# Install packages
install.packages("translations")
install.packages("caretEnsemble")
install.packages("doParallel")
#load library
#library(translations)
library(dplyr)
library(caret)
library(caretEnsemble)
library(mice)
library(doParallel)
library(caTools)
library(caret)
#ingest data
#
#names(full_df)

#Split data set into training and testing
#Create train and test data sets
set.seed(101)
library(caTools)
sample <- sample.split(full_df, SplitRatio = .80)
sample
train <- subset(full_df, sample == TRUE) #X_train
test  <- subset(full_df, sample == FALSE) #X_test
trainingOutcomes <- as.factor(train$Claim) #y_train
testOutcomes <- as.factor(test$Claim) #y_test
levels(testOutcomes)
#Setting Claim(Target/Outcome Variable to NULL)
train$Claim <- NULL
test$Claim <- NULL
# Munge the data (if necessary)
# Converting categorical variables to factors
full_df$Claim <- as.factor(full_df$Claim)
full_df$Building_Painted <- as.factor(full_df$Building_Painted)
full_df$Building_Fenced <- as.factor(full_df$Building_Fenced)
full_df$Garden <- as.factor(full_df$Garden)
full_df$Settlement <- as.factor(full_df$Settlement)

#trainingOutcomes <- as.factor(trainingOutcomes)
#testOutcomes <- as.factor(testOutcomes)
#outcomes <- as.numeric(full_df$Claim)
#testOutcomes.num <- outcomes[6577:10229]
#trainOutcomes.num <- outcomes[1:6576]
#Train the model using the training data
#use glm, the general linear model function
#Dependent variable is Claim, independent variables are:
#The family argument should be binomial to indicate logistic regression
#
#
## Training and test data sets
# We divide 70% of the data set as training, and 30% as testing
mymodel <- glm(trainingOutcomes ~ ., family = "binomial",  data = train)
summary(mymodel)
anova(mymodel)
effects(mymodel)
fitted.values(mymodel)
residuals(mymodel)
#Run the test
attach(test)
pred <- predict(mymodel, newdata = test, type = 'response')
summary(pred)
pred = ifelse(pred > .5, 1, 0)
pred
table(pred, test$Claim)
#pred No Claim Claim
#0     1835     0
#1        0   357
#Validate the model -- confusion matrix (for test)
confmatrix_test <- table(Actual_val = testOutcomes, predicted_value = pred > 0.5)
confmatrix_test
#predicted_value
#Actual_val FALSE TRUE
#No Claim  1835    0
#Claim        0  357
##correct predictions
#(#1835 were predicted false and are actually false)
#(357 were predicted true and are actually true)
#incorrect predictions
#(0 were predicted true, but were actually false, 0 were false, but were actually true)
(confmatrix_test[[1,1]] + confmatrix_test[[2,2]]) / sum(confmatrix_test)
#[1] 1 (100%)
#


#predicted_value
#Actual_valu FALSE TRUE
#0          1834   26
#1           306  24
#correct predictions
#(#1834 was predicted false and are actually false)
#(24 was predicted true and are actually true)
#incorrect predictions
#(26 were predicted true, but were actually false, 306 were false, but were actually true)
#Accuracy
(confmatrix_test[[1,1]] + confmatrix_test[[2,2]]) / sum(confmatrix_test)
#[1]  0.8484018 (85 approx%)
#
pred_train <- predict(mymodel, train, type = 'response')
pred_train

#Validate the model -- confusion matrix (for train)
confmatrix_train <- table(Actual_val = trainingOutcomes, predicted_value = pred_train > 0.5)
confmatrix_train

#predicted_value
#Actual_valu FALSE TRUE
#0          6637  98
#1           1191 113
#correct predictions
#(#6637 was predicted false and are actually false)
#113 was predicted true and are actually true)
#incorrect predictions
#(98 were predicted true, but were actually false, 1191 were false, but were actually true)
#Accuracy
(confmatrix_train[[1,1]] + confmatrix_train[[2,2]]) / sum(confmatrix_train)
#[1]   0.8396567 (84 approx%)
#Second Model building
#MODEL 1: LOGISTIC REGRESSION WITH ALL VARIABLES
model.lr1 <- glm(trainingOutcomes ~ ., data = train, family = binomial(link = "logit"))

summary(model.lr1)
anova(model.lr1)
aov(model.lr1)
model.lr1$fitted.values
model.lr1$residuals
model.lr1$coefficients
model.lr1$effects
model.lr1$rank
model.lr1$assign
model.lr1$qr
model.lr1$df.residual
model.lr1$xlevels
model.lr1$call
model.lr1$terms
model.lr1$model
head(model.lr1$model, n=500)
tail(model.lr1$model, n=500)
nobs(model.lr1)
#Adj R-squared - higher is better.  AIC, BIC - lower the better
aic <- AIC(model.lr1) #[1] 5318.031, [1] 30(newest)
bic <- BIC(model.lr1) #[1] 5318.031, [1] 134.8772
# Mean Prediction Error and Std Error - lower is better
# #keep target variable separate
target_v <- full_df$Claim
trainingOutcomes.num <- target_v[2193:10229] #numeric y_train
testOutcomes.num <- target_v[0:2192] #numeric y_test
pred.model.lr1 <- predict(model.lr1, newdata = test) # validation predictions
meanPred <- mean((testOutcomes.num - pred.model.lr1)^2) # mean prediction error == [1] 11.30005,
#705.756 new
stdError <- sd((testOutcomes.num - pred.model.lr1)^2)/length(testOutcomes.num)
# std error==0.008451522, [1] 2.905017e-09 new
### Create a list to store the model metrics
modelMetrics <- data.frame( "Model" = character(), "adjRsq" = integer(),  "AIC"= integer(), "BIC"= integer(),
                            "Mean Prediction Error"= integer(), "Standard Error"= integer(), stringsAsFactors=FALSE)
# Append a row with metrics for model.lr1
modelMetrics[nrow(modelMetrics) + 1, ] <- c( "model.lr1", 0.279, aic, bic, meanPred, stdError)
modelMetrics
#      Model adjRsq              AIC              BIC Mean.Prediction.Error       Standard.Error
#1 model.lr1  0.279 5466.98050169315 5563.53465053812     0.121123534585718 7.84512070510394e-05

#MODEL 2: BACKWARD SELECTION LINEAR REGRESSION
step <- stepAIC(model.lr1, direction="backward")
step$anova # display results
#Stepwise Model Path
#Analysis of Deviance Table

#Initial Model:
#Claim ~ Customer.Id + YearOfObservation + Insured_Period + Residential +
#Building_Painted + Building_Fenced + Garden + Settlement +
#Building.Dimension + Building_Type + Date_of_Occupancy +
# NumberOfWindows + Geo_Code

#Final Model:
#
#Final Model:
#  trainingOutcomes ~ Claim (new)
#
#Claim ~ Insured_Period + Residential + Building_Painted + Building_Fenced +
# Building.Dimension + Building_Type + Date_of_Occupancy +
#  NumberOfWindows + Geo_Code

#                 Step Df   Deviance Resid. Df Resid. Dev       AIC
#1                                        7295   900.6332 -15274.23
#2        - Settlement  0 0.00000000      7295   900.6332 -15274.23
#3            - Garden  1 0.01923166      7296   900.6524 -15276.07
#4       - Customer.Id  1 0.14921935      7297   900.8016 -15276.86
#5 - YearOfObservation  1 0.17797121      7298   900.9796 -15277.41
#We will then plug these into a new linear modeL as shown below
model.bkwd.lr1 <- glm(trainingOutcomes ~ Claim, train, family = binomial(link = "logit"))
#model.bkwd.lr1 <- glm(trainingOutcomes ~ Insured_Period + Residential + Building_Painted + Building_Fenced +
 #                      Building.Dimension + Building_Type + Date_of_Occupancy +
  #                     NumberOfWindows + Geo_Code, train, family = binomial(link = "logit"))
summary(model.bkwd.lr1)
#Measure the model and add the metrics to our list
#Adj R-squared - higher is better, AIC, BIC - lower the better
aic <- AIC(model.bkwd.lr1) #5327.038
bic <- BIC(model.bkwd.lr1) #5394.95
# mean Prediction Error and Std Error - lower is better
pred.model.bkwd.lr1 <- predict(model.bkwd.lr1, newdata = test) # validation predictions
meanPred <- mean((testOutcomes.num - pred.model.bkwd.lr1 )^2) #  9.935557 mean prediction error
stdError <- sd((testOutcomes.num - pred.model.bkwd.lr1 )^2)/length(testOutcomes.num)#0.001285681std error
# Append a row to our modelMetrics
modelMetrics[nrow(modelMetrics) + 1, ] <- c( "model.bkwd.lr1", 0.2795, aic, bic, meanPred, stdError)
modelMetrics
#Model adjRsq              AIC              BIC Mean.Prediction.Error       Standard.Error
# Model adjRsq              AIC              BIC Mean.Prediction.Error      Standard.Error
#1      model.lr1  0.279 5327.03792079285 5394.94974016021      11.3000456149731 0.00845152218459953
#2 model.bkwd.lr1 0.2795 5327.03792079285 5394.94974016021      9.93555660015492 0.00128568089190738

#MODEL 3: HIGHLY CORRELATED LINEAR REGRESSION
#Create correlation matrix including all numeric variables.  5-8 are categorical data not numeric.
corData <- train[ -c(5:8) ]
head(corData)
corMatrix <- cor(corData, use="complete.obs", method="pearson")
corrplot(corMatrix, type = "upper", order = "hclust", col = c("black", "white"),
         bg = "lightblue", tl.col = "black")
#Create the model with some of the most highly correlated variables as displayed in the above graph.
model.lr2 <- glm(trainingOutcomes ~ Building.Dimension + Customer.Id + NumberOfWindows + Residential +
                  Insured_Period + Geo_Code, train, family = binomial(link = "logit"))
summary(model.lr2)
#Measure and display the model performance.
#Measure performance. Adj R-squared - higher is better, AIC, BIC - lower the better
aic <- AIC(model.lr2)
bic <- BIC(model.lr2)
# mean Prediction Error and Std Error - lower is better
pred.model.lr2 <- predict(model.lr2, newdata = test) # validation predictions
meanPred <- mean((testOutcomes.num - pred.model.lr2 )^2) # mean prediction error
stdError <- sd((testOutcomes.num - pred.model.lr2 )^2)/length(testOutcomes.num) # std error
# Append a row to our modelMetrics
modelMetrics[nrow(modelMetrics) + 1, ] <- c( "model.lr2", 0.2353, aic, bic, meanPred, stdError)
modelMetrics
#Model adjRsq              AIC              BIC Mean.Prediction.Error      Standard.Error
#1      model.lr1  0.279 5327.03792079285 5394.94974016021      11.3000456149731 0.00845152218459953
#2 model.bkwd.lr1 0.2795 5327.03792079285 5394.94974016021      9.93555660015492 0.00128568089190738
#3      model.lr2 0.2353 5361.01214316886 5408.55041672601      9.80278550143275 0.00122463185911191
plot(full_df)
#MODEL 4: BEST SUBSETS LINEAR REGRESSION
model.bestSub = regsubsets(trainingOutcomes ~ ., train, nvmax =25)
summary(model.bestSub)
reg.summary = summary(model.bestSub)
which.min (reg.summary$bic )
which.max (reg.summary$adjr2 ) # just for fun

#Plot the variable bic values by number of variables
plot(reg.summary$bic ,xlab=" Number of Variables ",ylab=" BIC",type="l")
points(6, reg.summary$bic [6], col =" red",cex =2, pch =20)

coef(model.bestSub, 6)

bestSubModel <- glm(trainingOutcomes ~ Insured_Period + Residential + Building_Painted + Building_Fenced +
                    Building_Type + Geo_Code, data=train, family = binomial(link = "logit"))
summary(bestSubModel)

#Adj R-squared - higher is better, AIC, BIC - lower the better
aic <- AIC(bestSubModel)
bic <- BIC(bestSubModel)
# mean Prediction Error and Std Error - lower is better
pred.bestSubModel <- predict(bestSubModel, newdata = test) # validation predictions
meanPred <- mean((testOutcomes.num - pred.bestSubModel)^2)# mean prediction error
stdError <- sd((testOutcomes.num - pred.bestSubModel)^2)/length(testOutcomes.num) # std error

# Append a row to our modelMetrics
modelMetrics[nrow(modelMetrics) + 1, ] <- c( "bestSubModel", 0.1789, aic, bic, meanPred, stdError)
modelMetrics
# Model adjRsq              AIC              BIC Mean.Prediction.Error      Standard.Error
#1      model.lr1  0.279 5327.03792079285 5394.94974016021      11.3000456149731 0.00845152218459953
#2 model.bkwd.lr1 0.2795 5327.03792079285 5394.94974016021      9.93555660015492 0.00128568089190738
#3      model.lr2 0.2353 5361.01214316886 5408.55041672601      9.80278550143275 0.00122463185911191
#4   bestSubModel 0.1789 5651.86167573879 5699.39994929595      9.13856984731041 0.00109118194522037

#Create standard model plots
par(mfrow = c(2, 2))  # Split the plotting panel into a 2 x 2 grid
plot(model.bkwd.lr1)

cutoff <- 4/((nrow(train)-length(model.bkwd.lr1$coefficients)-2))#Create Cooks Distance Plot
plot(model.bkwd.lr1, which=4, cook.levels=cutoff)

#visualize the model
visreg2d(model.bkwd.lr1, "Insured_Period", "Residential", plot.type="persp" )
visreg2d(model.bkwd.lr1, "Insured_Period", "Residential", plot.type="image" )
visreg(model.bkwd.lr1, "Insured_Period")


#Prediction
Predictions <- predict(model.bkwd.lr1, test) # test predictions

#Compring the predicted vs the actual values
plot(trainingOutcomes, type ='l', lty = 1.8, col = 'red')
lines(Predictions, type = 'l', lty = 1.8, col = 'green')



#Check for outliers and other inconsistent data points
cooksd <- cooks.distance(glm(trainingOutcomes~ .,
                             family = "binomial",
                             data = train))


#
#Model 2
registerDoParallel(3)
getDoParWorkers()
set.seed(123)

levels(train$Claim) <- c("no_claim", "claim")
train <- train  %>%
  mutate(trainingOutcomes = factor(trainingOutcomes,
                        labels = make.names(levels(trainingOutcomes))))

my_ctrl <- trainControl(method = "cv",
                        number = 5,
                        classProbs = FALSE,
                        savePredictions = "final",
                        index = createResample(trainingOutcomes, 3),
                        sampling = "up", allowParallel = TRUE)
model2 <- caretList(trainingOutcomes ~ ., data = train, methodList = c("glm", "nb"),
                    metric = "Kappa", tuneList = NULL, continue_on_fail = FALSE,
                    preprocess = c("center", "scale"), trControl = my_ctrl)

install.packages("mlbench")
install.packages("caret")
install.packages("doMC")

#load libraries
library(mlbench)
library(caret)

# load data
#data(PimaIndiansDiabetes)
# rename dataset to keep code below generic
#dataset <- PimaIndiansDiabetes

# set-up test options
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

#  train algorithms

# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(Claim ~., data = train, method="lda", metric=metric, preProc=c("center", "scale"),
                 trControl=control)

# Logistic Regression
set.seed(seed)
fit.glm <- train(Claim~., data = full_df, method="glm", metric=metric, trControl=control)

# GLMNET
set.seed(seed)
fit.glmnet <- train(Claim~., data = full_df, method="glmnet", metric=metric,
                    preProc=c("center", "scale"), trControl=control)

# SVM Radial
set.seed(seed)
fit.svmRadial <- train(Claim~., data = full_df, method="svmRadial", metric=metric,
                       preProc=c("center", "scale"), trControl=control, fit=FALSE)

# kNN
set.seed(seed)
fit.knn <- train(Claim~., data = full_df, method="knn", metric=metric,
                 preProc=c("center", "scale"), trControl=control)

# Naive Bayes
set.seed(seed)
fit.nb <- train(Claim~., data = full_df, method="nb", metric=metric, trControl=control)

# CART
set.seed(seed)
fit.cart <- train(Claim~., data = full_df, method="rpart", metric=metric, trControl=control)

# C5.0
set.seed(seed)
fit.c50 <- train(Claim~., data = full_df, method="C5.0", metric=metric, trControl=control)

# Bagged CART
set.seed(seed)
fit.treebag <- train(Claim~., data = full_df, method="treebag", metric=metric, trControl=control)

# Random Forest
set.seed(seed)
fit.rf <- train(Claim~., data = full_df, method="rf", metric=metric, trControl=control)

# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(Claim~., data = full_df, method="gbm", metric=metric, trControl=control,
                 verbose=FALSE)


# Compare algorithms
results <- resamples(list(lda = fit.lda, logistic = fit.glm, glmnet = fit.glmnet,
                          svm = fit.svmRadial, knn = fit.knn, nb = fit.nb, cart = fit.cart,
                          c50 = fit.c50, bagging = fit.treebag, rf = fit.rf, gbm = fit.gbm))

# Table comparison
summary(results)

# boxplot comparison
bwplot(results)

# Dot-plot comparison
dotplot(results)
