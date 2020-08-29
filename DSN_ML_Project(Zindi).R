#Webscrapping in R
url<- "'https://..."
data <- read.csv(url, header = TRUE)
#Zindi Project from DSN ML Open Competition
##Data Science Nigeria 2019 Challenge #1: Insurance Prediction
##
##
###Description of the challenge:

#Recently, there has been an increase in the number of building collapse in Lagos and major cities
#in Nigeria. Olusola Insurance Company offers a building insurance policy that protects buildings
#against damages that could be caused by a FIRE or VANDALISM, by a FLOOD or STORM.

##You have been appointed as the Lead Data Analyst to BUILD a PREDICTIVE MODEL to determine
##if a building will have an INSURANCE CLAIM during a certain period or NOT. You will have to
##PREDICT THE PROBABILITY of having AT LEAST ONE CLAIM OVER THE INSURED PERIOD of the building.
#Most categorical data like Building_Painted, Building_Fenced, Garden, Settlement are
#converted using map encoding to 0 and 1.

##THE MODEL WILL BE BASED ON THE BUILDING CHARACTERISTICS, which includes:
##NumberOfWindows,  Building.Dimension, Building_Painted, Building_Fenced, Garden.
##The target variable, Claim, is a:
#1 if the building has at least a claim over the insured period.
#0 if the building doesn't have a claim over the insured period.

#Predictor Variables: Insured_Period, NumberOfWindows,  Building.Dimension, Building_Painted,
#Building_Fenced, Garden.
#Target Variable: Claim (At least one claim:1, No claim:0)
#
#Logistic Regression(LR) Algorithm (A Type of Classification algorithm under the Supervised
#Learning type of ML)
#LR returns binary result unlike in statistics where regression implies continuous values. The
#algorithm measures the relationship between features are weighted and impact the result
#(1 and 0; in this case, 1, if the building has at least one claim over the insured period
#or 0, if the building doesn't have a claim over the insured period, no claim)



#install packages - do this one time
install.packages("data.table")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("ggplot")
install.packages("gcookbook")
install.packages("caret")
install.packages("hexbin")
install.packages("leaps")
install.packages("plyr")
install.packages("plotly")
install.packages("waffle")
install.packages("dummies")
install.packages("caTools")
install.packages("wesanderson")
install.packages("visreg")
install.packages("car")
install.packages("leaps")
install.packages("MASS")
install.packages("corrplot")
# Load the relevant libraries - do this every time
library(data.table)
library(corrplot)
library(ggplot2)
library (gcookbook)
library(caret)
library(hexbin)
library(leaps)
library(plyr)
library(plotly)
library(waffle)
library(dummies)
library(caTools)
library(wesanderson)
library(visreg)
library(car)
library(rpart)
library(leaps)
library(MASS)
library(skimr)
library(corrplot)


#Load train_data and test_data
train_data <- read_csv("train_data.csv")
train_data <- read_csv("Zindi(DSN Dataset)/train_data.csv")
test_data <- read_csv("Zindi(DSN Dataset)/test_data.csv")
library(readr)
train_data <- read_csv("Machine Learning Projects/Zindi(DSN Dataset)/train_data.csv")
test_data <- read_csv("Machine Learning Projects/Zindi(DSN Dataset)/test_data.csv")

#Part One: Problem Definition and Forming Solution Statement.
#Colllecing, Preparing and Cleaning Data.
##
library(dplyr)
#Joined by columns
full_data <- full_join(test_data, train_data)
#Join by rows == bind_rows(Test, train)
full <- bind_rows(train_data, test_data)

#saving train and test data as full_data
save(list=c("test_data","train_data"),file="full_data.Rdata")
data(full_data.Rdata)
#Exploring data structure
str(full_data)
#10229 obs. (records or examples: customers, n = 1000) of  14 variables(features)
str(train_data)
#7160 obs. (records or examples: customers, n=7160) of  14 variables(features)
str(test_data)
#3069 obs. (records or examples: customers, n=3069) of  13 variables(features)

#listing the names of variables/features/columns
names(test_data)
names(train_data)
names(full_data)
#Data types #types of each variable or attributes
sapply(test_data, class)
sapply(train_data, class)
sapply(full_data, class)
sapply(full, class)


#Quick View of Dataset
head(train_data)
tail(train_data)
str(train_data)

head(test_data)
tail(test_data)
str(test_data)

head(full_data)
tail(full_data)
str(full_data)
library(dplyr)
glimpse(train_data)
glimpse(test_data)
glimpse(full_data)
#Checking for missing values
is.na(test_data)
is.na(train_data)
is.na(train_data$Date_of_Occupancy)
is.na(full_data)


summary(test_data)
summary(train_data)
summary(full_data)
#plots for train and test datasets after conversion are now rectangular
hist(train_data$Claim)
hist(train_data$Geo_Code)
hist(train_data$Insured_Period)

#before transformation
#skewed to the left (negatively skewed). It is skewed towards the older age grp
# how will this impact my results when using it to predict over the
# remaining customer base?
hist(test_data$Geo_Code)
hist(test_data$Insured_Period)
hist(test_data$`Building Dimension`)

#skewed to the left (negatively skewed)
hist(full_data$Claim)
hist(full_data$Geo_Code)
hist(full_data$Insured_Period)
#To calculate the skewness for each variable
skew <- apply(full_data$Claim[, 1:29], 2, skew)
#Summary statistics of the entire datasets
library(skimr)
skim(full_df)
skim(train_data)
skim(test_data)
#To remove a value, say a col(var.) X, set it to NULL, thus
#train_data$X <- NULL
#saving train and test data as full_data
save(list=c("full_df"),file="full_df.Rdata")
data(full_df.Rdata) #Full data

save(list=c("full_cat"),file="full_cat.Rdata")
data(full_cat.Rdata) #Full_categorical

save(list=c("full_numerics"),file="full_numerics.Rdata")
data(full_numerics.Rdata) #Full_numerical
full_df <- data.frame(full)
#rm(full)
table(full_df$Claim)
#0    1
#8595 1634
round(prop.table(table(full_df$Claim)) * 100, digits = 1)
#0  1
#84 16
summary(full_df[c("Claim", "Insured_Period", "Building_Type")])
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
normalize(c(1,2,3,4,5)) #[1] 0.00 0.25 0.50 0.75 1.00
normalize(c(10,20,30,40,50)) #0.00 0.25 0.50 0.75 1.00 (function is working normally)
normalize(full_numerics)
summary(normalize(full_numerics))
full_df["Claim"] <- lapply(full["Claim"], factor,
                                      labels = c("No Claim", "Claim"), levels = c(0,1))
#Or
full_df$Claim <- factor(full$Claim, levels = c(0,1), labels = c("No Claim", "Claim"))

#EXPLORATORY DATA ANALYSIS
#Understand the correlation between columns
#attach allows us to reference columns by their name
attach(full_df)
#Initail EDA
head(full_df)
tail(full_df)
dim(full_df)
str(full_df)
summary(full_df)
skim(full_numerics) # Descriptive statistics of dataset
skim(full_cat) # Descriptive statistics of dataset
skim(full_df) # Descriptive statistics of dataset

#Creating numeric variables
full_numerics <- subset(full_df,
select = -c(Building_Painted, Building_Fenced, Settlement, Claim))#Subsetting numeric variables
corMatrix <-cor(full_numerics, use="complete.obs", method="pearson")#Check Correlations of numeric columns
round(corMatrix, 2)#round to two decimals
#                       Customer.Id YearOfObservation Insured_Period  Residential Building.Dimension
#Customer.Id               1.00              0.05          -0.01        0.19               0.12
#YearOfObservation         0.05              1.00           0.03       -0.05               0.01
#Insured_Period           -0.01              0.03           1.00       -0.06               0.01
#Residential               0.19             -0.05          -0.06        1.00               0.03
#Building.Dimension        0.12              0.01           0.01        0.03               1.00
#Building_Type             0.06              0.00          -0.03        0.29               0.02
#Date_of_Occupancy         0.00             -0.03          -0.01       -0.03               0.18
#NumberOfWindows           0.06              0.02           0.01        0.09               0.33
#Geo_Code                  0.09             -0.01           0.00        0.25              -0.02
#Claim                     0.08             -0.01           0.07        0.08               0.25
#                   Building_Type Date_of_Occupancy NumberOfWindows Geo_Code Claim
#Customer.Id                 0.06              0.00            0.06     0.09  0.08
#YearOfObservation           0.00             -0.03            0.02    -0.01 -0.01
#Insured_Period             -0.03             -0.01            0.01     0.00  0.07
#Residential                 0.29             -0.03            0.09     0.25  0.08
#Building.Dimension          0.02              0.18            0.33    -0.02  0.25
#Building_Type               1.00             -0.12            0.16    -0.02  0.06
#Date_of_Occupancy          -0.12              1.00           -0.04     0.03  0.00
#NumberOfWindows             0.16             -0.04            1.00     0.02  0.17
#Geo_Code                   -0.02              0.03            0.02     1.00  0.04
#Claim                       0.06              0.00            0.17     0.04  1.00
#Summary of the above:
#Numerical Variables with positive correlation with the target variable in descending order are:
#Building.Dimension, NumberOfWindows, Residential, Customer.Id, Insured_Period, Building_Type.
#Only YearOfObservation has negative correlation with the target variable
#
#Creating categorical variables
full_cat <- subset(full_df, select = c(Building_Painted, Building_Fenced, Claim, Settlement))
full_df$key <- NULL #Deleting the variable key from the full_df
corrplot(cor(full_numerics), method = "circle")#correlation matricies package
#Create a better visualization for the column correlation
corrplot(corMatrix, method="circle")
corrplot(corMatrix, method="square")
corrplot(corMatrix, method="number")
corrplot(corMatrix, method="shade")
corrplot(corMatrix, type = "upper")
corrplot.mixed(corMatrix)
#Categorical Variables with positive correlation with the target variable include:
#Building.Dimension, NumberOfWindows, Residential, Customer.Id, Insured_Period, Building_Type.
#Only YearOfObservation has negative correlation with the target variable
#
#View a histogram of the average amount of hours worked per week - Histograms
## Run a histogram for all numeric variables to understand distribution
#Creating histogram for all numeric variables
hist(full_numerics$Insured_Period, main="Insured_Period per year", xlab="Insured_Period", breaks=7, col="lightblue")
hist(full_numerics$Customer.Id, main="Customer_Id", xlab="Customer.Id", breaks=7, col="lightblue")
hist(full_numerics$YearOfObservation, main="Years of Observation", xlab="YearofObs", breaks=7, col="lightblue")
hist(full_numerics$Residential, main="Residential Address", xlab="Residential", breaks=7, col="lightblue")
hist(full_numerics$Claim, main="Claim over the years", xlab="Insurance_Claim", breaks=7, col="lightblue")
hist(full_numerics$NumberOfWindows, main="Number of Windows", xlab="Numbrof windows", breaks=7, col="lightblue")
hist(full_numerics$Building_Type, main="Types of building", xlab="Building types", breaks=7, col="lightblue")
hist(full_numerics$Building.Dimension, main="Dimension of Building", xlab="Building Dimension", breaks=7, col="lightblue")
hist(full_numerics$Date_of_Occupancy, main="Date of Occupancy Per Year", xlab="Date of Occupancy", breaks=7, col="lightblue")
hist(full_numerics$Geo_Code, main="Geographical Code", xlab="Geo_Code", breaks=7, col="lightblue")


#relationship between the quality of diamonds and their price: low quality diamonds
#(Insured_Period, NumberOfWindows, Building.Dimension, )
ggplot(full_df, aes(full_df$Insured_Period, full_df$Claim)) + geom_boxplot()
ggplot(full_df, aes(NumberOfWindows, Claim)) + geom_boxplot()
ggplot(full_df, aes(Building.Dimension, Claim)) + geom_boxplot()

library(dplyr)
#Make full dataset
full <- full_join(test, train)
#Before imputation of NAs, form new datasets
train <- full[full$key=='train', ]
test <- full[full$key=='test', ]
train <- full_data[full_data$key=='train_data', ]
test <- full_data[full_data$key=='test_data', ]

#Create some new variables to better display the graph output.
hr$leftFactor <- factor(left,levels=c(0,1),
                        labels=c("Did Not Leave Company","Left Company"))

hr$promoFactor <- factor(fiveYrPrmo,levels=c(0,1),
                         labels=c("Did Not Get Promoted","Did Get Promoted"))

hr$wrkAcdntFactor <- factor(wrkAcdnt,levels=c(0,1),
                            labels=c("No Accident","Accident"))

#View a density plot showing the average hours per week by salary category
#density plot
qplot(avgHrs/4, data=hr, geom="density", fill=salary, alpha=I(.5),
      main="Avg Weekly Hours by Salary Category", xlab="Average Weekly Hours",
      ylab="Density")
#View a density plot showing the average hours per week by employee retention
qplot(avgHrs/4, data=hr, geom="density", fill=leftFactor, alpha=I(.5),
      main="Average Weekly Hours and Employee Retention", xlab="Average Weekly Hours",
      ylab="Density")
#Create a box plot to show the percentile distribution of average hours per week by job type.
#boxplot
boxplot(Claim~Insured_Period,data=full_df, main="Insurance_Data",
        xlab="Insured_Period", ylab="Claim", col="lightblue")

#Next, create a violin plot to visualize the same variables.

#violin plot for Claim and Geo_Code
ClaimBox <- ggplot(full_df, aes(y=Claim, x=Geo_Code))
ClaimBox + geom_violin(trim=FALSE, fill="lightblue")
#violin plot for Claim and Insured_Period
ClaimBox_Insured <- ggplot(full_df, aes(y=Claim, x=Insured_Period))
ClaimBox_Insured + geom_violin(trim=FALSE, fill="lightblue")

#Plot a chart with many dimensions #many dimension charts
qplot(avgHrs/4, timeCpny, data=hr, shape=leftFactor, color=salary, facets=numProj~promoFactor,
      size=I(3), xlab="average hours per week", ylab="time at company")
#Find clusters of Claim when considering two variables in a scatter plot
fullcat <- ggplot(full_df, aes(x = Insured_Period, y = Claim))
fullcat + geom_point()
#make the points more transparent so that it's less intense
#hrScat + geom_point(alpha=.01)
#hrScat + stat_bin2d
#Execute the same chart as above but with a hexagon function
#hrScat + stat_binhex()
#hrScat + stat_binhex() + scale_fill_gradient(low="lightblue", high="red")
#Try a different variable combination.  Examine last employee evaluation vs their sat level.
#LEvSL <-ggplot(hr, aes(x=satLevel, y=lastEval))
#LEvSL + geom_point()
#LEvSL + stat_binhex() + scale_fill_gradient(low="lightblue", high="red")


#To eliminate columns that are: Not used, no values/Full of NAs Values, Duplicates.This can be done
#by correlated function, which helps to identify cols stating the same variables in different ways.
# This can amplify bias cos some algorithms can naively treat them as diffent variables. This can
# be done by using the cor function.
#
#Check for correltion of variables using the cor(), the closer the val to 1, the more correlated
cor(train_data[c("Building Dimension" , "Building_Type")])
cor(test_data[c("Building Dimension" , "Building_Type")])

#    ORIGIN_AIRPORT_ID ORIGIN_AIRPORT_SEQ_ID
#ORIGIN_AIRPORT_ID                     1                     1
#ORIGIN_AIRPORT_SEQ_ID                 1                     1
#This shows a perfect correlation between the two. Hence, one of the pairs has to be dropped.
cor(train[c("Geo_Code" , "Date_of_Occupancy")])
#                         Geo_Code         Date_of_Occupancy
#Geo_Code                  1.00000000             0.0215906
#Date_of_Occupancy         0.0215906              1.00000000
cor(test[c("Geo_Code" , "Date_of_Occupancy")])
#                         Geo_Code   Date_of_Occupancy
#Geo_Code                 1.00000000        0.04301084
#Date_of_Occupancy        0.04301084        1.00000000

cor(train[c("Claim" , "Insured_Period")])
#               Claim      Insured_Period
#Claim          1.00000000     0.06503408
#Insured_Period 0.06503408     1.00000000

cor(full_df[c("Claim" , "Insured_Period")])
#               Claim      Insured_Period
#Claim          1.00000000     0.06564176
#Insured_Period 0.06564176     1.00000000

##Part two: Molding Data (Data Transformation and Feature Engineering)
##Removing rows that could not be used, changing data types columns to what is needed such as
##Using as.numeric, etc and finally verify that the data could be used for prediction
#SOlution statement: finding flights that arrives 15 or more minutes late
#Molding Data: Dropping rows, Adjusting data types, Creating new columns, if required.
#Arr_Del15 = 1 if 15 minutes delay, value we are trying to predict, must be 0 or 1: 0 for FALSE,
# 1 for TRUE. We need to drop values set to Na or an empty string, "" in fields we might be using
# for our prediction.
#
#
# Note: For this project,the number of variables in train_data is 14, but 13 in test_data,
# hence, there is need to ensure that the missing variable in test_data(Claim) which is
# available in train_data is added to the test_data.This is the first task in this face.
# This problem is partly solved by combining the train and test data set into full_data,
# and all necessary conversions made.
# The lines of code below is not needed nor useful here since there are lots of missing values
# in our target variable and we can afford to drop all these cos this will negatively affect data
# integrity and findings ultimately. Hence, missing value techniques will have to come to use here
# OnTimeData <- OrigData[!is.na(OrigData$ARR_DEL15) & OrigData$ARR_DEL15!="" & !is.na(
  #OrigData$DEP_DEL15) & OrigData$DEP_DEL15!="",]


#I define the function `missing_vars`,
#which I can use to get an overview of what proportion of each variable is #missing,
#and re-use it later if I need to.
missing_vars <- function(x) {
  var <- 0
  missing <- 0
  missing_prop <- 0
  for (i in 1:length(names(x))) {
    var[i] <- names(x)[i]
    missing[i] <- sum(is.na(x[, i]))
    missing_prop[i] <- missing[i] / nrow(x)
  }
  (missing_data <- data.frame(var = var, missing = missing, missing_prop = missing_prop) %>%
      arrange(desc(missing_prop)))
}
#Call the function on full dataset
missing_vars(test)
missing_vars(full_cat)
missing_vars(full_numerics)

#var missing missing_prop
#1     NumberOfWindows    5802  0.567210871
#2               Claim    3069  0.300029328
#3   Date_of_Occupancy    1236  0.120832926
#4            Geo_Code     323  0.031576889
#5  Building Dimension     119  0.011633591
#6              Garden      11  0.001075374
#7         Customer Id       0  0.000000000
#8   YearOfObservation       0  0.000000000
#9      Insured_Period       0  0.000000000
#10        Residential       0  0.000000000
#11   Building_Painted       0  0.000000000
#12    Building_Fenced       0  0.000000000
#13         Settlement       0  0.000000000
#14      Building_Type       0  0.000000000
#
##Before imputation of NAs, form new datasets
##

#Imputting missing values using Hmisc
library(missForest)
library(Hmisc)

#create numeric factors for the two levels in Building_Painted, Building_Fenced, and
#Settlement.
full_data["Building_Painted"] <- lapply(full_data["Building_Painted"], factor,
                                        levels = c("N", "V"), label = c(1,0))

full_data["Building_Fenced"] <- lapply(full_data["Building_Fenced"], factor,
                                       levels = c("N", "V"), label = c(1,0))
full_data["Garden"] <- lapply(full_data["Garden"], factor, levels = c("V", "O"), label = c(1,0))

full_data["Settlement"] <- lapply(full_data["Settlement"], factor, levels = c("R", "U"),
                                  label = c(1,0))

#Converting variables type such that R and algorithms could easily work on them
#as.numeric, as.string functions could be use to change back to formal type if necessary.
full_data$`Customer Id` <- as.integer(full_data$`Customer Id`)
full_data$NumberOfWindows <- as.integer(full_data$NumberOfWindows)
full_data$Building_Painted <- as.factor(full_data$Building_Painted)
full_data$Building_Fenced <- as.factor(full_data$Building_Fenced)
full_data$Garden <- as.factor(full_data$Garden)
full_data$Settlement <- as.factor(full_data$Settlement)



#My target variable for this project is the Claim variable
#At least, one claim == 1, No claim == 0
#Next is to check the data distribution if it will allow for training the prediction
#Data Rule 3: Accurately predicting rare events is difficult
#With True == 1, False == 0, use the tapply () to check
tapply(full_df$Claim, full_df$Claim, length)
#0(No Claims)    1(Claims)
#8595            1634
#Percentage of Claims to No-Claims in the dataset
(1634 / (1634 + 8595))
#[1] 0.1597419 About 16% of having at least one claim, meaning that we can reasonably train our
#model with the data we have

#DataRule 4: Track how you manipulate data. This is to help check if the meaning of data has not
#be changed intentionally or unintentionally
#
#Part Three: SELECTING ALGORITHMS
#The problem knowlege helps to determine the algorithm to use
#Task here:
#Role of the algorithm, Perform algorithm selection  (By using solution statement as a
#guide to filter algorithms, discuss best algorithms based on the solution statement), finally,
#select one algorithm to be the initial algorithm.

#A: The Role of the
#The algorithm is the engine that drives the entire process
#The function train() is called om the Training Data which contains variables we are trying to
#predict. Using the training data, the algorithm gives us a trained model which contains codes
#with which we can evaluate
#The model also contains a function called predict(). The real data is passed into this to
#evaluate the data and produce result
#
#
#Selecting our Initial Algorithm.
#To do this, we can compare the algorithms with a lot of factors (Difference of opinions about which factors are important) but you will decide your factors as you gain experience.
#Algorithm Decision factors: Learning type, Result, Complexity, Basic vs enhanced.
#Learning Type
#Algorithm work best based on the type of prob at hand. Check solution statement again:
#"Use the Machine Learning Workflow to process and transform DOT data to create a prediction model. This model must predict whether a flight would arrive 15+ minutes after the scheduled arrival time with 70+% accuracy."
#Our solution is focused on Prediction,
#Prediction Model => Supervised machine learning (SML). Then, all algorithms that do not support SML are dropped. Reducing over 50 Algorithms to 28.
#Result Type
#These are of two types:
#1. Regression (Continuous Values, price = A * #bedroom + B * size + ...)
#2. Classification (Discrete Values such as small, medium, large, 1-100, 101-200, 201-300,
#True or False: Arr_Del15 is a binary outcome with 1 for True and 0 for False). This reduced the
#no of algorithms to 20. Not much this bcos many algorithms support regression and Classification
#algorithms.
#Complexity
#Keep it simple(Eliminate "ensemble" algorithms which are algorithms that are Container algorithms,
#multiple child algorithms, they Boosts performance, they can be complex ans difficult to debug
#(diagnose when there are problems in the code) . Reduced to 14
#We can further divide this into two types: Enhanced vs Basic
#Enhanced: Variation of Basic, Performance improvements, Additional functionality, more complex.
#Basic: are simpler and thus easier to understand.


#Basic Algorithms for Initial Training and Evaluation:
#1. Naive Bayes
#2. Logistic Regression
#3.  Decision Tree
#An understanding of each of these can greatly help in understanding more complex ones since these
#are the foundation for the complex ones.

#1. Naive Bayes: Based on likelihood and probability. It is based on Bayes theorem, its assumes
#that every feature has the same weight.
#It calculate the probability of flight being delayed by using the likelihood of delay based on
#previous data combined with probability of delay based on nearby feature values. It makes a naive
#assumption that all features we passed are independent of each other and equally impact the result
#(every feature has the same weight).
#This assumption allows for fast conversion and therefore requires smaller amount of data to train.

#Logistic Regression(LR) Algorithm
#LR returns binary result unlike in statistics where regression implies continuous values. The
#algorithm measures the relationship between features are weighted and impact the result
#(1 and 0; in this case, Delayed or not Delayed)

#Decision Tree Algorithm
#It uses binary tree structure with each node contains decision based on the value to feature.
#It requires enough data to determine nodes and splits

#For this Course, we select the LG cos:
#it is simple- easy to understand.
#Once the value hits a treshood, it is delayed(True), otherwise not delay(False).
#It is fast- up to 100X faster
#is stable to data changes.  unlike other algorithms, in which small changes to data can cause
#a wide difference.
#DEALING WITH MISSING VALUES IN EDA (FILLING IN MISSING VALUES)
library(tidyverse)
## replacing "--" with NA
full_df$NumberOfWindows <- full_df %>%
  mutate(NumberOfWindows = replace(NumberOfWindows, NumberOfWindows ==  "--", NA))
# replace NA with "unavailable"
full_df$NumberOfWindows <- full_df %>%
  mutate(NumberOfWindows = replace(NumberOfWindows, is.na(NumberOfWindows), "unavailable"))

# replace missing values with median
full_df <- full_df %>%
  mutate(NumberOfWindows = replace(NumberOfWindows,
                                is.na(NumberOfWindows),
                                median(NumberOfWindows, na.rm = T)))

missing_vars(full_df)
full_df <- full_df %>%
  mutate(Claim = replace(Claim, is.na(Claim),
                                     median(Claim, na.rm = T)))

full_df <- full_df %>%
  mutate(Date_of_Occupancy = replace(Date_of_Occupancy, is.na(Date_of_Occupancy),
                        median(Date_of_Occupancy, na.rm = T)))
full_df <- full_df %>%
  mutate(Geo_Code = replace(Geo_Code, is.na(Geo_Code),
                                     median(Geo_Code, na.rm = T)))

full_df <- full_df %>%
  mutate(Building.Dimension = replace(Building.Dimension, is.na(Building.Dimension),
                                     median(Building.Dimension, na.rm = T)))

full_df$Garden <- as.numeric(full_df$Garden)
full_df <- full_df %>%
  mutate(Garden = replace(Garden, is.na(Garden),
                                      median(Garden, na.rm = T)))
missing_vars(full_df)
sum(is.na(full_df))
#Patterns and Model under EDA
ggplot(data = full_df) +
geom_point(mapping = aes(x = Insured_Period, y = Claim))

library(modelr)
mod <- lm(Claim ~ Insured_Period, data = full_df)#This model predicts claim from insured_Period
coef(mod)
#(Intercept) Insured_Period
#0.0657584      0.1028635. The equation: y(Claim) = 0.1028635(Insured_Period) + 0.0657584(Intercept)
summary(mod)
mod$fitted.values
mod$coefficients
mod$residuals
mod$effects
mod$rank
mod$assign
mod$qr
mod$df.residual
mod$xlevels
mod$call
mod$terms
mod$model
head(mod$model, n=500)
tail(mod$model, n=500)
nobs(mod)#Checking the number of observations used in the model
#[1] 10229
#This computes the residuals (the difference between the predicted value and the actual value)
full_df2 <- full_df %>%
  add_residuals(mod) %>%
  mutate(resid = exp(resid)) #Save the residual as a feature in the full_df (full_df2)
#Plotting the graph of residual and Insured_Period
ggplot(data = full_df2) +
  geom_point(mapping = aes(x = Insured_Period, y = resid))
#The residuals give us a view of the Claim of the , once the effect of Insured_Period
#has been removed.
#To visualise the predictions from a model, we start by generating an evenly spaced grid of values
# that covers the region where our data lies using  modelr::data_grid()
grid <- full_df %>%
data_grid(Insured_Period)
summary(grid)
#Next we add predictions. We'll use modelr::add_predictions() which takes a data frame and a model.
#It adds the predictions from the model to a new column in the data frame:
grid <- grid %>%
  add_predictions(mod)
grid
#Insured_Period   pred
#<dbl>  <dbl>
#1        0       0.0658
#2        0.00273 0.0660
#3        0.0109  0.0669
#4        0.0110  0.0669
#5        0.0164  0.0674
#6        0.0164  0.0674
#7        0.0219  0.0680
#8        0.0246  0.0683
#9        0.0247  0.0683
#10        0.0274  0.0686
# ... with 457 more rows
#
#Next, we plot the predictions(pred) against the Insured_Period in grid above, using geom_abline()
ggplot(full_df, aes(Insured_Period)) +
  geom_point(aes(y = Claim)) +
  geom_line(aes(y = pred), data = grid, colour = "red", size = 1)
#
mod1 <- full_df %>%
  add_residuals(mod)
mod1
#Freq Polygon plot showing the info about residual of the model
ggplot(mod1, aes(resid)) +
  geom_freqpoly(binwidth = 0.5)

ggplot(mod1, aes(Insured_Period, resid)) +
  geom_ref_line(h = 0) +
  geom_point()
#Part 4
#raining the Model (Understanding the training process)
#Caret package which makes the training and evaluation processes easier. We will go back to our
#train algorithm with DOT data and produce a trained model.
#his is letting specific data teach a Machine Learning Algorithm to create a specific forecast
#model. (Note the term "specific".

#rm(full_cat, full_numerics, full_df, t_d_2, test, test_data, testSet, train, train_data, sample,
   #testOutcomes, trainingOutcomes)
#TRANSFORM DATA(Data Transformation)
library(dummies)
#create dummy variables for job and salary
t_d_2 <- cbind(train_data, dummy(train_data$Settlement), dummy(train_data$Garden),
              dummy(train_data$Garden), dummy(train_data$Building_Painted),
              dummy(train_data$Building_Fenced))

names(full_df)
#Split data set into training and testing
#Create train and test data sets
set.seed(101)
library(caTools)
sample = sample.split(full_df, SplitRatio = .75)
sample
train = subset(full_df, sample == TRUE) #X_train
test  = subset(full_df, sample == FALSE) #X_test
trainingOutcomes <- train$Claim #y_train
testOutcomes <- test$Claim #y_test

#Setting Claim(Target/Outcome Variable to NULL)
train$Claim <- NULL
test$Claim <- NULL
#You're looking for about a 75% split as seen in the SplitRatio above.
dim(train) #[1] 7308   13(75%)
dim(test) #[1] 2921   13( 25%)
summary(trainingOutcomes) #(7308, )
summary(testOutcomes) #(2921, )

#MODEL 1: LINEAR REGRESSION WITH ALL VARIABLES
model.lr1 <- glm(trainingOutcomes ~ ., data = train) #The target variable(label, y) has to be contained in train for
#this to run
#model.lr1$coefficients
#(Intercept)        Customer.Id  YearOfObservation     Insured_Period        Residential
#7.888821e+00       9.741293e-07      -3.662975e-03       1.066882e-01       3.941408e-02
#Building_Painted0   Building_Fenced0             Garden        Settlement0 Building.Dimension
#3.889593e-02      -6.651305e-02      -4.918304e-02                 NA       3.586903e-05
#Building_Type  Date_of_Occupancy    NumberOfWindows           Geo_Code
#9.685311e-03      -3.061224e-04       2.603598e-02       3.764562e-07
#y  = mx + b: Here, b(intercept=7.888821e+00), m(slope=)
summary(model.lr1)
anova(model.lr1)
aov(model.lr1)
model.lr1$fitted.values
model.lr1$coefficients
model.lr1$residuals
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
aic <- AIC(model.lr1) #[1] 5466.981
bic <- BIC(model.lr1) #[1] 5563.535
# Mean Prediction Error and Std Error - lower is better
pred.model.lr1 <- predict(model.lr1, newdata = test) # validation predictions
meanPred <- mean((test$Claim - pred.model.lr1)^2) # mean prediction error == [1] 0.1211235
stdError <- sd((test$Claim - pred.model.lr1)^2)/length(test$Claim) # std error==[1] 7.845121e-05
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
  #Claim ~ Insured_Period + Residential + Building_Painted + Building_Fenced +
 # Building.Dimension + Building_Type + Date_of_Occupancy +
#  NumberOfWindows + Geo_Code

#Step Df   Deviance Resid. Df Resid. Dev       AIC
#1                                        7295   900.6332 -15274.23
#2        - Settlement  0 0.00000000      7295   900.6332 -15274.23
#3            - Garden  1 0.01923166      7296   900.6524 -15276.07
#4       - Customer.Id  1 0.14921935      7297   900.8016 -15276.86
#5 - YearOfObservation  1 0.17797121      7298   900.9796 -15277.41
#We will then plug these into a new linear modeL as shown below
model.bkwd.lr1 <- lm(Claim ~ Insured_Period + Residential + Building_Painted + Building_Fenced +
    Building.Dimension + Building_Type + Date_of_Occupancy +
    NumberOfWindows + Geo_Code, train)
summary(model.bkwd.lr1)
#Measure the model and add the metrics to our list
#Adj R-squared - higher is better, AIC, BIC - lower the better
aic <- AIC(model.bkwd.lr1)
bic <- BIC(model.bkwd.lr1)
# mean Prediction Error and Std Error - lower is better
pred.model.bkwd.lr1 <- predict(model.bkwd.lr1, newdata = test) # validation predictions
meanPred <- mean((test$Claim - pred.model.bkwd.lr1 )^2) # mean prediction error
stdError <- sd((test$Claim - pred.model.bkwd.lr1 )^2)/length(test$Claim) # std error
# Append a row to our modelMetrics
modelMetrics[nrow(modelMetrics) + 1, ] <- c( "model.bkwd.lr1", 0.2795, aic, bic, meanPred, stdError)
modelMetrics
#Model adjRsq              AIC              BIC Mean.Prediction.Error       Standard.Error
#1      model.lr1  0.279 5466.98050169315 5563.53465053812     0.121123534585718 7.84512070510394e-05
#2 model.bkwd.lr1 0.2795  5463.7909320728 5539.65490616528     0.121304387490843  7.8528627168664e-05

#MODEL 3: HIGHLY CORRELATED LINEAR REGRESSION
#Create correlation matrix including all numeric variables.  5-8 are categorical data not numeric.
corData <- train[ -c(5:8) ]
head(corData)
corMatrix <- cor(corData, use="complete.obs", method="pearson")
corrplot(corMatrix, type = "upper", order = "hclust", col = c("black", "white"),
         bg = "lightblue", tl.col = "black")
#Create the model with some of the most highly correlated variables as displayed in the above graph.
model.lr2 <- lm(Claim ~ Building.Dimension + Customer.Id + NumberOfWindows + Residential +
                  Insured_Period + Geo_Code, train)
summary(model.lr2)
#Measure and display the model performance.
#Measure performance. Adj R-squared - higher is better, AIC, BIC - lower the better
aic <- AIC(model.lr2)
bic <- BIC(model.lr2)
# mean Prediction Error and Std Error - lower is better
pred.model.lr2 <- predict(model.lr2, newdata = test) # validation predictions

meanPred <- mean((test$Claim - pred.model.lr2 )^2) # mean prediction error
stdError <- sd((test$Claim - pred.model.lr2 )^2)/length(test$Claim) # std error
# Append a row to our modelMetrics
modelMetrics[nrow(modelMetrics) + 1, ] <- c( "model.lr2", 0.2353, aic, bic, meanPred, stdError)
modelMetrics
plot(full_df)
#MODEL 4: BEST SUBSETS LINEAR REGRESSION
model.bestSub = regsubsets(Claim ~ ., train, nvmax =25)
summary(model.bestSub)
reg.summary = summary(model.bestSub)
which.min (reg.summary$bic )
which.max (reg.summary$adjr2 ) # just for fun

#Plot the variable bic values by number of variables
plot(reg.summary$bic ,xlab=" Number of Variables ",ylab=" BIC",type="l")
points(6, reg.summary$bic [6], col =" red",cex =2, pch =20)

coef(model.bestSub, 6)

bestSubModel = lm(Claim ~ Insured_Period + Residential + Building_Painted + Building_Fenced +
                    Building_Type + Geo_Code, data=train)
summary(bestSubModel)

#Adj R-squared - higher is better, AIC, BIC - lower the better
aic <- AIC(bestSubModel)
bic <- BIC(bestSubModel)
# mean Prediction Error and Std Error - lower is better
pred.bestSubModel <- predict(bestSubModel, newdata = test) # validation predictions
meanPred <- mean((test$Claim - pred.bestSubModel )^2) # mean prediction error
stdError <- sd((test$Claim - pred.bestSubModel )^2)/length(test$Claim) # std error

# Append a row to our modelMetrics
modelMetrics[nrow(modelMetrics) + 1, ] <- c( "bestSubModel", 0.1789, aic, bic, meanPred, stdError)
modelMetrics

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
plot(test$Claim, type ='l', lty = 1.8, col = 'red')
lines(predictions, type = 'l', lty = 1.8, col = 'green')



#some of the highest correlated are:
#numProj, lastEvalLog, timeCmpy, left, greatEvalLowSat
#Ensure data characteristics are roughly the same
summary(train)
summary(test)
summary(trainingOutcomes)
summary(testOutcomes)
(70/100)*10229 #[1] 7160.3
#A 70/30 split seems reasonable. 70% is 7160.37 cases, so the first 7160.3 cases are reserved for
#training and the remaining 3069 cases will be used for testing. In addition, we need to separate
#out the Insurance observations (first 13 columns) from the diagnoses (last column). The data is
#splitted as follows:
#trainingSet <- full_df[1:7160, 1:13]
#testSet <- full_df[7161:10229, 1:13]
#Similarly, we need to split the Target/Output/Outcome variable into training and test outcome
#sets: (Target/Output/Outcome variable: Claim Variable)
#trainingOutcomes <- full_df[1:7160, 14]
#testOutcomes <- full_df[7161:10229, 14]

#Finally we are ready to create the model. To do this, load the classification package (library)
#and run a k-nearest neighbor classification (knn) on the training set and training outcomes.
#The test data set is also passed in to allow us to evaluate the effectiveness of the model.
#We choose the number of neighboring data points to be considered in the analysis (i.e. k) to be
#85 as that's the square root of the number of training examples (7308). k should be an odd number
#to avoid "tie-breaker" situations.

library(class)
predictions <- knn(train = trainingSet, cl = trainingOutcomes, k = 85,
                   test = testSet)
#KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn(train = train, cl = trainingOutcomes, k = 85, test = test) #To build the model on the training set

X_new = np.array([[5, 2.9, 1, 0.2]]) #Making Predictions (The X_new must be same dim with )
prediction = knn.predict(test)

y_pred = knn.predict(X_test) #Evaluating the Model
(np.mean(y_pred == y_test)) #OR using the score method of the knn object,
#which will compute the test set accuracy for us:
(knn.score(X_test, y_test))
#The output of the classification is a set of predictions for the 205 test cases. Enter the
#predictions variable in R to view these (excerpted):
predictions
#[1] malignant benign    benign    benign    benign
#Evaluating the predictive power of the model
#While we could manually compare the predictions to the known outcomes of the test cases,
#you won't be surprised to learn that R can do this for us-via a cross-tabulation:
table(testOutcomes, predictions)
#              predictions
#testOutcomes    0    1
#0             2329   0
#1             740    0
##The table tells us that all 2329 Zeros(No Claim) testOutcomes cases were predicted correctly, as were all
##740 Ones(Claim)
#cases-i.e. the model had a perfect score on the test data. If there had been any inaccurate
#predictions they would have been shown in the top-right or bottom-left cells (both 0 in this
#example).

#testOutcomes benign malignant
#enign       160         0
#malignant      0        45
#The table tells us that all 160 benign cases were predicted correctly, as were all 45 malignant
#cases-i.e. the model had a perfect score on the test data. If there had been any inaccurate
#predictions they would have been shown in the top-right or bottom-left cells (both 0 in this
#example).

#Summary
#That's all there is to building a predictive model in R. If you want to predict the diagnoses
#for new cases, just pass them to the knn function as the test set and the predicted diagnoses
#will be returned, e.g.:
knn(train = trainingSet, cl = trainingOutcomes, k = 21, test = testSet)
#[1] malignant
attributes(.Last.value)




mod_1 <- lm(Claim ~ Insured_Period + NumberOfWindows + Building.Dimension +
              Settlement + Residential+ Building_Fenced + Building_Painted, data = full_df)
summary(mod_1)
mod_1$fitted.values
mod_1$coefficients
mod_1$residuals
mod_1$effects
mod_1$rank
mod_1$assign
mod_1$qr
mod_1$df.residual
mod_1$xlevels
mod_1$call
mod_1$terms
mod_1$model
tail(mod_1$model, n=500)


#We can then use the complete.cases function to identify the rows without missing data:
full_df_complete <- full_df[complete.cases(full_df),]

library(missForest)
full_mis <- prodNA(full_df, noNA = 0.1)
summary(full_mis)
library(mice)
md.pattern(full_mis)
library(VIM)
mice_plot <- aggr(full_mis, col=c('navyblue', 'yellow'),numbers=TRUE, sortVars=TRUE,
                  labels=names(full_mis), cex.axis=.7, gap=3, ylab=c("Missing data", "Pattern"))
#Imputing the missing values using MICE
#Imputing the missing values using MICE
imputed_data <- mice(full_mis, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_data)
summary(imputed_data$imp$Claim)
imputed_data$imp$Customer.Id
imputed_data$imp$Claim
imputed_data$imp$Insured_Period
imputed_data$imp$Residential
imputed_data$imp$Building_Painted
imputed_data$imp$Building_Fenced
imputed_data$imp$Settlement
imputed_data$imp$Garden
complete(imputed_data, 2)
complete(imputed_data, 1)
complete(imputed_data, 5)
fit <- with(data = fill_mis, exp = lm(Claim ~ Insured_Period + Settlement))
fit <- with(data = full_mis, exp = lm(Claim ~ Insured_Period + Settlement))
summary(fit)
combine <- pool(fit)
summary(combine)
pool(fit)
subset(full_mis, select = -c(Building_Painted, Building_Fenced, Garden, Settlement))
#Removing categorical variable data from the dataset
full_mis_noncat <- subset(full_mis, select = -c(Building_Painted, Building_Fenced, Garden, Settlement))
#
md.pattern(full_mis_concat)
full_mis_noncat <- subset(full_mis, select = -c(Building_Painted, Building_Fenced, Garden, Settlement))
#md.pattern returns a tabualar form of missing values present in each variables
md.pattern(full_mis_concat)
#
md.pattern(full_mis_noncat)
mice_plot <- aggr(full_mis_noncat, col=c('navyblue', 'yellow'),numbers=TRUE, sortVars=TRUE,
                  labels=names(full_mis_noncat), cex.axis=.7, gap=3, ylab=c("Missing data", "Pattern"))
#
#Imputing the missing values using MICE
imputed_data <- mice(full_mis, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_data)
#Checking imputed values
imputed_data$imp$Customer.Id
imputed_data$imp$Claim
imputed_data$imp$Insured_Period
imputed_data$imp$Residential
imputed_data$imp$Building_Painted
imputed_data$imp$Building_Fenced
imputed_data$imp$Settlement
imputed_data$imp$Garden
#Get complete data
#e.g 2nd out of 5
complete_data <- complete(imputed_data, 2)
#All can be done s=using the with() command, use pool() command, thus obtaining a
#consolidated output
#Build predictive model
fit <- with(data = full_mis, exp = lm(Claim ~ Insured_Period + Settlement))
summary(fit)
#Combine result of all 5 models
combine <- pool(fit)
summary(combine)
imputed_data
summary(mice_plot)
complete_data
full_mis_noncat
#Imputing the missing values using MICE
imputed_data <- mice(full_mis_noncat, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_data)
imputed_data$imp$Customer.Id
imputed_data$imp$Claim
imputed_data$imp$Insured_Period
imputed_data$imp$Residential
imputed_data$imp$Building_Painted
imputed_data$imp$Building_Fenced
imputed_data$imp$Settlement
imputed_data$imp$Garden
complete_data
#e.g 2nd out of 5
complete_data <- complete(imputed_data, 2)
complete_data
fit <- with(data = full_mis, exp = lm(Claim ~ Insured_Period + Settlement))
summary(fit)

#The tidymodels framework is a collection of packages for modeling
#and machine learning using tidyverse principles.
#Install tidymodels with:
install.packages("Rtools")
install.packages("tidymodels")
library(tidymodels)

install.packages("reticulate")
library(reticulate)
library(reticulate)
py_install("scipy")
install_scipy <- function(method = "auto", conda = "auto") {
  reticulate::py_install("scipy", method = method, conda = conda)
}
