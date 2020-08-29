# logistic-insurance-prediction
I used logistic regression model in R programming to predict if there will be a claim or no claim for an insurance company.
The details of the project is stated below:

#Zindi Project from DSN ML Open Competition
##Data Science Nigeria 2019 Challenge #1: Insurance Prediction
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
