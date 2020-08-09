# Data Science Football Project: Project Overview

- Created a predictive model that answers the question: “what sorts of people were more likely to survive the sinking of the Titanic?”.

- Used passenger data (i.e. name, age, gender, socio-economic class, etc) from Kaggle to develop the model.

- Undertook exploratory data analysis to understand the key relationships and characteristics of the data

- Developed Random Forest, Decision Tree, Logistic Regression and Nave Bayes classifier models

- Tuned and optimised Random Forest and Decision Tree models (best performers) using GridSearchCV

- Submitted results to Kaggle Titanic Competition

## Resources

**Python Version:** 3.7

**Packages:** pandas, numpy, sklearn, matplotlib, seaborn

**Kaggle Competition Page:** https://www.kaggle.com/c/titanic/overview


## EDA

I started with some exploratory data analysis to summarise the main characteristics of the data and find the key relationships. Some highlights are below. (If images don't load see images folder).

![](/images/titanic_lmplot.PNG)

![](/images/titanic_heatmap.PNG)

![](/images/titanic_jointplot.PNG)


## Model Building

First, I transformed the data structure into a dataframe and produced dummy variables where applicable.

I developed three clasification models and used GridSearchCV to optimsie the model parameters, using 'Homogeneity' as the evaluation criteria. 

The models were:
- **Random Forest**
- **Decision Tree**
- **Nave Bayes**
- **Logistic Regression**


## Model Performance

The models performed realtively well after a first pass iteration. 

- **Random Forest:** Homogeneity = 0.23
- **Decision Tree:** Homogeneity = 0.24
- **Nave Bayes:** Homogeneity = 0.25
- **Logistic Regression:** Homogeneity = 0.25
 
 ## Submission to Competition
 
 I submitted the results to the Kaggle Titanic Competition. As of 09/08/20 the results sit in the top 30% of submissions.
 
 



