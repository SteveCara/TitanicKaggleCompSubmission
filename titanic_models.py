import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import homogeneity_score

# import data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# set target value
y = train_data["Survived"]


# set model features (parameters used in model)
features = ["Pclass", "Sex", "SibSp", "Parch"]


# get dummy data (conversion of categorical variables into numerical data)
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


# set up models, with guesses at model parameters
rm_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=1)

dt_model = DecisionTreeClassifier(max_depth=5)

nb_model = GaussianNB()

lr_model = LogisticRegression()

# fit models
rm_model.fit(X, y)

dt_model.fit(X, y)

nb_model.fit(X, y)

lr_model.fit(X, y)

# cross val score to find model errors
rm_cross_val_score = np.mean(cross_val_score(
    rm_model, X, y, scoring='homogeneity_score', cv=100))

dt_cross_val_score = np.mean(cross_val_score(
    dt_model, X, y, scoring='homogeneity_score', cv=100))

nb_cross_val_score = np.mean(cross_val_score(
    nb_model, X, y, scoring='homogeneity_score', cv=100))

lr_model_score = np.mean(cross_val_score(
    nb_model, X, y, scoring='homogeneity_score', cv=100))

# print(rm_cross_val_score, dt_cross_val_score,
#   nb_cross_val_score, lr_model_score)


# # tune rm models using GridSearchCV
# # set parameters to tune model
# parameters = {'n_estimators': range(5, 100, 5), 'criterion': (
#     'gini', 'entropy'), 'max_features': ('auto', 'sqrt', 'log2')}

# gs = GridSearchCV(rm_model, parameters,
#                   scoring='homogeneity_score', cv=3)
# gs.fit(X, y)

# tune dt model using GridSearchCV
# set parameters to tune model

parameters = {'criterion': ('gini', 'entropy'), 'n_estimators': (
    10, 200, 10), 'splitter': ('best', 'random'), 'max_features': ('auto', 'sqrt', 'log2')}

gs_dt = GridSearchCV(rm_model, parameters,
                     scoring='homogeneity_score', cv=3)

gs_dt.fit(X, y)


# # generate predictions
# gs_predictions = gs.best_estimator_.predict(X_test)
# dt_predictions = dt_model.predict(X_test)
# nb_predictions = nb_model.predict(X_test)

gs_dt_predictions = gs_dt.best_estimator_.predict(X_test)

# # test ensembles

# ens_model = VotingClassifier(
#     estimators=[('rm', gs), ('dt', dt_model)], voting='soft', weights=[1, 1])

# ens_model.fit(X, y)
# ens_model_predictions = ens_model.predict(X_test)

# # create output to submit to Kaggle competition
# # rm_output = pd.DataFrame(
# #     {'PassengerId': test_data.PassengerId, 'Survived': gs_predictions})

# # dt_output = pd.DataFrame(
# #     {'PassengerId': test_data.PassengerId, 'Survived': dt_predictions})

# # nb_output = pd.DataFrame(
# #     {'PassengerId': test_data.PassengerId, 'Survived': dt_predictions})

gs_dt_output = pd.DataFrame(
    {'PassengerId': test_data.PassengerId, 'Survived': gs_dt_predictions})

# ens_model.to_csv('my__titanic_submission.csv', index=False)
gs_dt_output.to_csv('my__titanic_submission.csv', index=False)

# print(gs_dt_output.head())
