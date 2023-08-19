import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd


warnings.filterwarnings('ignore', category=ConvergenceWarning)


def main(dataset, pred):
    optimal_logistic = find_optimal_logistic(dataset, 60, pred)
    best_accuracy = optimal_logistic[0]
    best_params = optimal_logistic

    # print(optimal_logistic)

# Continuous Models

# --------------------------Linear Regression--------------------------------

def run_linear_regression(data: pd.DataFrame, target_column: int, predictor_columns: list):
    print(data.iloc[:,target_column])
    
# ---------------------End of Linear Regression------------------------------

# Discrete Models

# --------------------------Logistic Regression--------------------------------
def find_optimal_logistic(dataset, target, predictors):
    best_accuracy = 0
    best_params = []
    for i in range (50):
        output = run_logistic_regression(dataset, target, predictors)
        if(output[0] > best_accuracy):
            best_accuracy = output[0]
            best_params = output
    return best_params
        
    
def run_logistic_regression(data: pd.DataFrame, target_column: int, predictor_columns: list):
    target_dataframe = data.iloc[:,target_column]
    predictors = []
    for i in predictor_columns:
        predictors.append(data.iloc[:,i])
    
    predictor_dataframe = pd.DataFrame(predictors).transpose()

    X_train, X_test, y_train, y_test = train_test_split(predictor_dataframe, target_dataframe, test_size=0.2)
    accuracy = 0
    optimal_c = .0001
    optimal_p = 'l2'
    optimal_s = 'lbfgs'
    c = .0001
    while c < 100000:
        model = LogisticRegression(C=c, max_iter=10000)
        try: 
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            temp_accuracy = accuracy_score(y_test, y_pred)
            if temp_accuracy > accuracy:
                accuracy = temp_accuracy
                optimal_c = c
        except:
            pass
        c *= 10
    solvers = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
    penalties = ['l1', 'l2', 'elasticnet']
    for s in solvers:
        for p in penalties:
            model = LogisticRegression(C=c, max_iter=optimal_c, penalty = p, solver=s)
            try: 
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                temp_accuracy = accuracy_score(y_test, y_pred)
                if temp_accuracy > accuracy:
                    accuracy = temp_accuracy
                    optimal_s = s
                    optimal_p = p
            except:
                pass
    model = LogisticRegression(C=c, max_iter=optimal_c)
    return [accuracy, optimal_c, optimal_s, optimal_p]

    # ---------------------End of Logistic Regression------------------------------

pred = []
for i in range(60):
    pred.append(i)
main(pd.read_csv("Data/sonar.csv"), pred)
