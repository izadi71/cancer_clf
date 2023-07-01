from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data import DATA_DIR
from reports import REPORT_DIR
from src.custom_models import KNN, NaiveBayes

# File paths
DATA_PATH = DATA_DIR / 'data_cancer.csv'
FIGURES_PATH = REPORT_DIR / 'figures'
REPORT_PATH = REPORT_DIR / 'report.md'

# Hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 0


# Classifiers
classifiers = {
        'Custom KNN':(KNN(), {}),
        'Custom NaiveBayes':(NaiveBayes(), {}),
        'Logistic Regression': (LogisticRegression(random_state=0), {
            'classifier__C': [0.01, 0.02],
            'classifier__penalty': ['l2']
        }),
        'Decision Tree': (DecisionTreeClassifier(random_state=0), {
            'classifier__max_depth': [3, 5],
            'classifier__min_samples_split': [2, 3],
            'classifier__min_samples_leaf': [1, 2]
        }),
        'Random Forest': (RandomForestClassifier(random_state=0), {
            'classifier__n_estimators': [10, 20],
            'classifier__max_depth': [3, 5],
            'classifier__min_samples_split': [2, 3],
            'classifier__min_samples_leaf': [1, 2]
        }),
        'SVM': (SVC(random_state=0, probability=True), {
            'classifier__C': [0.01, 0.1, 1],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__degree':[3, 5]
        }),
        'Naive Bayes': (GaussianNB(), {}),
        'KNN': (KNeighborsClassifier(), {
            'classifier__n_neighbors': [3, 5],
            'classifier__weights':['uniform']
        }),
        'AdaBoost': (AdaBoostClassifier(random_state=0), {
            'classifier__n_estimators': [10, 20],
            'classifier__learning_rate':[1, 2]
        }),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=0), {
            'classifier__n_estimators': [10, 20],
            'classifier__learning_rate':[1, 2]
        }),
          'XGBoost': (XGBClassifier(random_state=0), {
            'classifier__n_estimators':[10, 20],
            'classifier__learning_rate':[0.1, 0.2],
            'classifier__max_depth':[3, 5]
        }),
    }
    
