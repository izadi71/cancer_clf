import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from src.config import DATA_PATH, TEST_SIZE

def load_data(path=DATA_PATH):
    # Load the dataset
    logger.info(f'Loading data from {path}')
    raw_cancer = pd.read_csv(path)
    cancer = raw_cancer.copy()
    return cancer

def split_data(cancer):
    # Split the dataset into features and target variable
    logger.info('Splitting data into features and target variable')
    X = cancer.drop(columns=['id', 'diagnosis']).values
    y = cancer.diagnosis.values

    # Encode the target variable as binary labels
    logger.info('Encoding target variable as binary labels')
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the dataset into training and test sets
    logger.info('Splitting data into training and test sets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
    
    return X_train, X_test, y_train, y_test, X, y
