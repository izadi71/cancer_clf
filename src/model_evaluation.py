import time

import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def evaluate_classifiers(X_train, X_test, y_train, y_test, classifiers, best_params, feature_selection, pca=PCA()):
    # Create a dictionary to store the evaluation metrics for each classifier
    logger.info('Evaluating classifiers')
    metrics = {}
    
    for name, (classifier, _) in tqdm(classifiers.items()):
        
        # Create a pipeline with feature selection and the classifier using the best hyperparameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', feature_selection),
            ('pca', pca),
            ('classifier', classifier)
        ])
        
        
        pipeline.set_params(**best_params[name])
        
        start_time = time.time()
        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        
        # Make predictions on the test data using the best hyperparameters
        y_pred = pipeline.predict(X_test)
        
        # Calculate the evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store the evaluation metrics for the classifier
        metrics[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Training Time': end_time - start_time
        }
    
    return metrics



def display_metrics(metrics):
    # Display the evaluation metrics for each classifier
    logger.info('Displaying evaluation metrics')
    # for name, metric in metrics.items():
    #     print(f'{name}:')
    #     for key,value in metric.items():
    #         print(f'  {key}: {value:.4f}')
    
    metrics_df = pd.DataFrame(metrics).T
    logger.info(f'\n {metrics_df}')
    
    return metrics_df
            
def cross_validate_classifiers(X, y, classifiers, best_params, feature_selection, pca=PCA()):
    # Create a dictionary to store the cross-validation scores for each classifier
    logger.info('Cross-validating classifiers')
    cross_val_scores = {}
    
    for name, (classifier, _) in tqdm(classifiers.items()):
        
        # Create a pipeline with feature selection and the classifier using the best hyperparameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', feature_selection),
            ('pca', pca),
            ('classifier', classifier)
        ])
        
        pipeline.set_params(**best_params[name])
        
        # Perform 10-fold cross-validation on the dataset using the best hyperparameters
        scores = cross_val_score(pipeline, X, y, cv=10)
        
        # Store the cross-validation scores for the classifier
        cross_val_scores[name] = scores
    
    return cross_val_scores

def display_cross_val_scores(cross_val_scores):
    # Display the cross-validation scores for each classifier
    logger.info('Displaying cross-validation scores')
    # for name,scores in cross_val_scores.items():
    #     print(f'{name}: {scores.mean():.4f} (+/- {scores.std():.4f})')
    
    cross_val_df = pd.DataFrame(cross_val_scores).T
    logger.info(f'\n {cross_val_df}')

    return cross_val_df