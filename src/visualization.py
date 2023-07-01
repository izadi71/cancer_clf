import os

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import FIGURES_PATH


def plot_roc_curve(fpr, tpr, name, figpath=FIGURES_PATH):
    logger.info(f'Plotting ROC curve for {name}')
    sns.set_style('whitegrid')
    sns.lineplot(x=fpr, y=tpr, color='orange', label='ROC')
    sns.lineplot(x=[0, 1], y=[0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name}')
    plt.legend()
    
    # Save the figure
    name_path = name.replace(' ', '_')
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    plt.savefig(f'{figpath}/roc_curve_{name_path}.png')
    plt.close()

def plot_precision_recall_curve(precision, recall, name, figpath=FIGURES_PATH):
    logger.info(f'Plotting precision-recall curve for {name}')
    sns.set_style('whitegrid')
    sns.lineplot(x=recall, y=precision, color='orange', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {name}')
    plt.legend()

    # Save the figure
    name_path = name.replace(' ', '_')
    if not os.path.exists(figpath):
        os.makedirs(FIGURES_PATH)
    plt.savefig(f'{FIGURES_PATH}/precision_recall_curve_{name_path}.png')
    plt.close()

def visualize_evaluation_metrics(X_train, X_test, y_train, y_test, classifiers, best_params, feature_selection, pca=PCA(), figpath=FIGURES_PATH):
    for name,(classifier, _) in tqdm(classifiers.items()):
        logger.info(f'Visualizing evaluation metrics for {name}')
        
        # Create a pipeline with feature selection and the classifier using the best hyperparameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', feature_selection),
            ('pca', pca),
            ('classifier', classifier)
        ])
        
        pipeline.set_params(**best_params[name])
       
        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)
        
        # Make predictions on the test data using the best hyperparameters
        y_pred = pipeline.predict(X_test)
        
        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot the confusion matrix
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title(f'Confusion Matrix for {name}')

        # Save the figure
        name_path = name.replace(' ', '_')
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        plt.savefig(f'{figpath}/confusion_matrix_{name_path}.png')
        plt.close()
        
        # Print the classification report
        logger.info(f'Classification Report for {name}:')
        clf_rep = classification_report(y_test, y_pred)
        logger.info(f'{classification_report(y_test, y_pred)}')
        
        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:,1])
        
        # Plot the ROC curve
        plot_roc_curve(fpr, tpr, name)
        
        # Calculate the precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, pipeline.predict_proba(X_test)[:,1])
        
        # Plot the precision-recall curve
        plot_precision_recall_curve(precision, recall, name)
        
    return clf_rep