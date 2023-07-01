from loguru import logger
from sklearn.decomposition import PCA
from src.config import classifiers
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def define_classifiers(classifiers=classifiers):
    # Define the classifiers and their hyperparameters to tune
    logger.info('Defining classifiers and their hyperparameters')
    return classifiers

def train_classifiers(X_train, y_train, classifiers, feature_selection):
    # Create a dictionary to store the best hyperparameters for each classifier
    logger.info('Training classifiers and finding best hyperparameters')
    best_params = {}
    
    # Define the PCA method
    pca = PCA()
    
    for name, (classifier, params) in tqdm(classifiers.items()):
        
        # Create a pipeline with feature selection and the classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', feature_selection),
            ('pca', pca),
            ('classifier', classifier)
        ])
        
        # Define the hyperparameters to tune for the pipeline
        params.update({
            'feature_selection__k':[7, 8, 9, 10],
            'pca__n_components':[3, 4, 5, 6, 7]
        })
        
        # Perform a grid search to find the best hyperparameters for the pipeline
        grid_search = GridSearchCV(pipeline, params)
        
        grid_search.fit(X_train, y_train)
        
        # Store the best hyperparameters for the classifier
        best_params[name] = grid_search.best_params_
    
    return best_params
