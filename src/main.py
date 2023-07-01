from src.data_preprocessing import load_data, split_data
from src.feature_selection import select_features
from src.model_evaluation import (cross_validate_classifiers, display_cross_val_scores,
                              display_metrics, evaluate_classifiers)
from src.model_training import define_classifiers, train_classifiers
from src.visualization import visualize_evaluation_metrics
from loguru import logger
from src.generate_report import generate_report

def main():
    logger.info('Loading data')
    cancer = load_data()
    
    logger.info('Splitting data')
    X_train, X_test, y_train, y_test, X, y = split_data(cancer)

    logger.info('Selecting features')
    feature_selection = select_features()

    logger.info('Defining classifiers')
    classifiers = define_classifiers()
    
    logger.info('Training classifiers')
    best_params = train_classifiers(X_train, y_train, classifiers, feature_selection)

    logger.info('Evaluating classifiers')
    metrics = evaluate_classifiers(X_train, X_test, y_train, y_test, classifiers, best_params, feature_selection)
    
    logger.info('Displaying evaluation metrics')
    display_metrics(metrics)

    logger.info('Cross-validating classifiers')
    cross_val_scores = cross_validate_classifiers(X, y, classifiers, best_params, feature_selection)
    
    logger.info('Displaying cross-validation scores')
    display_cross_val_scores(cross_val_scores)

    logger.info('Visualizing evaluation metrics')
    visualize_evaluation_metrics(X_train, X_test, y_train, y_test, classifiers, best_params, feature_selection)
    
    generate_report(classifiers, best_params, metrics, cross_val_scores)
    logger.info('Report generated')
    
    logger.info('Successfully finished comparison')
    
if __name__ == '__main__':
    main()
