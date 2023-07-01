# Cancer Classification Report

## Evaluation Metrics

### Custom KNN

- Accuracy: 0.9561
- Precision: 0.9773
- Recall: 0.9149
- F1 Score: 0.9451
- Training Time: 0.0020

### Custom NaiveBayes

- Accuracy: 0.9298
- Precision: 0.9149
- Recall: 0.9149
- F1 Score: 0.9149
- Training Time: 0.0024

### Logistic Regression

- Accuracy: 0.9561
- Precision: 1.0000
- Recall: 0.8936
- F1 Score: 0.9438
- Training Time: 0.0049

### Gradient Boosting

- Accuracy: 0.9825
- Precision: 0.9787
- Recall: 0.9787
- F1 Score: 0.9787
- Training Time: 0.0268

### XGBoost

- Accuracy: 0.9561
- Precision: 0.9565
- Recall: 0.9362
- F1 Score: 0.9462
- Training Time: 0.0206

## Cross-Validation Scores

- Custom KNN: 0.9491 (+/- 0.0380)
- Custom NaiveBayes: 0.9332 (+/- 0.0291)
- Logistic Regression: 0.9367 (+/- 0.0274)
- Gradient Boosting: 0.9404 (+/- 0.0353)
- XGBoost: 0.9561 (+/- 0.0335)

## Classification Reports

### Best Params for Custom KNN model

- {'feature_selection__k': 8, 'pca__n_components': 5}

### ROC Curve

### Custom KNN

![ROC Curve for Custom KNN](reports/figures/roc_curve_Custom_KNN.png)

### Precision-Recall Curve

### Custom KNN

![Precision-Recall Curve for Custom KNN](reports/figures/precision_recall_curve_Custom_KNN.png)

### Best Params for Custom NaiveBayes model

- {'feature_selection__k': 7, 'pca__n_components': 3}

### ROC Curve

### Custom NaiveBayes

![ROC Curve for Custom NaiveBayes](reports/figures/roc_curve_Custom_NaiveBayes.png)

### Precision-Recall Curve

### Custom NaiveBayes

![Precision-Recall Curve for Custom NaiveBayes](reports/figures/precision_recall_curve_Custom_NaiveBayes.png)

### Best Params for Logistic Regression model

- {'classifier__C': 0.02, 'classifier__penalty': 'l2', 'feature_selection__k': 10, 'pca__n_components': 4}

### ROC Curve

### Logistic Regression

![ROC Curve for Logistic Regression](reports/figures/roc_curve_Logistic_Regression.png)

### Precision-Recall Curve

### Logistic Regression

![Precision-Recall Curve for Logistic Regression](reports/figures/precision_recall_curve_Logistic_Regression.png)

### Best Params for Gradient Boosting model

- {'classifier__learning_rate': 1, 'classifier__n_estimators': 20, 'feature_selection__k': 7, 'pca__n_components': 6}

### ROC Curve

### Gradient Boosting

![ROC Curve for Gradient Boosting](reports/figures/roc_curve_Gradient_Boosting.png)

### Precision-Recall Curve

### Gradient Boosting

![Precision-Recall Curve for Gradient Boosting](reports/figures/precision_recall_curve_Gradient_Boosting.png)

### Best Params for XGBoost model

- {'classifier__learning_rate': 0.2, 'classifier__max_depth': 5, 'classifier__n_estimators': 20, 'feature_selection__k': 7, 'pca__n_components': 5}

### ROC Curve

### XGBoost

![ROC Curve for XGBoost](reports/figures/roc_curve_XGBoost.png)

### Precision-Recall Curve

### XGBoost

![Precision-Recall Curve for XGBoost](reports/figures/precision_recall_curve_XGBoost.png)

