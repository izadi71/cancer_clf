import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
            
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # Calculate the distances between the test points and the training points
        distances = cdist(X, self.X_train)
        
        # Find the indices of the k nearest neighbors
        indices = np.argsort(distances)[:, :self.n_neighbors]
        
        # Find the labels of the k nearest neighbors
        labels = self.y_train[indices]
        
        # Find the most common label among the k nearest neighbors
        y_pred = mode(labels, axis=1)[0].ravel()
        
        return y_pred
    
    def predict_proba(self, X):
        # Calculate the distances between the test points and the training points
        distances = cdist(X, self.X_train)
        
        # Find the indices of the k nearest neighbors
        indices = np.argsort(distances)[:, :self.n_neighbors]
        
        # Find the labels of the k nearest neighbors
        labels = self.y_train[indices]
        
        # Calculate the class probabilities for each test point
        p = np.mean(labels == 1, axis=1)
        
        # Return the class probabilities
        return np.vstack([1 - p, p]).T
    
    # calculate the accuracy of the classifier
    def score(self, X, y):
            y_pred = self.predict(X)
            
            # Calculate the accuracy of the predictions
            accuracy = accuracy_score(y, y_pred)
            
            return accuracy
        
        
        
class NaiveBayes:
    def fit(self, X, y):
        
        self.X_train = X
        self.y_train = y
        # Calculate the prior probabilities for each class
        self.priors = np.bincount(self.y_train) / len(self.y_train)
        
        # Calculate the mean and standard deviation for each feature for each class
        self.means = np.array([self.X_train[self.y_train == c].mean(axis=0) for c in range(len(self.priors))])
        self.stds = np.array([self.X_train[self.y_train == c].std(axis=0) for c in range(len(self.priors))])
    
    def predict(self, X):
        # Calculate the likelihoods for each feature for each class
        likelihoods = -0.5 * np.log(2 * np.pi * self.stds ** 2) - 0.5 * ((X[:, np.newaxis] - self.means) / self.stds) ** 2
        
        # Sum the log-likelihoods for all features for each class
        log_likelihoods = likelihoods.sum(axis=2)
        
        # Add the log-prior probabilities to the log-likelihoods
        log_posteriors = log_likelihoods + np.log(self.priors)
        
        # Find the class with the highest log-posterior probability
        y_pred = np.argmax(log_posteriors, axis=1)
        
        return y_pred
    
    def predict_proba(self, X):
        # Calculate the likelihoods for each feature for each class
        likelihoods = -0.5 * np.log(2 * np.pi * self.stds ** 2) - 0.5 * ((X[:, np.newaxis] - self.means) / self.stds) ** 2
        
        # Sum the log-likelihoods for all features for each class
        log_likelihoods = likelihoods.sum(axis=2)
        
        # Add the log-prior probabilities to the log-likelihoods
        log_posteriors = log_likelihoods + np.log(self.priors)
        
        # Calculate the posterior probabilities for each class
        posteriors = np.exp(log_posteriors)
        posteriors /= posteriors.sum(axis=1, keepdims=True)
        
        return posteriors
    
    def score(self, X, y):
        y_pred = self.predict(X)
        
        # Calculate the accuracy of the predictions
        accuracy = accuracy_score(y, y_pred)
        
        return accuracy



# class PCA:
#     def __init__(self, n_components=None):
#         self.n_components = n_components
    
#     def fit(self, X):
#         # Center the data
#         X_centered = X - X.mean(axis=0)
        
#         # Calculate the covariance matrix
#         cov_matrix = np.cov(X_centered.T)
        
#         # Calculate the eigenvalues and eigenvectors of the covariance matrix
#         eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
#         # Sort the eigenvalues and eigenvectors in descending order of eigenvalue magnitude
#         indices = np.argsort(eigenvalues)[::-1]
#         eigenvalues = eigenvalues[indices]
#         eigenvectors = eigenvectors[:, indices]
        
#         # Select the first k eigenvectors as the principal components
#         if self.n_components is not None:
#             eigenvectors = eigenvectors[:, :self.n_components]
        
#         self.components_ = eigenvectors.T
    
#     def transform(self, X):
#         # Project the data onto the principal components
#         X_transformed = X.dot(self.components_.T)
        
#         return X_transformed
