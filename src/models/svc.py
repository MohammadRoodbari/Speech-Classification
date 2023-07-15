from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score

class SVMClassifier:
    """
    A wrapper class for Support Vector Machine (SVM) classification using scikit-learn.

    Parameters:
    ----------
        kernel (str, optional): Specifies the kernel type to be used in the SVM algorithm.
            Supported options are 'linear', 'rbf' (default), 'poly', 'sigmoid'.
        C (float, optional): Regularization parameter. The strength of the regularization is inversely
            proportional to C. Must be strictly positive. Defaults to 1.0.
        gamma (str or float, optional): Kernel coefficient. If 'scale' (default), it uses 1 / (n_features * X.var())
            as value of gamma. If 'auto', uses 1 / n_features. If a float, it specifies the value directly.

    Methods:
    ----------
        train(X_train, y_train):
            Trains the SVM model with the provided training data.

        predict(X_test):
            Predicts the labels for the given test data using the trained SVM model.

    Example usage:
    ----------
        # Assuming you have your features (X) and labels (y) ready
        svm = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        svm.train(X_train, y_train)
        predictions = svm.predict(X_test)
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, X_train, y_train):
        """
        Trains the SVM model with the provided training data.

        Parameters:
            X_train  The input training samples.
            y_train  The target values (class labels) for the training samples.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the labels for the given test data using the trained SVM model.

        Parameters:
            X_test : The input test samples.

        Returns:
            The predicted labels for the test samples.
        """
        return self.model.predict(X_test)

class Evaluation:

    def __init__(self, y_actual, y_predicted):
        self.y_actual = y_actual
        self.y_predicted = y_predicted

    def accuracy(self):
        return accuracy_score(self.y_actual, self.y_predicted)

    def precision(self):
        return precision_score(self.y_actual, self.y_predicted)

    def f1(self):
        return f1_score(self.y_actual, self.y_predicted)

    def ConfusionMatrix(self):
        return confusion_matrix(self.y_actual, self.y_predicted)

