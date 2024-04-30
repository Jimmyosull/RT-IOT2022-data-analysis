""" Code based on https://towardsdatascience.com/logistic-regression-from-scratch-69db4f587e17 """
import numpy as np
from numpy.random import rand
from numpy  import dot, e, log

class LogisticReg:


    def sigmoid(self, z):
        sig = 1 / (1 + np.e**(-z))
        return np.nan_to_num(sig, True, 0)

    def cost_fn(self, X, y, weights):
        z = np.dot(X, weights)
        predicted_1 = y * np.log(1 - self.sigmoid(z))
        predicted_0 = (1 - y) * np.log(1 - self.sigmoid(z))
        return sum(predicted_1 + predicted_0) / len(X)
    
    def fit(self, X, y, epochs = 25, lr = .05):
        loss = []
        weights = rand(X.shape[1])
        N = len(X)

        # grad descent
        for _ in range(epochs):
            y_hat = self.sigmoid(np.dot(X, weights))
            weights -= lr * np.dot(X.T, y_hat - y) / N
            loss.append(self.cost_fn(X, y ,weights))
        
        self.weights = weights
        self.loss = loss

    def predict(self, X):
        z = np.dot(X, self.weights)
        return [1 if i > .5 else 0 for i in self.sigmoid(z)]
