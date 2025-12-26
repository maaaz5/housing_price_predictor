import numpy as np
import pickle 
import os

class CustomLinearRegression:
  def __init__(self):
    self.theta = None
    self.history = None

  def _add_bias_column(self, X):
    return np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
  
  def _model(self, X, theta):
    return X.dot(theta)

  def _const_function(self, X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((self._model(X,theta) - y)**2)
  
  def _gradient(self, X, y, theta):
    m = len(y)
    return X.T.dot((self._model(X, theta)-y)) / m

  def fit(self,X,y,learning_rate=0.01,n_iterations=1000):
    
    #making sure y is column vector
    if len(y.shape) == 1:
      y = y.reshape(-1,1)
    
    #adding the bias to X
    X_with_bias = self._add_bias_column(X)

    #random theta
    if self.theta is None:
      n_features = X_with_bias.shape[1]
      self.theta = np.random.randn(n_features, 1)
    
    #storing the cost history
    self.history = np.zeros(n_iterations)

    #gradient descent
    for i in range(n_iterations):
      self.theta = self.theta - learning_rate * self._gradient(X_with_bias, y, self.theta)

      self.history[i] = self._const_function(X_with_bias,y,self.theta)

  
  def predict(self,X):
    """
    making predictions using my trained model
    """
    if self.theta is None:
      raise ValueError("Model has not been trained yet, call the fit() function first")

    X_with_bias = self._add_bias_column(X)

    predictions = self._model(X_with_bias,self.theta)

    return predictions
  
  def save_model(self, filepath):
    """ saving the trained model or theta to pickle file"""

    if self.theta is None:
      raise ValueError("No model to save. Train the model first.")
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)


    with open(filepath, 'wb') as f:
      pickle.dump(self.theta, f)


 

  def load_model(self, filepath):
    """ loading my trained model from that pickle file """

    if not os.path.exists(filepath):
      raise FileNotFoundError('Model file was not found')
    
    with open(filepath, 'rb') as f:
      self.theta = pickle.load(f)
    
    return self




  
  