import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        

        self._X = X
        self._y = one_hot_encoding(y)

        np.random.seed(42)
        n_features = self._X.shape[1]
        n_outputs = self._y.shape[1]


        # Random initialization of weights
        self._h_weights = np.random.randn(n_features, self.n_hidden) * 0.01
        self._o_weights = np.random.randn(self.n_hidden, n_outputs) * 0.01
        self._h_bias = np.zeros((1, self.n_hidden))
        self._o_bias = np.zeros((1, n_outputs))


    def fit(self, X, y):
        

        self._initialize(X, y)
        n_features = self._X.shape[1]
        n_samples = self._X.shape[0]
        for i in range(self.n_iterations):

            # Forward propagation
            z1 = np.dot(self._X,self._h_weights) + self._h_bias
            a1 = self.hidden_activation(z1)
            z2 = np.dot(a1, self._o_weights) + self._o_bias
            a_softmax = self._output_activation(z2)

            # Calculate loss
            loss = self._loss_function(self._y, a_softmax)
            # Store loss in loss history for every 20 iterations.
            if i % 20 == 0:
                self._loss_history.append(loss)


            # Backward propagation
            dz2 = a_softmax - self._y
            dw2 = np.dot(a1.T, dz2)/n_samples
            db2 = np.sum(dz2, axis = 0, keepdims = True)/n_samples
            dz1 = np.dot(dz2, self._o_weights.T)*self.hidden_activation(z1, derivative = True)
            dw1 = np.dot(self._X.T, dz1)
            db1 = np.sum(dz1, axis = 0, keepdims = True)/n_samples

            # Updating the weights and biases
            self._h_weights = self._h_weights - self.learning_rate*dw1
            self._h_bias = self._h_bias - self.learning_rate*db1
            self._o_weights = self._o_weights - self.learning_rate * dw2
            self._o_bias = self._o_bias - self.learning_rate * db2




    def predict(self, X):
        

        X_test = X
        predictions = np.zeros(X_test.shape[0])

        # While predicting, we already have the trained weights and biases, so we just perform 1 step of forward
        # propagation and predict which class the test sample belongs to.
        z1 = np.dot(X_test, self._h_weights) + self._h_bias
        a1 = self.hidden_activation(z1)
        z2 = np.dot(a1, self._o_weights) + self._o_bias
        a_softmax = self._output_activation(z2)

        for i,prediction in enumerate(a_softmax):
            predictions[i] = np.argmax(prediction)

        return predictions

        
