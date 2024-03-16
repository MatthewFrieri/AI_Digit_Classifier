import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                   PREPARING DATA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import training data from KAGGLE in pandas format
data = pd.read_csv("train.csv")

# Convert pandas data to numpy format
data = np.array(data)

# Get size of dataset 
# m = amount of rows (images)
# n = amount of pixels (+ the label)
m, n = data.shape
np.random.shuffle(data)

# Takes the first 1000 images and transposes them so a collumn is an image
data_dev = data[0:1000].T    # Data to save for later
Y_dev = data_dev[0]          # Labels
X_dev = data_dev[1:n]        # Pixel values
X_dev = X_dev / 255.

# Takes the rest of the images and transposes them so a collumn is an image
data_train = data[1000:m].T  # Training data
Y_train = data_train[0]      # Labels
X_train = data_train[1:n]    # Pixel values
X_train = X_train / 255.
_,m_train = X_train.shape

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                       FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def init_params():
    # Initialize weights and biases
    
    W1 = np.random.rand(10, 784) - 0.5    # Creates a 2D array of size 10x784 of random numbers from -0.5 to 0.5
    b1 = np.random.rand(10, 1) - 0.5
    
    W2 = np.random.rand(10, 10) - 0.5     # Creates a 2D array of size 10x10 of random numbers from -0.5 to 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2


def ReLU(Z):
    # Applies the ReLU function to every item in the Z matrix
    # Returns Z if Z>0
    # Returns 0 if Z<0
    return np.maximum(Z, 0)

def softmax(Z):
    # Applies softmax function to every item in the Z matrix
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1   # For every X in the matrix, multiply it by its weight and add its bias
    A1 = ReLU(Z1)         # Apply ReLU to the entire Z1 matrix
    
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2

def one_hot(Y):
    # One hot encodes
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))    # Creates a 2D matrix of 0s where there are Y.size examples (images) and 10 possible outputs (0-9) 
    one_hot_Y[np.arange(Y.size), Y] = 1            # Replaces a 0 with a 1 for each correct output value
    one_hot_Y = one_hot_Y.T                        # Transpose the matrix so that every column is an example
    return one_hot_Y

def deriv_ReLU(Z):
    # Derivative of ReLU function
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Opposite of forward propagation
    # Uses derivatives of previous functions

    one_hot_Y = one_hot(Y)  
    
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # Updates all initial weights and biases
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    # Gets the index of the highest prediction
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # For every example if the predictions is correct it converts the boolean to 1 
    return np.sum(predictions == Y) / Y.size

def gradient_decent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    
    for i in range(iterations+1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Display regular updates
        if i % (iterations//10) == 0:
            print("Iteration:", i)
            predictions = get_predictions(A2)
            print(f"Accuracy: {round(get_accuracy(predictions, Y)*100, 2)}%\n")
            
    return W1, b1, W2, b2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                   RUNNING IT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

W1, b1, W2, b2 = gradient_decent(X_train, Y_train, 300, 0.1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                 DISPLAYING MISTAKES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
    
    
def show_mistakes(m, W1, b1, W2, b2):
    for index in range(m):
        current_image = X_train[:, index, None]
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]
        if prediction != label:
            print("Prediction: ", prediction)
            print("Label: ", label)
            print()

            current_image = current_image.reshape((28, 28)) * 255
            plt.gray()
            plt.imshow(current_image, interpolation='nearest')
            plt.show()

show_mistakes(100, W1, b1, W2, b2)