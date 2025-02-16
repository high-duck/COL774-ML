# Imports - you can add any other permitted libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    def __init__(self):
        """uncomment all before running"""
        dfX = pd.read_csv("../data/Q3/logisticX.csv")
        np_x = dfX.to_numpy()
        dfY = pd.read_csv("../data/Q3/logisticY.csv")
        np_y = dfY.to_numpy()
        np_y = np.reshape(np_y , (-1 ,1))
        self.fit(np_x, np_y)
        self.param_vec = None
    
    def nomarlize(self , X):
        mean = X.mean(axis = 0)
        std = X.std(axis=0)
        X = (X - mean) / std

        return X

    def sigmoid(self , param_vec , x):
        val = np.dot(param_vec , np.append(x , 1))
        f = np.exp(-1.0 * val)
        f = 1/(1 + f)
        return f

    def plot(self, X, y, param_vec):
        X1_0, X2_0 = [], []
        X1_1, X2_1 = [], []
        

        for i in range(len(X)):
            x1, x2 = X[i][0], X[i][1]
            if y[i][0] == 0:
                X1_0.append(x1)
                X2_0.append(x2)
            else:
                X1_1.append(x1)
                X2_1.append(x2)
        

        fig, ax = plt.subplots()
        

        ax.scatter(X1_0, X2_0, color='blue', marker='o', label="Class 0")
        ax.scatter(X1_1, X2_1, color='orange', marker='^', label="Class 1")


        X1_range = np.linspace(-3, 3, 100)
        X2_range = -(param_vec[0] / param_vec[1]) * X1_range - (param_vec[2] / param_vec[1])
        ax.plot(X1_range, X2_range, color="black", linewidth=2, label="Decision Boundary")

        ax.set_xlabel(r"x$1$", fontweight="bold", fontsize=12)
        ax.set_ylabel(r"x$2$", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.show()

    def fit(self, X, y, learning_rate=0.01):
        
        X = self.nomarlize(X)
        param_vec = np.zeros(len(X[0]) + 1)
        while(True):
            grad_vec = np.zeros_like(param_vec)
            for i in range(len(X)):
                err = y[i] - self.sigmoid(param_vec , X[i])
                grad_vec += err * np.append(X[i] , 1)
            hessian = np.zeros((len(X[0]) + 1 , len(X[0]) + 1))
            
            for i in range(len(X)):
                U = np.append(X[i] , 1)
                predicted = self.sigmoid(param_vec , X[i])
                for j in range(len(X[0]) + 1):
                    for k in range(len(X[0]) + 1):
                        hessian[j][k] += U[j] * U[k] * predicted * (1.0 - predicted)

            param_vec += np.dot(np.linalg.inv(hessian) , grad_vec.T)

            if(np.dot(grad_vec , grad_vec) < 0.0005):
                break
        
        #plotting code
        self.plot(X , y ,param_vec)

        ret_vec = np.roll(param_vec , 1)  
        self.param_vec = param_vec
        #print(ret_vec)
        self.predict(X)
        return ret_vec
    
    def predict(self, X):
        Y = []
        for x in X:
            y = self.sigmoid(self.param_vec , x)
            if y > 0.5:
                Y.append(1)
            else:
                Y.append(0)
        return np.array(Y)

# slgr = LogisticRegressor()
