# Imports - you can add any other permitted libraries
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation, cm 
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    X = []
    Y = []
    rng = np.random.default_rng()
    for i in range(N):
        x1 = rng.normal(loc = input_mean[0] , scale = input_sigma[0])
        x2 = rng.normal(loc = input_mean[1] , scale = input_sigma[1])
        X_vec = np.array([1 , x1 , x2])
        X.append([x1 , x2])
        t = np.dot(X_vec , theta.T)
        y = t + rng.normal(0 , noise_sigma)
        Y.append(y)
    
    return np.array(X) , np.array(Y)
    
def plot_err(err_list):
    ax = plt.axes()
    X = np.arange(len(err_list))
    ax.plot(X[::20] , err_list[::20])
    plt.show()

def plot_param(point_list):
    ax = plt.axes(projection = "3d")
    X = [] 
    Y = []
    Z = []
    for t in range(0 , len(point_list) , 5):
        point = point_list[t]
        X.append(point[0])
        Y.append(point[1])
        Z.append(point[2])
    ax.plot(point_list[-1][0] , point_list[-1][1] , point_list[-1][2] , c="r")
    ax.plot(X , Y , Z , c = "black")
    ax.set_ylabel("\u03B8\u2081")
    ax.set_xlabel("\u03B8\u2080")
    ax.set_zlabel("\u03B8\u2082")
    plt.show()


class StochasticLinearRegressor:
    def __init__(self):
        self.param_vec = None
        pass
    
    def animate_3d(self, point_list):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        point_list = np.array(point_list)
        param1 = point_list[::5, 0]  
        param2 = point_list[::5, 1]  
        err_values = point_list[::5, 2]

        ax.set_xlabel("Parameter 0")
        ax.set_ylabel("Parameter 1")
        ax.set_zlabel("Parameter 2")
        ax.set_title("Gradient Descent Convergence")

        # Plot static surface (optional for better visualization)
        ax.plot(param1, param2, err_values, 'gray', alpha=0.3)  # Light gray path

        # Initialize the animated point
        point, = ax.plot([], [], [], 'bo-', markersize=6)  

        def update(num):
            point.set_data(param1[:num], param2[:num])  # Update x and y data
            point.set_3d_properties(err_values[:num])  # Update z data
            return point,

        ani = animation.FuncAnimation(fig, update, frames=len(point_list), interval=50, blit=False)
        plt.show()
    
    def fit(self, X, y, learning_rate = 0.01):
        batch = 1
        point_list = []
        err_list = []
        param_vec = np.zeros(len(X[0]) + 1 , )
        point_list.append(param_vec.copy())
        epochs = 20
        index = np.arange(len(X))
        prev_err = 0
        iter = 0
        while epochs > 0:
            iter += 1
            grad_vec = np.zeros_like(param_vec)
            np.random.shuffle(index)  
            for b in range(0, len(X), batch):
                grad_vec[:] = 0  
                err = 0.0  

                for c in range(batch):
                    t = index[min(b + c, len(X) - 1)]  
                    val = np.dot(np.append(X[t] , 1), param_vec) 
                    error = y[t] - val
                    err += (1 / (2 * batch)) * (error ** 2)  
                    grad_vec += error * np.append(X[t], 1)  

                param_vec += learning_rate * (grad_vec / batch)  
                err_list.append(err)
                ret_vec = np.roll(param_vec , 1)  
                point_list.append(ret_vec)
                
                if np.dot(grad_vec/batch , grad_vec/batch) < 0.0005:
                    epochs -= 7
                    break
            
            #code for large batch size convergence
            if abs(err - prev_err) < 0.0005:
                break
            prev_err = err
            epochs -= 1

        #print number of iterations
        
        print(f"Total number of iterations : {iter}")
        self.param_vec = point_list[-1]
        
        plot_err(err_list)
        plot_param(point_list)
        
        self.animate_3d(point_list)
        #print train and test error
        
        print(f"Test Error: {self.err_test()}")
        print(f"Train Error: {self.train_err(X , y)}")

        if batch == 1:
            print(self.closed_form_solution(X[::100] , y[::100]))
        
        return self.param_vec
    
    def closed_form_solution(self , X, y):
        X_n = []
        for i in range(len(X)):
            x = X[i]
            x = np.array([1 , x[0] , x[1]])
            X_n.append(x)
        X = np.array(X_n)
        X_transpose = X.T
        theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
        return theta
    
    def predict(self, X):
        Y = []
        for x in X:
            y = self.param_vec[0]
            for i in range(len(x)):
                y += x[i] * self.param_vec[i+1]
            Y.append(y)
        return np.array(Y)
    
    def train_err(self , X , y):
        mse = 0.0
        for i in range(len(X)):
            prediction = self.param_vec[0]
            for j in range(len(X[0])):
                prediction += self.param_vec[j+1] * X[i][j]
            
            mse += (1/len(X))*(y[i] - prediction) ** 2
        
        return mse ** 0.5

    def err_test(self):
        mse = 0.0
        X , y = generate(200000 , np.array([3, 1 , 2]) , np.array([3 , -1]) , np.array([4 , 4]) , 1)
        for i in range(len(X)):
            prediction = self.param_vec[0]
            for j in range(len(X[0])):
                prediction += self.param_vec[j+1] * X[i][j]
            
            mse += (1/len(X))*(y[i] - prediction) ** 2
        
        return mse ** 0.5
        



X , y = generate(800000 , np.array([3, 1 , 2]) , np.array([3 , -1]) , np.array([4 , 4]) , 1)


sgd = StochasticLinearRegressor()
sgd.fit(X , y)

