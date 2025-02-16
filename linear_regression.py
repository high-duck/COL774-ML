# Imports - you can add any other permitted libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, cm 

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):
        """uncomment all before running"""
        dfX = pd.read_csv("../data/Q1/linearX.csv")
        np_x = dfX.to_numpy()
        dfY = pd.read_csv("../data/Q1/linearY.csv")
        np_y = dfY.to_numpy()
        np_x = np.reshape(np_x , (-1 ,1))
        np_y = np.reshape(np_y , (-1 ,1))
        self.fit(np_x, np_y)
        self.param_vec = None     
    
    def plot_3d(self, err_list, point_list , coeff):
        ax = plt.axes(projection = "3d")
        X = []
        Y = []
        Z = []
        for i in range(0 , len(err_list) , 10):
            X.append(point_list[i][1])
            Y.append(point_list[i][0])
            Z.append(err_list[i])
        #print(X)
        t0_d = np.linspace(0 , 35 , 30)
        t1_d = np.linspace(0  ,30 , 30)
        t0 , t1 = np.meshgrid(t0_d , t1_d)
        val = coeff[0] * t0 ** 2 + coeff[1] * t1 ** 2 + t0 * t1 * coeff[2] + coeff[3] * t0 + coeff[4] * t1 + coeff[5]
        ax.contour3D(t0 , t1 , val , 100 , cmap=cm.rainbow , alpha=0.7 , zorder = -1)
        ax.plot(X , Y , Z , c = "black")
        ax.set_ylabel("\u03B8\u2081")
        ax.set_xlabel("\u03B8\u2080")
        ax.set_zlabel("Error Function")
        plt.show()
    
    # def step_plot_3d(self, err_list, point_list , coeff):
    #     ax = plt.axes()
    #     X = []
    #     Y = []
    #     Z = []
    #     for i in range(0 , len(err_list)):
    #         X.append(point_list[i][1])
    #         Y.append(point_list[i][0])
    #         Z.append(err_list[i])

    #     t0_d = np.linspace(-10 , 80 , 50)
    #     t1_d = np.linspace(-10  ,80 , 50)
    #     t0 , t1 = np.meshgrid(t0_d , t1_d)
    #     val = coeff[0] * t0 ** 2 + coeff[1] * t1 ** 2 + t0 * t1 * coeff[2] + coeff[3] * t0 + coeff[4] * t1 + coeff[5]
    #     ax.contour(t0 , t1 , val , 50)
    #     ax.plot(X , Y , c = "black")
    #     ax.set_ylabel("\u03B8\u2081")
    #     ax.set_xlabel("\u03B8\u2080")
    #     plt.show()

    def plot_learnt(self , x , y):
        ax = plt.axes()
        X = np.linspace(-2 , 2)
        Y = self.param_vec[1] + self.param_vec[0] * X
        ax.plot(X , Y)
        ax.scatter(x[::15] , y[::15] , c = "orange")
        ax.set_ylabel("Y" , fontweight = "bold")
        ax.set_xlabel("X", fontweight = "bold")
        plt.show()
    
    def animate_3d(self, point_list, err_list):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        point_list = np.array(point_list)
        param1 = point_list[:, 0]  
        param2 = point_list[:, 1]  
        err_values = np.array(err_list)  

        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 0")
        ax.set_zlabel("Error")
        ax.set_title("Gradient Descent Convergence")


        ax.plot(param1, param2, err_values, 'gray', alpha=0.3)  

        # Initialize the animated point
        point, = ax.plot([], [], [], 'bo-', markersize=6)  

        def update(num):
            point.set_data(param1[:num], param2[:num]) 
            point.set_3d_properties(err_values[:num]) 
            return point,

        ani = animation.FuncAnimation(fig, update, frames=len(point_list), interval=50, blit=False)
        plt.show()

    def fit(self, X, y, learning_rate=0.025):
        param_vec = np.zeros(len(X[0]) + 1)
        point_list = []
        err_list = []
        prev_err = 0.0
        coeff = [0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]
        
        for i in range(len(X)):
            coeff[0] += 1
            coeff[1] += X[i] ** 2
            coeff[2] += 2 * X[i] 
            coeff[3] += -2 * y[i]
            coeff[4] += -2 * X[i] * y[i]
            coeff[5] += y[i] ** 2
        coeff = [t * (1 / (2*len(X))) for t in coeff]
        itr = 0
        
        iter = 0
        while (True):
            err = 0
            iter += 1
            grad_vec = np.zeros_like(param_vec)
            for i in range(len(X)):
                curr_fnc = np.dot(np.append(X[i] , 1) , param_vec)
                err += (1/(2* len(X))) * ((y[i] - curr_fnc) ** 2)
                grad_vec += (y[i] - curr_fnc) * np.append(X[i] , 1)
            
            if(abs(err - prev_err) < 0.0005):
                break
            
            param_vec = param_vec + learning_rate * (grad_vec / len(X))
            point_list.append(param_vec)
            err_list.append(err[0])
            prev_err = err
            itr += 1
            
        print(f"Total number of iterations : {iter}")
        self.plot_3d(err_list , point_list , coeff)
        self.param_vec = param_vec
        self.animate_3d(point_list , err_list)
        
        #print(param_vec)
        
        self.plot_learnt(X , y)  
        
        #self.plot_learnt(X , self.predict(X)) 

        return np.roll(param_vec , 1)  
    
    def predict(self, X):
        Y = []
        for i in range(len(X)):
            y = np.dot(self.param_vec , np.append(X[i] , 1))
            Y.append(y)
        return np.array(Y)

# regressor = LinearRegressor()
