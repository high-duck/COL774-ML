from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class GaussianDiscriminantAnalysis:
    def __init__(self):
        """uncomment all before running"""
        dfX = pd.read_csv("../data/Q4/q4x.dat" , delim_whitespace=True , header=None)
        np_x = dfX.to_numpy()
        dfY = pd.read_csv("../data/Q4/q4y.dat" , header=None)
        np_y = dfY.to_numpy()
        np_y = np.reshape(np_y , (-1 ,1))
        self.fit(np_x, np_y)
        self.means = None
        self.cov_matrices = None
        self.labels = None
    
    def nomarlize(self , X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    def plot_decision_boundary(self , X, y, mean_vectors, cov_matrices, labels , assume , shared_cov):
        plt.figure(figsize=(8, 6))

        for label, color in zip(labels.keys(), ['blue', 'red']):
            plt.scatter(X[y.flatten() == label, 0], X[y.flatten() == label, 1], 
                        label=label, color=color, edgecolor='k')

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]


        mu_0, mu_1 = mean_vectors
        Sigma_0, Sigma_1 = cov_matrices

        inv_Sigma_0 = np.linalg.inv(Sigma_0)
        inv_Sigma_1 = np.linalg.inv(Sigma_1)

        print(shared_cov)

        Sigma_inv = np.linalg.inv(shared_cov) 
        w = Sigma_inv @ (mu_1 - mu_0)
        w0 = -0.5 * (mu_1.T @ Sigma_inv @ mu_1 - mu_0.T @ Sigma_inv @ mu_0)
        Z = (grid @ w + w0).reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], colors='green', linewidths=2)

        term1 = np.sum(grid @ (inv_Sigma_1 - inv_Sigma_0) * grid, axis=1)
        term2 = 2 * (mu_0.T @ inv_Sigma_0 - mu_1.T @ inv_Sigma_1) @ grid.T
        term3 = mu_1.T @ inv_Sigma_1 @ mu_1 - mu_0.T @ inv_Sigma_0 @ mu_0
        term4 = np.log(np.linalg.det(Sigma_1) / np.linalg.det(Sigma_0))
        Z = (term1 + term2 + term3 + term4).reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary (LDA / QDA)")
        plt.legend()
        plt.show()
    
    def qda_decision_boundary(self , mean0, cov0, mean1, cov1, prior0=0.5, prior1=0.5):

        inv_cov0 = np.linalg.inv(cov0)
        inv_cov1 = np.linalg.inv(cov1)

        W = 0.5 * (inv_cov0 - inv_cov1)
        
      
        w = inv_cov1 @ mean1 - inv_cov0 @ mean0
        
        w_0 = (0.5 * (mean0.T @ inv_cov0 @ mean0 - mean1.T @ inv_cov1 @ mean1) +
            np.log(np.linalg.det(cov1)**0.5 / np.linalg.det(cov0)**0.5) +
            np.log(prior0 / prior1))
        
        
        w_xx, w_xy = W[0, 0], W[0, 1]
        w_yy = W[1, 1]
        w_x, w_y = w[0], w[1]

        print("Quadratic Decision Boundary Equation:")
        print(f"{w_xx:.3f} * x^2 + {2 * w_xy:.3f} * xy + {w_yy:.3f} * y^2 + {w_x:.3f} * x + {w_y:.3f} * y + {w_0:.3f} = 0")
            

    def gaussian(self, x, mean, cov_matrix):
        n = len(mean) 
        det_cov = np.linalg.det(cov_matrix) 
        inv_cov = np.linalg.inv(cov_matrix) 
        c = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_cov))
        diff = x - mean
        f = np.exp(-0.5 * (diff @ inv_cov @ diff.T))  
        return c * f

    
    def fit(self, X, y, assume_same_covariance=True):
        X = self.nomarlize(X)
        labels = {'Alaska':0 , 'Canada':1}
        mean_vectors = np.array([np.zeros_like(X[0]) , np.zeros_like(X[0])])
        samples = np.array([0 , 0])
        
        for i in range(len(X)):
            label = labels[y[i][0]]
            mean_vectors[label] += X[i]     
            samples[label] += 1

        mean_vectors[0] /= samples[0]
        mean_vectors[1] /= samples[1]

        cov_matrices = np.zeros((2, len(X[0]), len(X[0])))
        shared_cov = np.zeros((len(X[0]) , len(X[0])))

        for i in range(len(X)):
            label = labels[y[i][0]]
            cov_matrices[label] += np.outer(X[i] - mean_vectors[label] , X[i] - mean_vectors[label])
            shared_cov += np.outer(X[i] - mean_vectors[label] , X[i] - mean_vectors[label])

        shared_cov /= samples[0] + samples[1]

        cov_matrices[0] /= samples[0]
        cov_matrices[1] /= samples[1]

        self.plot_decision_boundary(X , y , mean_vectors , cov_matrices , labels , assume_same_covariance , shared_cov)
        #print(mean_vectors)
        #print(cov_matrices)

        self.labels = labels
        self.means = mean_vectors

        if assume_same_covariance:
            cov_matrices[0] = cov_matrices[1]
            self.cov_matrices = cov_matrices
            #print(self.predict(X))
            return mean_vectors[0] , mean_vectors[1] , cov_matrices[0]
        else:
            self.cov_matrices = cov_matrices
            #print(self.predict(X))
            self.qda_decision_boundary(mean_vectors[0] ,cov_matrices[0], mean_vectors[1] , cov_matrices[1] , 
                                       prior0=samples[0] / (samples[0] + samples[1]) , 
                                       prior1=samples[1] / (samples[0] + samples[1]))
            return mean_vectors[0] , mean_vectors[1] , cov_matrices[0] , cov_matrices[1]
        
        
            
    def predict(self, X):
        Y = []  
        for x in X:
            likelihood_0 = self.gaussian(x, self.means[0], self.cov_matrices[0])
            likelihood_1 = self.gaussian(x, self.means[1], self.cov_matrices[1])
            if likelihood_1 > likelihood_0:
                key = next((k for k, v in self.labels.items() if v == 1), None)
            else:
                key = next((k for k, v in self.labels.items() if v == 0), None)

            Y.append(key)
        
        return np.array(Y)

# gda = GaussianDiscriminantAnalysis()
