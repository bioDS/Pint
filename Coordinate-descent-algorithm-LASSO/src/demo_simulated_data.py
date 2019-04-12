import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import copy

class Algorithm:
	"""
	This is a demo of using coordinate descent algorithm (including both cyclic coordinate descent 
	and randomized coordinate descent) to solve the LASSO problem, that is the `l1-regularized 
	least-squares regression problem.

	"""
	def __init__(self, lambduh=0.05, max_iter=1000, beta0=1, beta1=2, beta2=3, beta3=4):
		"""
		Coordinate descent algorithm to solve the LASSO problem.

		Parameters
		----------
		lambduh : float
		Regularization parameter.
		The default lambduh is set as 0.05(the optimal lambduh from sklearn cross validation result based on 
		default coefficients settings). 
		However, users are allowed set lambduh in the __main__ function to see different training performances. 

		max_iter: int
		Default is set as 1000.
		However, users are allowed set lambduh in the __main__ function to see different training performances. 

		beta0, beta1, beta2, beta3: float
		Default is set as 1, 2, 3, 4 respectively.
		However, users are allowed set lambduh in the __main__ function to see different training performances. 

		"""
		self.lambduh = lambduh
		self.max_iter = max_iter
		self.beta0 = beta0 
		self.beta1 = beta1 
		self.beta2 = beta2 
		self.beta3 = beta3 


	def soft_threshold(self, a, lambduh):
		"""
		Solving l1-norm gradient problem
		"""
		if a < -lambduh:
			return a+lambduh
		elif a > lambduh:
			return a-lambduh
		else:
			return 0
				   

	def min_beta_multivariate(self, x, y, beta, j):
		"""
		Solving partial minimization problem with respect to beta_j for any j = 1...d.
		"""
		n = len(y)
		selector = [i for i in range(x.shape[1]) if i != j]
		norm_x_j = np.linalg.norm(x[:, j])
		a = x[:, j].dot(y[:, np.newaxis] - x[:, selector].dot(beta[:, np.newaxis][selector, :]))
		passin = self.lambduh*n/2
		res = self.soft_threshold(a, passin)
		return res/(norm_x_j**2)

	def generate_simulate_data(self):
		"""
		Generate simulated data sets and standarize data.
		"""
		n = 100
		np.random.seed(0)
		X = np.random.normal(loc=1, scale=1, size=n)
		epsilon = np.random.normal(loc=0, scale=0.01, size=n)
		beta0 = self.beta0
		beta1 = self.beta1
		beta2 = self.beta2
		beta3 = self.beta3
		y = beta0 + beta1*X + beta2*X**2 + beta3*X**3 + epsilon
		predictors = np.vstack([X**i for i in range(1, 11)]).T
		scaler = preprocessing.StandardScaler()
		predictors = scaler.fit_transform(predictors)
		y = y-np.mean(y)
		x_train, x_test, y_train, y_test = train_test_split(predictors, y, test_size=0.25, random_state=0)
		return x_train, x_test, y_train, y_test

	def computeobj(self, beta, x, y):
		"""
		Compute objective value with certain beta.
		"""
		n = len(y)
		return 1/n*np.sum((y-x.dot(beta))**2) + self.lambduh*np.linalg.norm(beta, ord=1)

	def cycliccoorddescent(self, x, y, beta_init):
		"""
		cycliccoorddescent that implements the cyclic coordinate descent algorithm. The cyclic 
		coordinate descent algorithm proceeds sequentially. At each iteration, the algorithm 
		increments the index j of the coordinate to minimize over. Then the algorithm performs 
		partial minimization with respect to the coordinate beta_j corresponding to that index. 
		After updating the coordinate beta_j , the algorithm proceeds to the next iteration. 
		The function takes as input the initial point, the initial step-size value, and the 
		maximum number of iterations. The stopping criterion is the maximum number of iterations.
		"""
		beta = copy.deepcopy(beta_init)
		beta_vals = beta
		d = np.size(x, 1)
		iter = 0
		while iter < self.max_iter:        
			for j in range(d):
				min_beta_j = self.min_beta_multivariate(x, y, beta, j)
				beta[j] = min_beta_j
			beta_vals = np.vstack((beta_vals, beta))
			iter += 1
			if iter % 100 == 0:
				print('Coordinate descent iteration', iter)
		return beta_vals

	def pickcoord(self, d, j):
		"""
		samples uniformly from the set  j = 1...d..
		"""
		num = np.random.randint(0, d)
		while num == j:
			num = np.random.randint(0, d) 
		return num

	def randcoorddescent(self, x, y, beta_init):
		"""
		Randcoorddescent that implements the randomized coordinate descent algorithm. The 
		randomized coordinate descent algorithm proceeds as follows. At each iteration, 
		the algorithm samples the index j of the coordinate to minimize over. Then the 
		algorithm performs partial minimization with respect to the coordinate beta_j 
		corresponding to that index. After updating the coordinate beta_j , the algorithm 
		proceeds to the next iteration. The function takes as input the initial point, 
		the initial step-size value, and the maximum number of iterations. The stopping 
		criterion is the maximum number of iterations.
		"""
		beta = copy.deepcopy(beta_init)
		beta_vals = beta
		d = np.size(x, 1)
		iter = 0
		while iter < self.max_iter:
			j = -1
			for i in range(d):
				j=self.pickcoord(d, j)
				min_beta_j = self.min_beta_multivariate(x, y, beta, j)
				beta[j] = min_beta_j
			beta_vals = np.vstack((beta_vals, beta))
			iter += 1
			if iter % 100 == 0:
				print('Coordinate descent iteration', iter)
		return beta_vals

	def objective_plot(self, betas_cyclic, betas_rand, x, y):
		"""
		Plot the curves of the objective values F(beta_t) for both algorithms versus 
		the iteration counter iter (use different colors)
		"""
		num_points = np.size(betas_cyclic, 0)
		objs_cyclic = np.zeros(num_points)
		objs_rand = np.zeros(num_points)
		for i in range(0, num_points):
			objs_cyclic[i] = self.computeobj(betas_cyclic[i, :], x, y)
			objs_rand[i] = self.computeobj(betas_rand[i, :], x, y)
		fig, ax = plt.subplots()
		ax.plot(range(1, num_points + 1), objs_cyclic, label='Cyclic coordinate descent', linewidth=5.0) 
		ax.plot(range(1, num_points + 1), objs_rand, c='red', label='Randomized coordinate descent')
		plt.xlabel('Iteration')
		plt.ylabel('Objective value')
		plt.title('Objective value vs. iteration when lambda='+str(self.lambduh))
		ax.legend(loc='upper right')
		plt.show()

	def compute_error(self, beta_opt, x, y):
		"""
		Calculate mean squared errors of certain groups of coefficients
		"""
		y_pred = x.dot(beta_opt)  
		error = (1/np.size(x, 0))*sum((y-y_pred)**2)
		return error

	def plot_error(self, betas_cyclic, betas_rand, x, y, title):
		"""
		Plot the curves of the mean squared errors of certain groups of coefficients 
		for both algorithms versus the iteration counter iter (use different colors)
		"""

		niter = np.size(betas_cyclic, 0)
		error_cyclic = np.zeros(niter)
		error_rand = np.zeros(niter)
		for i in range(niter):
			error_cyclic[i] = self.compute_error(betas_cyclic[i, :], x, y)
			error_rand[i] = self.compute_error(betas_rand[i, :], x, y)
		fig, ax = plt.subplots()
		ax.plot(range(1, niter + 1), error_cyclic, label='Cyclic coordinate descent')
		ax.plot(range(1, niter + 1), error_rand, c='red', label='Randomized coordinate descent')
		plt.xlabel('Iteration')
		plt.ylabel('Mean Squared Error')
		if title:
			plt.title(title)
		ax.legend(loc='upper right')
		plt.show()


if __name__=='__main__':
	"""
	Start training process once the file is directly called.
	Users are allowed to set lambduh, max_iter freely.
	Users are also allowed to set coefficients beta0, beta1, beta2, beta3 (used for simulating data) freely.
	Change this line of code accordingly: algorithm = Algorithm()
			
	Example: 
	algorithm = Algorithm(lambduh=2, max_iter=500, beta0=6, beta1=7, beta2=8, beta3=9)

	"""
	print('Start training process..')
	algorithm = Algorithm()
	x_train, x_test, y_train, y_test = algorithm.generate_simulate_data()
	beta_init = np.zeros(np.size(x_train, 1))
	betas_cyclic = algorithm.cycliccoorddescent(x_train, y_train, beta_init)
	betas_rand = algorithm.randcoorddescent(x_train, y_train, beta_init)
	algorithm.objective_plot(betas_cyclic, betas_rand, x_train, y_train)
	algorithm.plot_error(betas_cyclic, betas_rand, x_train, y_train, \
		'Mean Squared Error on Train Data')
	algorithm.plot_error(betas_cyclic, betas_rand, x_test, y_test, \
		'Mean Squared Error on Test Data')
	print('Training process finished. Thank you for viewing!')
