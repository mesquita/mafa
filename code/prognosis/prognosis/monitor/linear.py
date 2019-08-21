#*******************************************************************************
# Universidade Federal do Rio de Janeiro
# Instituto Alberto Luiz Coimbra de Pos-Graduacao e Pesquisa de Engenharia
# Programa de Engenharia Eletrica
# Signal, Multimedia and Telecommunications Lab
#
# Author: Felipe Moreira Lopes Ribeiro, Luiz Gustavo Tavares
#
#*******************************************************************************

#*******************************************************************************
# linear.py
#
# Linear monitor class
#
# Created: 2016/08/19
# Modified: 2016/09/12
#
#*******************************************************************************

"""
linear.py

Linear monitor class
"""

#*******************************************************************************
# Imports
#*******************************************************************************

# For pickle
try:
	import cPickle as pickle
except:
	import pickle

# For parallel jobs support
import multiprocessing
import copyreg
import types

# Auxiliary
import numpy as np # Numpy library (Numeric algebra)
from prognosis.monitor.similarity import ssimilarity, psimilarity # Similarity
from prognosis.utils.cluster import fast_median # Median
from sklearn.multiclass import OneVsRestClassifier # OvR wrapper
from sklearn.metrics import accuracy_score # Accuracy measurement
from sklearn.cross_validation import train_test_split # Validation split
from sklearn.preprocessing import normalize # Vector normalize

# Sklearn compatibility
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

#*******************************************************************************
# Functions and global variables
#*******************************************************************************

# Solving issues with pickle and parallel jobs training
def _reduce_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)
copyreg.pickle(types.MethodType, _reduce_method)

#*******************************************************************************
# Class UnaryLinearMonitor
#*******************************************************************************

class UnaryLinearMonitor(BaseEstimator, TransformerMixin, ClassifierMixin):
	"""
	Unary linear autoregressor. Assumes a binary label.
	"""
	
	def __init__(self, kernel='IES', gamma=1e-2, min_err=1e-2, max_itr=100,\
		norm=2, verbose=False, n_jobs=1):
		"""
		Inits UnaryLinearMonitor
		
		@param kernel Similarity kernel.
		@param gamma Similarity kernel auxiliary parameter.
		@param min_err Minimum error threshold stop criterium.
		@param max_itr Maximum number of iterations stop criterium.
		@param norm Minkowski metric parameter.
		@param verbose Show training messages (verbose mode).
		@param n_jobs Number of parallel. Not implemented.
		"""
		
		# Setting parameters
		self.kernel = kernel
		self.gamma = gamma
		self.norm = norm
		self.min_err = min_err
		self.max_itr = max_itr
		
		# Execution modes
		self.verbose = verbose
		self.n_jobs = n_jobs
		
		# Hat matrix and threshold theta
		self.hat = None
		self.theta = 0.5
		
	def validation_split(self, X, y):
		"""
		Computes the validation split for iterative methods
		
		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		
		@return Four elements:
			- Training samples matrix;
			- Validation samples matrix;
			- Training labels;
			- Validation labels.
		"""
		
		# Split
		Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size=0.33, \
			random_state=42)
		
		# Returning
		return Xtr, Xvl, ytr, yvl
	
	def find_theta(self, X, y):
		"""
		Computes classification threshold for a trained model
		
		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		
		@return Two elements:
			- Selected theta;
			- Accuracy score.
		"""
		
		# Setting initial values
		cur_theta = 1.0
		bst_theta = cur_theta
		
		# Computing decision function (similarity scores) for training set
		sim_score = self.decision_function(X)
		
		# Computing initial solution accuracies
		cur_acc = accuracy_score(y, 1*(sim_score>cur_theta))
		bst_acc = cur_acc
		
		# Setting step value
		theta_step = 1e-2
		
		# Finding best theta
		while (cur_theta > 0.0):
			
			# Update theta
			cur_theta = cur_theta-theta_step
			
			# Computing accuracy
			cur_acc = accuracy_score(y, 1*(sim_score>cur_theta))
			
			# Testing
			if cur_acc >= bst_acc:
				
				# Update
				bst_theta = cur_theta
				bst_acc = cur_acc
				
			elif cur_acc < bst_acc:
				
				# Stop search
				break
		
		# Return values
		return bst_theta, bst_acc
	
	def fit(self, X, y=None):
		"""
		Computes hat matrix an threshold.
		
		@param X Input matrix [n_samples, n_features].
		@param y Binary labels vector (N-0/P-1) [n_samples].
		
		@return self
		"""
		
		# Cleaning previous matrix
		self.hat = None
		
		# Testing training case
		if (y is None) or (y.dtype == np.dtype('float64')):
			
			# Setting y as a all one array
			y = np.ones(X.shape[0])
			
		# Finding unique classes
		self.classes_ = np.unique(y)
		
		# Converting to boolean values
		y = (y > 0)
		
		# Breaking training samples into training and validation sets
		Xtr, Xvl, ytr, yvl = self.validation_split(X, y)
		
		# Selecting only positive (true) class elements
		X_true = Xtr[ytr > 0, :].copy()
		
		# Normalizing
		X_nrm = normalize(X_true)
		
		# Computing the current covariance matrix
		covmat = np.dot(X_nrm.T, X_nrm)
		
		# Finding the element with maximum correlation
		corval = np.sum(np.dot(X_nrm, covmat)*X_nrm, axis=1)
		cur_idx = [np.argmax(corval)]
		D = X_true[cur_idx][np.newaxis, :]
		
		# Update the normalized matrix
		X_nrm = X_nrm - X_nrm[cur_idx[-1]]
		X_nrm = normalize(X_nrm)
		X_nrm[cur_idx] = 0
		
		# Setting maximum number of iterations
		max_itr = min(self.max_itr, X_true.shape[0]-2, X_true.shape[1]-1)
		
		# Initial error and hat matrix
		bst_err = 1
		bst_hat = None
		
		# For each iteration
		for itr_idx in range(self.max_itr):
			
			# Computing the covariance matrix
			covmat = np.dot(X_nrm.T, X_nrm)
			
			# Finding the element with maximum correlation
			corval = np.sum(np.dot(X_nrm, covmat)*X_nrm, axis=1)
			cur_idx.append(np.argmax(corval))
			D = X_true[cur_idx]
			
			# Update the normalized matrix
			X_nrm = X_nrm - X_nrm[cur_idx[-1]]
			X_nrm = normalize(X_nrm)
			X_nrm[cur_idx] = 0
			
			# Compute the hat matrix
			cur_hat = np.dot(D, D.T)
			cur_hat = np.linalg.pinv(cur_hat)
			cur_hat = np.dot(cur_hat, D)
			cur_hat = np.dot(D.T, cur_hat)
			
			# Setting current hat matrix
			self.hat = cur_hat.copy()
			
			# Computing current theta
			self.theta, _ = self.find_theta(Xtr, ytr)
			
			# Computing prediction and prediction error
			yerr = 1-self.decision_function(X_true)
			cur_err = 1-accuracy_score(yvl, self.predict(Xvl))
			
			# Testing if verbose
			if self.verbose:
				
				# Print current loop error
				print "Loop {0} error: {1}.".format(itr_idx, cur_err)
				
			
			# Testing current error
			if (bst_err > cur_err):
				
				# Update
				bst_err =  cur_err
				bst_hat = cur_hat.copy()
				stop_con = False
				
			else:
				
				# Set stop condition
				stop_con =  True
				
			# Testing stop conditions
			stop_con = stop_con or (cur_err < self.min_err)
			if stop_con:
				# End loop
				break
		
		# Setting best set
		self.hat = bst_hat.copy()
		self.theta, cur_acc = self.find_theta(Xtr, ytr)
		
		# Testing if verbose
		if self.verbose:
			
			# Print training accuracy
			print "Training accuracy: ", cur_acc, " Length: ", len(cur_idx)-1
		
		# Return self for sklearn API
		return self
	
	def estimate(self, X):
		"""
		Estimates input data using linear model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Input data estimative [n_samples, n_features].
		"""
		
		# Computing the estimative
		X_hat = np.dot(X, self.hat)
		return X_hat
	
	def decision_function(self, X):
		"""
		Returns input data similarity using the similarity model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Similarity with the positive class [n_samples].
		"""
		
		# Computing estimative
		Xhat = self.estimate(X)
		
		# Computing similarity between the original and the estimate
		Xsim = np.linalg.norm(X-Xhat, axis=1, ord=self.norm)
		Xsim = ssimilarity(Xsim, kernel=self.kernel, gamma=self.gamma)
		
		# Returning
		return Xsim
	
	def predict(self, X):
		"""
		Predict class labels for samples in X.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Predicted class label per sample [n_samples].
		"""
		
		# Computing decision function and classes labels and returning
		yhat = self.decision_function(X)
		yhat = 1*(yhat > self.theta)
		return yhat
	
	def transform(self, X):
		"""
		Returns the data residual against the estimated using the similarity
		model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Transformed data.
		"""
		
		# Computing similarity index
		Xsim = self.decision_function(X)
		Xsim = Xsim[:, np.newaxis]
		
		# Concatenating the similarity with x
		Xsim = np.c_[X, Xsim]
		
		# Returning
		return Xsim
	
	def save(self, path):
		"""
		Saves current model.
		
		@param path File path.
		"""
		
		# Opening file
		with open(path, 'w') as fp:
			
			# Saving on disk
			pickle.dump(self.__dict__, fp, 2)
			
	def load(self, path):
		"""
		Loads a saved model.
		
		@param path File path.
		"""
		
		# Opening file
		with open(path, 'r') as fp:
			
			# Loading from disk
			tmp_dict = pickle.load(fp)
			self.__dict__.update(tmp_dict)

#*******************************************************************************
# Class LinearMonitor
#*******************************************************************************

class LinearMonitor(BaseEstimator, TransformerMixin, ClassifierMixin):
	"""
	Linear autoregressor.
	"""
	
	def __init__(self, kernel='IES', gamma=1e-2, min_err=1e-2, max_itr=100,\
		norm=2, verbose=False, n_jobs=1):
		"""
		Inits LinearMonitor.
		
		@param kernel Similarity kernel.
		@param gamma Similarity kernel auxiliary parameter.
		@param min_err Minimum error threshold stop criterium.
		@param max_itr Maximum number of iterations stop criterium.
		@param norm Minkowski metric parameter.
		@param verbose Show training messages (verbose mode).
		@param n_jobs Number of parallel. Not implemented.
		"""
		
		# Setting parameters
		self.kernel = kernel
		self.gamma = gamma
		self.norm = norm
		self.min_err = min_err
		self.max_itr = max_itr
		
		# Execution modes
		self.verbose = verbose
		self.n_jobs = n_jobs
		
		# Setting model
		self.mdl = None
		
	def fit(self, X, y=None):
		"""
		Computes SBM model.
		
		@param X Input matrix [n_samples, n_features].
		@param y Classes labels [n_samples].
		
		@return self
		"""
		
		# Setting model
		self.mdl = OneVsRestClassifier(UnaryLinearMonitor(**self.get_params()),\
			self.n_jobs)
		self.mdl = self.mdl.fit(X,y)
		
		# Return self for sklearn API
		return self
	
	def estimate(self, X):
		"""
		Estimates input data using similarity model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Input data estimative [n_samples, n_classes*n_features].
		"""
		
		# Extracting trained models
		estimators = self.mdl.estimators_
		
		# Computing estimative for each model and concatenating
		Xhat = [e.estimate(X) for e in estimators]
		Xhat = np.concatenate(Xhat, axis=1)
		
		# Returning
		return Xhat
	
	def decision_function(self, X):
		"""
		Returns input data similarity using the similarity model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Similarity with each class [n_samples, n_classes].
		"""
		
		# Just compute the decision function for each class and return
		return self.mdl.decision_function(X)
	
	def predict(self, X):
		"""
		Predict class labels for samples in X.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Predicted class label per sample [n_samples].
		"""
		
		# Just compute the predicted class and return
		return self.mdl.predict(X)
	
	def transform(self, X):
		"""
		Returns the data residual against the estimated using the similarity
		model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Transformed data.
		"""
		
		# Computing the decision function (similarity indices)
		Xsim = self.decision_function(X)
		
		# Concatenating the similarity with x
		Xsim = np.c_[X, Xsim]
		
		# Returning
		return Xsim
	
	def save(self, path):
		"""
		Saves current model.
		
		@param path File path.
		"""
		
		# Opening file
		with open(path, 'w') as fp:
			
			# Saving on disk
			pickle.dump(self.__dict__, fp, 2)
			
	def load(self, path):
		"""
		Loads a saved model.
		
		@param path File path.
		"""
		
		# Opening file
		with open(path, 'r') as fp:
			
			# Loading from disk
			tmp_dict = pickle.load(fp)
			self.__dict__.update(tmp_dict)