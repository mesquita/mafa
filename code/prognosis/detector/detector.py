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
# detector.py
#
# Detector functions and classes.
#
# Created: 2016/03/10
# Modified: 2016/05/30
#
#*******************************************************************************

"""
detector.py

Detector functions and classes.
"""

#*******************************************************************************
# Imports
#*******************************************************************************

# For pickle
try:
	import cPickle as pickle
except:
	import pickle

import numpy as np # Numpy library (Numeric algebra)
from sklearn.svm import OneClassSVM # One class SVM
from scipy.spatial.distance import mahalanobis # To compute Mahalanobis dist.
from scipy.stats import scoreatpercentile # To compute the score at a percentile
from sklearn.base import TransformerMixin # To be compatible with sklearn

#*******************************************************************************
# Class OSVM
#*******************************************************************************

class OSVM(TransformerMixin):
	"""
	One-class SVM used for outlier and novelty detection. Wrapper for
	sklearn implementation of Scholkopf2000.
	"""
	
	def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,\
		tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False,\
			max_iter=-1, random_state=None):
		"""
		Inits OSVM.
		
		@param kernel Kernel type. String ['linear', 'poly', 'rbf', 'sigmoid'].
		@param degree Polynomial kernel degree. Integer.
		@param gamma Kernel coefficient.
		@param coef0 Independent term in kernel function. Scalar.
		@param tol Tolerance for stopping criterion. Scalar float.
		@param nu Error upper bound and SV upper bound. Scalar [0,1].
		@param shrinking Whether to use the shrinking heuristic. Boolean.
		@param cache_size Specify the size of the kernel cache (in MB).
		@param verbose Enable verbose output. Boolean.
		@param max_iter Hard limit on iterations within solver. -1 for no limit.
		@param random_state Random seed.
		"""
		
		# Setting parameters for classifier
		self.__mdl = OneClassSVM(kernel, degree, gamma, coef0, tol, nu,\
			shrinking, cache_size, verbose, max_iter, random_state)
		
	def fit(self, X, y=None, w=None):
		"""
		Detects the soft boundary of the set of samples X.
		
		@param X Input matrix [n_samples, n_features].
		@param y Labels vector [n_samples].
		@param w Per-sample weights [n_samples].
		
		@return self
		"""
		
		# Fit classifier
		self.__mdl = self.__mdl.fit(X, y=y, sample_weight=w)
		
		# Return self for sklearn API
		return self
	
	def predict(self, X):
		"""
		Estimates input data class (normal, novelty or outlier)
		
		@param X Input matrix [n_samples, n_features].
		
		@return Data labels (+1 or -1).
		"""
		
		# Predict
		labels = self.__mdl.predict(X)
		
		# Return
		return labels
	
	def transform(self, X):
		"""
		Returns the data class given the detection model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Data with detection labels data.
		"""
		
		# Computing error
		Xlbs = self.predict(X)
		
		# Concatenating errors to x
		Xlbs = np.hstack((X, Xlbs))
		
		# Returning
		return Xlbs
	
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
# Class MahalDet
#*******************************************************************************

class MahalDet(TransformerMixin):
	"""
	Covariance based outlier/novelty detector.
	"""
	
	def __init__(self, rcond=1e-15, contamination=0.1, tau=1e-2, raw=True):
		"""
		Inits MahalDet.
		
		@param rcond Regularization parameter.
		@param contamination The proportion of outliers in the data set.
		@param tau Kernel precision
		@param raw If true, predicts probabilities. Else, 1/-1 for in/outliers.
		"""
		
		# Setting parameters for classifier
		self.rcond = rcond # Singular values cutoff
		self.contamination = min(0.5, contamination) # Outliers ratio
		self.tau = tau
		self.raw = raw # Prediction options
		self.mean_vc = None # Mean vector
		self.cov_mat = None # Covariance matrix
		self.pre_mat = None # Precision matrix
		self.thrs = 1.0 # Threshold value
		
	def fit(self, X, y=None):
		"""
		Detects the soft boundary of the set of samples X.
		
		@param X Input matrix [n_samples, n_features].
		@param y Labels vector [n_samples].
		
		@return self
		"""
		
		# Computing data mean
		self.mean_vc = X.mean(axis=0)
		
		# Computing data covariance matrix and inverse covariance matrix
		self.cov_mat = np.cov(X, rowvar=0)
		self.pre_mat = np.linalg.pinv(self.cov_mat, self.rcond)
		
		# Computing training srt Mahalanobis distance
		mahal = np.apply_along_axis(mahalanobis, 1, X, \
			self.mean_vc, self.pre_mat)
		mahal = np.exp(-(self.tau*mahal)**2)
		
		# Computing threshold
		self.thrs = scoreatpercentile(mahal, 100*self.contamination)
		print "DBL201. thrs: ", self.thrs
		print "DBL202. mahal: ", mahal.max()
		
		# Return self for sklearn API
		return self
	
	def predict(self, X):
		"""
		Estimates input data class (normal, novelty or outlier)
		
		@param X Input matrix [n_samples, n_features].
		
		@return Mahalanobis distance.
		"""
		
		# Computing Mahalanobis distance
		output = np.apply_along_axis(mahalanobis, 1, X, \
			self.mean_vc, self.pre_mat)
		output = np.exp(-(self.tau*output)**2)
		
		# Testing mode
		if not self.raw:
			
			# Computing threshold
			output = 1.0*(output >= self.thrs) - 1.0*(output < self.thrs)
		
		# Return
		return output
	
	def transform(self, X):
		"""
		Returns the data class given the detection model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Data with detection labels data.
		"""
		
		# Computing error
		Xlbs = self.predict(X)
		
		# Concatenating errors to x
		Xlbs = np.hstack((X, Xlbs))
		
		# Returning
		return Xlbs
	
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