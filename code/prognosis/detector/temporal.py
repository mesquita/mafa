#*******************************************************************************
# Universidade Federal do Rio de Janeiro
# Instituto Alberto Luiz Coimbra de Pos-Graduacao e Pesquisa de Engenharia
# Programa de Engenharia Eletrica
# Signal, Multimedia and Telecommunications Lab
#
# Author: Felipe M. L. Ribeiro, Luiz G. C. Tavares, Matheus A. Marins
#
#*******************************************************************************

#*******************************************************************************
# temporal.py
#
# Temporal detectors functions and classes.
#
# Created: 2016/05/25
# Modified: 2016/09/27
#
#*******************************************************************************

"""
temporal.py

Temporal detectors functions and classes.
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
from sklearn.base import TransformerMixin # To be compatible with sklearn
from sklearn.metrics import accuracy_score # Accuracy measurement
from scipy.signal import medfilt # Median filter

#*******************************************************************************
# Class SPRT
#*******************************************************************************

class SPRT(TransformerMixin):
	"""
	Sequential Probability Ratio Test.
	"""
	
	def __init__(self, alpha=0.1, beta=0.1):
		"""
		Inits SPRT.
		
		@param alpha False positive error probability [0, 1].
		@param beta False negative error probability [0, 1].
		"""
		
		# Setting parameters
		self.alpha = alpha
		self.beta = beta
		
		# Computing boundaries
		self.A = np.log(beta/(1.0-alpha))
		self.B = np.log((1-beta)/alpha)
		
		# Setting initial probability ratio
		self.lratio = 0.0
		
	def fit(self, X, y=None):
		"""
		Finds the SPRT parameters and boundaries.
		
		@param X Input matrix [n_samples, n_features].
		@param y Labels vector [n_samples].
		
		@return self
		"""
		
		# Setting initial probability ratio
		self.lratio = 0.0
		
		# Return self for sklearn API
		return self
	
	def predict(self, X):
		"""
		Data hypothesis test.
		
		@param X Input scalar or vector [n_samples].
		
		@return Data labels (+1, 0 or -1).
		"""
		
		# Testing if is iterable
		if hasattr(X, "__iter__"):
			
			# Vectorial case
			
			# For each element, computes the label
			label = [self.predict(x) for x in X]
			label = np.array(label)
			
			# Replicating next non-zero value
			nzr_idx = np.nonzero(label)[0] # Non-zero index
			nzr_lbl = np.array(label, dtype=np.bool) # Non-zero Boolean array
			nzr_cnt = np.cumsum(nzr_lbl) # Cumulative array
			nzr_cnt[nzr_lbl] += 1 # The next non-zero value
			nzr_cnt = np.minimum(nzr_cnt, len(nzr_idx)) # Limited to last valid
			label = np.where(nzr_cnt, label[nzr_idx[nzr_cnt-1]], 0)
			
		else:
			
			# Scalar case. Computing the cumulative likelihood ratio.
			self.lratio += np.log(X) - np.log(1.0-X)
			
			# Initial labels
			label = 0
			
			# Testing threshold
			if self.lratio > self.B:
				
				# Return label and reset
				label = 1.0
				self.lratio = np.log(X) - np.log(1.0-X)
				
			elif self.lratio < self.A:
				
				# Return label and reset
				label = -1.0
				self.lratio = np.log(X) - np.log(1.0-X)
		
		# Return
		return label
	
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
# Class MedianFilter
#*******************************************************************************

class MedianFilter(TransformerMixin):
	"""
	Median filtering.
	"""
	
	def __init__(self):
		"""
		Inits median filter.
		"""
		
		# Setting initial filter size 
		self.size = 3
		
		# Setting classes
		self.classes_ = None
		
	def fit(self, X, y):
		"""
		Finds the best filter size.
		
		@param X Input decision function [n_samples, n_classes].
		@param y Labels vector [n_samples].
		
		@return self
		"""
		
		# Save labels
		self.classes_ = np.unique(y)
		
		# Computing initial and best accuracy
		yhat = self.classes_[X.argmax(axis=1)]
		cur_acc = accuracy_score(y, yhat)
		bst_acc = cur_acc
		
		# Setting minimum, maximum and initial best size
		max_size = min(1443, X.shape[0]-1)
		min_size = 3
		bst_size = 1
		
		# Finding best size
		for cur_size in range(min_size, max_size, 2):
			
			# Setting current size
			self.size = cur_size
			
			# Computing prediction
			yhat = self.predict(X)
			
			# Testing prediction
			cur_acc = accuracy_score(y, yhat)
			
			# Comparing
			if cur_acc > bst_acc:
				
				# Update
				bst_acc = cur_acc
				bst_size = cur_size
		
		# Saving best size
		self.size = bst_size
		
		# Return self for sklearn API
		return self
	
	def decision_function(self, X):
		"""
		Computes the smoothed decision function.
		
		@param X Input decision function [n_samples, n_classes].
		
		@return Smoothed decision function.
		"""
		
		# Adding border conditions samples
		X_scn = X[self.size-1::-1, :] # Start condition
		X_ecn = X[:self.size-1:-1, :] # End condition
		X_ext = np.r_[X_scn, X, X_ecn]
		
		# Smoothing
		X_ext = np.apply_along_axis(medfilt, 0, X_ext, kernel_size=self.size)
		
		# Removing border samples
		X_out = X_ext[self.size:(X.shape[0]+self.size), :]
		
		# Returning
		return X_out
	
	def predict(self, X):
		"""
		Computes the smoothed labels.
		
		@param X Input decision function [n_samples, n_classes].
		
		@return Data labels.
		"""
		
		# Computing the decision function
		X_dec = self.decision_function(X)
		
		# Predicting
		yhat = self.classes_[X_dec.argmax(axis=1)]
		
		# Return
		return yhat
	
	def transform(self, X):
		"""
		Returns the data class given the detection model.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Data with detection labels data.
		"""
		
		# Computing error
		Xlbs = self.predict(X)
		
		# Concatenating labels to x
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