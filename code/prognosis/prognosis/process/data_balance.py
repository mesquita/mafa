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
# data_balance.py
#
# Auxiliary methods to mitigate data imbalance.
#
# Created: 2016/02/02
# Modified: 2016/04/13
#
#*******************************************************************************

"""
data_balance.py

Auxiliary methods to mitigate data imbalance.
"""

#*******************************************************************************
# Imports
#*******************************************************************************

import numpy as np # Numpy library (numeric algebra)
from sklearn.base import BaseEstimator, ClassifierMixin # sklearn compatibility
from sklearn.base import clone # For cloning
from sklearn.ensemble import RandomForestClassifier # Base classifier

#*******************************************************************************
# Functions
#*******************************************************************************

def awgn(x, snr=10):
	"""
	Adds white Gaussian noise to a signal.
	
	@param x Input signal.
	@param snr Signal to power ratio in dB.
	
	@return Signal corrupted by addictive white Gaussian noise.
	"""
	
	# Computing input signal power.
	xpw = np.sum(np.abs(x)**2, axis=0)/x.shape[0]
	xpw = 10*np.log10(xpw)
	
	# Noise power (in dB) and converting to linear power
	npw = xpw-snr
	npw = np.power(10, (npw/10.0))
	
	# Adding noise and returning
	y = x + np.random.normal(size=x.shape)*np.sqrt(npw)
	return y

#*******************************************************************************
# Class RUSBoost
#*******************************************************************************

class RUSBoost(BaseEstimator, ClassifierMixin):
	"""
	RUSBoost algorithm for imbalanced data sets. Based on Seiffert2010. 
	"""
	
	def __init__(self, n_rounds=10, round_est=20, voting='hard',\
		target_class=0, proportion=2.0, verbose=0, n_jobs=1):
		"""
		Inits RUSBoost.
		
		@param n_rounds Number of boosting and random sampling rounds
		@param round_est Number of estimator per round
		@param voting Voting mode. See sklearn.ensemble.VotingClassifier.
		@param target_class Target class label to be balanced.
		@param proportion New proportion of samples for target class.
		@param verbose Show training messages (verbose mode).
		@param n_jobs Number of jobs to run in parallel.
		"""
		
		# Setting training parameters
		self.n_rounds = n_rounds
		self.round_est = round_est
		self.voting = voting
		self.target_class = target_class
		self.proportion = proportion
		self.verbose = verbose
		self.n_jobs = n_jobs
		
		# Initializing voting classifier and label encoder
		self.ensemble = []
		self.est_w = []
	
	def fit(self, X, y):
		"""
		Fits estimators ensemble using RUSBoost method.
		
		@param X Input matrix [n_samples, n_features].
		@param y Labels vector [n_samples].
		"""
		
		# Initializing estimators and estimators weights' list
		self.ensemble = []
		self.est_w = []
		
		# Finding number of target class elements and indices
		tgt_idx = np.where(y==self.target_class)[0]
		num_trg = tgt_idx.shape[0]
		
		# Setting other classes samples indices
		oth_idx = np.where(y!=self.target_class)[0]
		
		# Breaking training samples in two groups
		trgX = X[tgt_idx, ] # Target class samples
		othX = X[oth_idx, ] # Other classes samples
		trgY = y[tgt_idx] # Target class labels
		othY = y[oth_idx] # Other classes labels
		
		# Setting proportion of samples
		num_smp = int(np.round(num_trg*self.proportion))
		
		# Setting training weights
		train_w = np.ones(y.shape[0])
		train_w = train_w/train_w.sum()
		
		# Setting base classifier
		base_est = RandomForestClassifier(n_estimators=self.round_est,\
			n_jobs=self.n_jobs, verbose=self.verbose, bootstrap=False)
		
		# For each round
		for ridx in range(self.n_rounds):
			
			# Testing if verbose
			if self.verbose:
				
				# Print current loop
				print "RUSBoost loop {0}".format(ridx)
			
			# Computing others classes samples distribution
			oth_w = train_w[oth_idx]
			oth_w = oth_w/oth_w.sum()
			
			# Random sampling with replacement
			rnd_idx = np.random.choice(oth_w.shape[0], size=num_smp, p=oth_w)
			
			# Setting random samples
			randX = othX[rnd_idx, ]
			randY = othY[rnd_idx]
			
			# Creating current training samples
			trainX = np.vstack((trgX, randX))
			trainY = np.concatenate((trgY, randY))
			
			# Training
			cur_est = clone(base_est).fit(trainX, trainY)
			
			# Computing predictions
			cur_prd = cur_est.predict(X)
			
			# Computing the pseudo-loss
			cur_mss = np.not_equal(y, cur_prd)
			cur_lss = np.sum(train_w[cur_mss])
			
			# Testing loss
			if cur_lss <= 0.5:
				
				# Computing weight update parameter
				alpha = cur_lss/(1-cur_lss)
				
				# Updating weights
				train_w += train_w*cur_mss*alpha
				train_w = train_w/train_w.sum()
				
				# Save current classifier and weight
				self.ensemble.append(cur_est)
				self.est_w.append(np.log(1/alpha))
				
			else:
				
				# Inverted hypothesis case. TODO: Handle this case
				print "RUSBoost loop {0}. Inverted hypothesis".format(ridx)
				
				# Computing weight update parameter
				alpha = cur_lss/(1-cur_lss)
				
				# Updating weights
				train_w += train_w*cur_mss*alpha
				train_w = train_w/train_w.sum()
				
				# Save current classifier and weight
				self.ensemble.append(cur_est)
				self.est_w.append(np.log(1/alpha))
		
		# Return self for sklearn API
		return self
	
	def predict(self, X):
		"""
		Estimates input data using RUSBoost trained ensemble.
		
		@param X Input matrix [n_samples, n_features].
		
		@return Data label for each sample.
		"""
		
		if self.voting=="hard":
			
			# For each classifier, compute the label
			cls_prd = np.asarray([est.predict(X) for est in self.ensemble]).T
			
			# Computing majority votes
			maj_vot = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,\
				weights=self.est_w)), axis=1, arr=cls_prd)
		
		# Computing predictions and returning
		return maj_vot