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
# similarity.py
#
# Similarity and similarity-based functions and classes
#
# Created: 2015/10/09
# Modified: 2017/03/02
#
#*******************************************************************************

"""
similarity.py

Similarity and similarity-based functions and classes
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
from scipy.spatial.distance import cdist, pdist, squareform # Pairwise distances
from scipy.linalg import pinv2 # Pseudo-inverse
from prognosis.utils.cluster import pam, clara, fast_median # Cluster methods
from sklearn.cluster import KMeans # KMeans
from sklearn.multiclass import OneVsRestClassifier # OvR wrapper
from sklearn.metrics import accuracy_score # Accuracy measurement
from sklearn.model_selection import train_test_split # Validation split

# Transformations
from sklearn.preprocessing import StandardScaler # Zscore scaler

# Sklearn compatibility
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

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

def ssimilarity(D, kernel='IES', gamma=0.5):
	"""
	Given an Euclidean distance measure, computes the similarity score.

	@param D Euclidean distance vector/matrix.
	@param kernel Similarity kernel.
	@param gamma Similarity kernel auxiliary parameter.

	@return Similarity score.
	"""

	# Testing similarity kernel
	if (kernel=='IES'):

		# Original similarity metric from Wegerich2003
		S = 1/(1+gamma*D)

	elif (kernel=='RBF'):

		# RBF kernel similarity
		S = np.exp(-gamma*D**2)

	elif (kernel=='EXP'):

		# Exponential/Laplacian kernel similarity
		S = np.exp(-gamma*D)

	elif (kernel=='IMK'):

		# Inverse Multiquadric kernel
		S = 1/np.sqrt(1 + (gamma*D)**2)

	elif (kernel=='CCK'):

		# Cauchy Kernel
		S = 1/(1+(gamma*D)**2)

	# Returning
	return S

def ssimil_diff(A, B=None, kernel='IES', gamma=0.5, norm=2):
	"""
	Computes the diffential similarity.

	@param A Input vector or array.
	@param B Input vector. If none, considers A as the difference vector.
	@param kernel Similarity kernel.
	@param gamma Similarity kernel auxiliary parameter.
	@param norm Minkowski metric parameter.

	@return Differential similarity vector.
	"""

	# Test matrix A shape
	if (A.ndim > 1):

		# Apply function recursively along rows
		dff_out = np.apply_along_axis(ssimil_diff, 1, A, B)

	else:

		# Initial difference vector
		dff_vec = (A-B) if B is not None else A.copy()

		# Computing the norm and the similarity
		dff_nrm = np.linalg.norm(dff_vec)
		sim_scr = ssimilarity(dff_nrm, kernel=kernel, gamma=gamma)

		# Multiplying the difference vector by the similarity and common constants
		dff_out = -gamma*dff_vec*sim_scr

		# Testing the similarity type
		if (kernel=='IES'):

			# Original similarity metric from Wegerich2003
			if dff_nrm > np.finfo('d').eps:

				# Compute diffential
				dff_out = (dff_out*sim_scr)/dff_nrm

			else:

				# Return zero vector
				dff_out = 0*dff_out

		elif (kernel=='RBF'):

			# RBF kernel similarity
			dff_out = 2*dff_out

		elif (kernel=='EXP'):

			# Exponential/Laplacian kernel similarity
			if dff_nrm > np.finfo('d').eps:

				# Compute diffential
				dff_out = dff_out/dff_nrm

			else:

				# Return zero vector
				dff_out = 0*dff_out

		elif (kernel=='IMK'):

			# Inverse Multiquadric kernel
			dff_out = 2*gamma*dff_out*sim_scr

		elif (kernel=='CCK'):

			# Cauchy Kernel
			dff_out = 2*gamma*dff_out*sim_scr

	# Returning
	return dff_out

def psimilarity(A, B=None, kernel='IES', gamma=0.5, norm=2, icov_mtx=None):
	"""
	Computes pairwise similarity between samples or collections.

	@param A Rowise sample matrix.
	@param B Rowise sample matrix.
	@param kernel Similarity kernel.
	@param gamma Similarity kernel auxiliary parameter.
	@param norm Minkowski metric parameter.
	@param metric metric option.
	@param icov_mtx Inverse of covariance matrix

	@return Similarity matrix.
	"""

	# Testing if there is a second collection
	if B is None:

		# Compute self-distance matrix
		if norm == 'mahalanobis':

			smat = pdist(A, metric='mahalanobis', VI=icov_mtx)
			smat = squareform(smat)

		else:
			smat = pdist(A, metric='minkowski', p=norm)
			smat = squareform(smat)

	else:

		# Compute cross-distance matrix
		if norm == 'mahalanobis':

			smat = cdist(A, B, metric='mahalanobis', VI=icov_mtx)

		else:
			smat = cdist(A, B, metric='minkowski', p=norm)


	# Computing the similarity
	smat = ssimilarity(smat, kernel=kernel, gamma=gamma)

	# Returning
	return smat

#*******************************************************************************
# Class USBM
#*******************************************************************************

class USBM(BaseEstimator, TransformerMixin, ClassifierMixin):
	"""
	Unary Similarity Based-Modeling from Wegerich2003. Unary autoregressor
	with kernel matrix transformation. Assumes a binary label.
	"""

	def __init__(self, model="SBM", method="kmeans", kernel='IES',\
		min_err=1e-2, max_itr=100, gamma=1e-2, tau=0.15, sampling=5,\
		norm=2, feat="similarity", scl=False, verbose=False, n_jobs=1):
		"""
		Inits SBM.

		@param model Model type.
		@param method Training method.
		@param kernel Similarity kernel.
		@param min_err Minimum error threshold stop criterium.
		@param max_itr Maximum number of iterations stop criterium.
		@param gamma Similarity kernel auxiliary parameter.
		@param tau Similarity score threshold value for prototype selection.
		@param sampling Subsampling factor for prototype selection.
		@param norm Minkowski metric parameter.
		@param feat Transformation feat (similarity score or residual).
		@param scl Z-score scaler option.
		@param verbose Show training messages (verbose mode).
		@param n_jobs Number of parallel. Not implemented.
		"""

		# Model parameters
		self.model = model
		self.method = method
		self.kernel = kernel
		self.gamma = gamma

		# Setting training parameters
		self.min_err = min_err
		self.max_itr = max_itr
		self.tau = tau
		self.sampling = sampling
		self.norm = norm

		# Setting transformation feat
		self.feat = feat

		# Setting verbose mode
		self.verbose = verbose

		# Number of parallel jobs
		self.n_jobs = n_jobs

		# Setting initial state
		self.D = None
		self.G = None
		self.theta = 0.5
		self.icov_mtx = None

		# Setting scaler
		self.scl = scl
		self.std_scl = StandardScaler(with_mean=scl, with_std=scl)

		# Setting training dictionary
		self.fit_method = {
			"dummy": self.fit_dummy,
			"centroids": self.fit_centroids,
			"kmeans": self.fit_kmeans,
			"original": self.fit_original,
			"threshold": self.fit_threshold,
			"threshold2": self.fit_threshold2,
			"perceptron": self.fit_perceptron,
			"greedy": self.fit_greedy,
			"rgreedy": self.fit_rgreedy
		}

	def compute_G(self, D):
		"""
		Computes matrix G given the selected model

		@param D Prototypes matrix [n_prototypes, n_features].

		@return Matrix G [n_prototypes, n_prototypes].
		"""

		# Computing matrix G according with selected model
		if (self.model == "SBM"):

			# Computing similarity matrix and inverting
			G = psimilarity(A=D, kernel=self.kernel, gamma=self.gamma,\
				norm=self.norm, icov_mtx=self.icov_mtx)
			G = pinv2(G)
			#G = np.linalg.pinv(G)

		elif (self.model == "AAKR"):

			# Using identity matrix as inverse similarity matrix
			G = np.eye(D.shape[0])

		# Return G
		return G

	def compute_icov(self, X, y):
		"""
		Evaluates the covariance matrix for the class  descriminated by y.

		@param X Input matrix [n_samples, n_features].
		@param y Binary labels vector (N-0/P-1) [n_samples].

		@return icov_mtx Inverse of covariance matrix
		"""

		X_true = X[y,:]

		# Initialize icov_mtx.
		icov_mtx = np.eye(X.shape[1])

		# Computing covariance matrix inverse if there is at least one sample
		# of the class discriminated on y.
		if y.sum():

			# rowvar is set to FAlSE in order to evaluate the covariance matrix
			# according to its features
			cov_mtx = np.cov(X_true, rowvar=False)
			icov_mtx = np.linalg.pinv(cov_mtx)

		return icov_mtx

	def fit(self, X, y=None):
		"""
		Computes SBM model.

		@param X Input matrix [n_samples, n_features].
		@param y Binary labels vector (N-0/P-1) [n_samples].

		@return self
		"""

		# First, cleaning previous data
		self.D = None
		self.G = None
		self.icov_mtx = None
		# Testing training case
		if (y is None) or (y.dtype == np.dtype('float64')):

			# Setting y as a all one array
			y = np.ones(X.shape[0])

		# Finding unique classes
		self.classes_ = np.unique(y)

		# Converting to boolean values
		y = (y > 0)

		# Normalizing for the target class
		self.std_scl = self.std_scl.fit(X[y, :])
		Xscl = self.std_scl.transform(X)

		# Training method
		if self.norm == 'mahalanobis':
			self.icov_mtx = self.compute_icov(Xscl, y).copy()

		self.fit_method[self.method](Xscl, y)
		# Return self for sklearn API
		return self

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

	def bien_2011(self, X, y, Z=None):
		"""
		Computes prototypes using approach proposed in Bien2011.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		@param Z Input prototype list [n_prototypes, n_features].

		@return Return prototypes index list.
		"""

		# Testing for prototypes
		Z = X[y, :] if (Z is None) else Z

		# Computing dissimilarity matrix
		dis_mat = psimilarity(A=X, B=Z, kernel=self.kernel, gamma=self.gamma,\
			norm=self.norm, icov_mtx=self.icov_mtx)
		dis_mat = 1 - dis_mat

		# Setting auxiliary variables
		classes = self.classes_.tolist() # Classes
		nproto = dis_mat.shape[1] # Number of prototypes
		nclass = len(classes)  # Number of classes
		nsmpls = dis_mat.shape[0] # Number of samples
		lmbd = 1.0/nsmpls # Cost of adding a prototype

		# Matrix indicating the class of each point
		clsmat = np.zeros((nsmpls, nclass))

		# Setting indicators
		for i,c in enumerate(classes):

			# Comparing
			clsmat[:, i] = 1.0*(y==c)

		# Setting the covering. Each col indicates points covered by prototype
		covmat = 1.0*(dis_mat < self.tau)

		# Corrected covered points
		cov_y = np.zeros(nsmpls)

		# Cost and score matrices
		cstmat = 2.0*clsmat-1.0
		scrmat = np.dot(covmat.T, cstmat)

		# Setting loop variables
		protos = [] # Prototypes list
		end_flg = False # End loop condition flag

		# Detection loop
		while (not end_flg):

			# Finding maximum element index
			pmax, cmax = np.unravel_index(np.argmax(scrmat), scrmat.shape)

			# Testing new element
			if scrmat[pmax, cmax] > lmbd:

				# Add this prototype and set it as active
				protos.append(pmax)

				# Updating scores

				# Identify points that are no longer uncovered
				cur_cov = np.logical_and(np.logical_not(cov_y), covmat[:, pmax])
				cur_cov = np.logical_and(cur_cov, clsmat[:, cmax])

				# Updating covered points
				cov_y = cov_y + cur_cov

				# Determine which (potential) prototypes' scores were affected
				aff_proto = np.where(np.sum(covmat[cur_cov, :], axis=0) > 0)[0]

				# For each affected proto, determine the # of points covered
				scrred = [np.sum(covmat[:, p]*cur_cov) for p in aff_proto]
				scrred = np.array(scrred)

				# Remove from score
				scrmat[aff_proto, cmax] = scrmat[aff_proto, cmax] - scrred

			else:

				# Set end loop condition
				end_flg = True

		# Returning prototype list
		return protos

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

	def fit_dummy(self, X, y):
		"""
		Just uses all training sample as prototypes.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting D and computing G and theta
		self.D = X[y, :].copy()
		self.G = self.compute_G(self.D)
		self.theta, cur_acc = self.find_theta(X, y)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: {}".format(cur_acc))

	def fit_original(self, X, y):
		"""
		Computes SBM model using original approach to select prototypes
		proposed in [Herzog1998].

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None

		# Selecting only positive class members
		X_true = X[y, :]

		# First, selecting each samples where extrema occurs as prototypes
		proto_idx = np.argmax(X_true, axis=0).tolist() # Max. per feature
		proto_idx.extend(np.argmin(X_true, axis=0).tolist()) # Min. per feature

		# Calculating l2-norms
		if self.norm == 'mahalanobis':
			norms = np.linalg.norm(X_true, axis=1, ord=2)
		else:
			norms = np.linalg.norm(X_true, axis=1, ord=self.norm)

		# Now, selecting prototypes given with its norms and the sampling ratio
		sampling = self.sampling
		proto_idx.extend(np.argsort(norms)[::-sampling])

		# Excluding repeated elements
		proto_idx = np.unique(proto_idx).tolist()

		# Saving representative matrix D
		D = X_true[proto_idx, :]

		# Computing matrix G given the selected model
		G = self.compute_G(D)

		# Saving data and similarity matrices
		self.D = D.copy()
		self.G = G.copy()

		# Computing theta
		self.theta, cur_acc = self.find_theta(X, y)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: {}".format(cur_acc))

	def fit_threshold(self, X, y):
		"""
		Computes SBM model using an approach based on a similarity threshold.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None

		# Selecting only positive class members
		X_true = X[y, :]

		# Starting D
		D = fast_median(X_true)[np.newaxis, :]

		# For each sample
		for smp_idx in range(X_true.shape[0]):

			# Computing current sample similarity
			cur_smp = X_true[smp_idx, :][np.newaxis, :]
			sim_score = psimilarity(D, cur_smp, self.kernel, self.gamma,\
				norm=self.norm, icov_mtx=self.icov_mtx)

			# Testing if current sample is similar to any prototype
			if (sim_score.max() < self.tau):

				# Add current sample to D
				D = np.r_[D, cur_smp]

		# Computing matrix G given the selected model
		G = self.compute_G(D)

		# Saving data and similarity matrices
		self.D = D.copy()
		self.G = G.copy()

		# Computing theta
		self.theta, cur_acc = self.find_theta(X, y)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: {}".format(cur_acc))

	def fit_threshold2(self, X, y):
		"""
		Computes SBM model using an approach based on a similarity threshold.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None

		# Selecting only positive class members
		X_true = X[y, :]

		# Starting D
		D = fast_median(X_true)[np.newaxis, :]

		# For each sample
		for smp_idx in range(X_true.shape[0]):

			# Setting current D
			self.D = D.copy()
			self.G = self.compute_G(D)

			# Computing current sample similarity
			cur_smp = X_true[smp_idx, :][np.newaxis, :]
			sim_score = self.decision_function(cur_smp)

			# Testing if current sample is similar to current D
			if (sim_score < self.tau):

				# Add current sample to D
				D = np.r_[D, cur_smp]

		# Computing matrix G given the selected model
		G = self.compute_G(D)

		# Saving data and similarity matrices
		self.D = D.copy()
		self.G = G.copy()

		# Computing theta
		self.theta, cur_acc = self.find_theta(X, y)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: {}".format(cur_acc))

	def fit_kmeans(self, X, y):
		"""
		Computes SBM model using k-means to select prototypes.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None
		bstD = None
		bstG = None

		# Breaking training samples into training and validation sets
		Xtr, Xvl, ytr, yvl = self.validation_split(X, y)

		# Selecting only positive (true) class elements
		X_true = Xtr[ytr > 0, :]

		# Setting initial error
		bst_err = 1
		cur_err = 1

		# Setting stop criteria
		max_itr = min(self.max_itr, X_true.shape[0]-1) # Max. number of iter.
		stp_itr = 3 # Max. number of consecutive iterations without improvement
		cur_stp = 0 # Current number of iterations without improvement

		# For each iteration
		for k in range(2, max_itr):

			# Update previous error
			bst_err = cur_err

			# Finding the clusters centers using the kmeans algorithm
			kmu_mdl = KMeans(n_clusters=k)
			kmu_mdl.fit(X_true)
			D = kmu_mdl.cluster_centers_

			# Computing matrix G given the selected model
			G = self.compute_G(D)

			# Predicting using current data
			self.D = D.copy()
			self.G = G.copy()

			# Find theta, computing accuracy and labels
			self.theta, _ = self.find_theta(Xtr, ytr)

			# Computing prediction error
			cur_err = 1-accuracy_score(yvl, self.predict(Xvl))

			# Testing if verbose
			if self.verbose:

				# Print current loop error
				print("Loop {0} error: {1}.".format(k, cur_err))

			# Testing if current results are the best found
			if (bst_err - cur_err> 1e3*np.finfo('d').eps):

				# Updating data
				bstD = D.copy()
				bstG = G.copy()
				bst_err = cur_err

				# Reseting the number of iterations without any improvement
				cur_stp = 0

			else:

				# Incrementing the number of iterations without improvement
				cur_stp = cur_stp+1

			# Testing stop conditions
			stop_con = (cur_err < self.min_err) or (X_true.size <= 0)
			stop_con = stop_con or (cur_stp >= stp_itr)
			if stop_con:
				# End loop
				break

		# Saving current data and similarity matrices
		self.D = bstD.copy()
		self.G = bstG.copy()

		# Computing theta
		self.theta, cur_acc = self.find_theta(Xtr, ytr)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: {}".format(cur_acc))

	def fit_centroids(self, X, y):
		"""
		Computes the SBM model using centroids to select prototypes.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None

		# Selecting only positive class members
		X_true = X[y, :]

		# Starting D and choosing the current sample
		cur_smp = fast_median(X_true)[np.newaxis, :]
		D = cur_smp.copy()

		# While sample are found in X
		while(X_true.shape[0]):

			# Testing all sample against current sample
			sim_score = psimilarity(X_true, cur_smp, self.kernel, self.gamma,\
				norm=self.norm, icov_mtx=self.icov_mtx)

			# Discarding samples given the threshold
			dis_idx = np.ravel(sim_score < self.tau)
			X_true = X_true[dis_idx, :]

			# Testing dimension
			if (X_true.shape[0] <= 0):

				# End loop
				break

			# Compute the current set median and add to D
			cur_smp = fast_median(X_true)[np.newaxis, :]
			D = np.r_[D, cur_smp]

		# Computing matrix G given the selected model
		G = self.compute_G(D)

		# Saving data and similarity matrices
		self.D = D.copy()
		self.G = G.copy()

		# Computing theta
		self.theta, cur_acc = self.find_theta(X, y)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: {}".format(cur_acc))

	def fit_perceptron(self, X, y):
		"""
		Computes the SBM model using the perceptron algorithm to choose
		prototypes.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None
		bstD = None
		bstG = None

		# Breaking training samples into training and validation sets
		Xtr, Xvl, ytr, yvl = self.validation_split(X, y)

		# Selecting only positive (true) class elements
		X_true = Xtr[ytr > 0, :]

		# Setting initial error
		bst_err = 1
		cur_err = 1

		# Setting stop criteria
		max_itr = self.max_itr

		# Setting initial vector and theta
		init_smp = fast_median(X_true)[np.newaxis, :]
		init_idx = np.argmax(psimilarity(init_smp, X_true,\
			norm=self.norm, icov_mtx=self.icov_mtx).ravel())
		h = np.zeros(X_true.shape[0])
		h[init_idx] = 1.0
		theta = 0.0
		step = 1e-2

		# Regularization parameters
		lmbd = 1e-1
		alpha = 0.5

		# Setting stop criteria
		stp_itr = 10 # Max. number of consecutive iterations without improvement
		cur_stp = 0 # Current number of iterations without improvement

		# Starting iterative procedure
		for itr_idx in range(max_itr):

			# For each sample
			for smp_idx in range(Xtr.shape[0]):

				# Extracting current sample and label
				x = Xtr[smp_idx, :]
				y_cur = ytr[smp_idx]

				# Computing similarity vector
				sim_vec = psimilarity(x[np.newaxis, :], X_true, norm=self.norm,\
							icov_mtx=self.icov_mtx)
				sim_vec = np.squeeze(sim_vec)

				# Predicting current sample
				x_hat = np.sum(h*sim_vec*X_true.T, axis=1)/np.sum(h*sim_vec)
				cur_sim = ssimilarity(np.linalg.norm(x-x_hat, ord=self.norm),\
					self.kernel, self.gamma)

				# Testing for prediction error
				if (y_cur != (cur_sim > theta)):

					# Updating theta
					theta = theta - step*(2*y_cur-1)

					# Computing diffential similarity matrix
					dff_mat = -ssimil_diff(X_true, x, self.kernel, self.gamma)

					# Computing product matrices
					dff_smp = np.dot(dff_mat, X_true.T)
					h_sim = h*sim_vec
					h_sim_p = np.sum(h_sim)

					# Computing gradient
					if h_sim_p**2 > np.finfo('d').eps:

						# Normal mode
						h_grad = np.dot(dff_smp, h_sim)
						h_grad = h_sim_p*sim_vec*np.diag(dff_smp) - h_grad
						h_grad = h_grad/(h_sim_p**2)

					else:

						# Just set to zero
						h_grad = 0.0*h

					# Regularization factor
					h_nrm = np.linalg.norm(h)
					h_reg = (1-alpha)*h/h_nrm if h_nrm > 1e-12 else 0.0*h
					h_reg = h_reg + alpha*np.sign(h)

					# Computing final gradient and updating h
					h_grad = - (1.0*y_cur)*h_grad + lmbd*h_reg
					h = h - step*h_grad

			# Updating the system
			theta = min(max(theta, 0.0), 1.0)
			h = np.maximum(0.0, h)

			# Finding the chosen indices
			h_idx = np.where(h>0)[0]

			# Testing for a empty vector
			if h_idx.size <= 0:

				# Select all elements, restarting the vector
				h_idx = np.array([init_idx])
				h = np.zeros(X_true.shape[0])
				h[init_idx] = 1.0

			# Extracting from select data
			D = X_true[h_idx, :]
			G = self.compute_G(D)

			# Predicting using current data
			self.D = D.copy()
			self.G = G.copy()
			self.theta, _ = self.find_theta(Xtr, ytr)

			# Computing prediction error
			cur_err = 1-accuracy_score(yvl, self.predict(Xvl))

			# Testing if verbose
			if self.verbose:

				# Print current loop error
				print("Loop {0} error: {1}.".format(itr_idx, cur_err))

			# Testing if current results are the best found
			if (bst_err >= cur_err):

				# Updating data
				if (bst_err > cur_err) or (bstD.shape[0] > D.shape[0]):
					bstD = D.copy()
					bstG = G.copy()
					bst_err = cur_err

					# Reseting the number of iterations without any improvement
					cur_stp = 0
					step = step/2.0

				else:

					# Incrementing the number of iterations without improvement
					cur_stp = cur_stp+1
					step = step*2.0

			else:

				# Incrementing the number of iterations without improvement
				cur_stp = cur_stp+1
				step = step*2.0

			# Testing stop conditions
			stop_con = (cur_err < self.min_err) or (cur_stp >= stp_itr)
			if stop_con:
				# End loop
				break

		# Saving current data and similarity matrices
		self.D = bstD.copy()
		self.G = bstG.copy()

		# Computing theta
		self.theta, cur_acc = self.find_theta(Xtr, ytr)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: ", cur_acc)

	def fit_greedy(self, X, y):
		"""
		Computes SBM model using original approach to select prototypes
		proposed in [Bien2011].

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None

		# Finding prototypes
		Z = X[y, :]
		protos = self.bien_2011(X, y, Z)

		# Saving prototypes
		self.D = Z[protos, :].copy()
		self.G = self.compute_G(self.D)
		self.theta, cur_acc = self.find_theta(X, y)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: ", cur_acc)

	def fit_rgreedy(self, X, y):
		"""
		Computes SBM model using original approach to select prototypes
		proposed in [Bien2011] with random subsampling.

		@param X Input matrix [n_samples, n_features].
		@param y Binary logical labels vector [n_samples].
		"""

		# Setting initial data
		D = None
		G = None

		# Computing number of random subsampling
		n_samples = X.shape[0]
		nsub = int(n_samples/float(self.sampling))

		# For each random subsampling turn
		protos = []
		for rssidx in range(10):

			# Extracting random samples
			while (True):

				# Random draw
				cur_ridx = np.random.random_integers(0, n_samples-1, nsub)
				Xsub = X[cur_ridx, :]
				ysub = y[cur_ridx]

				# Drawing prototypes
				cur_ridx = cur_ridx[ysub]
				if cur_ridx.size == 0:
					continue
				Zsub = X[cur_ridx, :]

				# Computing prototypes
				proto_idx = self.bien_2011(Xsub, ysub, Zsub)
				if len(proto_idx) > 0:
					break

			# Saving prototypes
			cur_ridx = cur_ridx[proto_idx]
			protos.extend(cur_ridx.tolist())

		# Finding unique elements
		protos = np.unique(protos)

		# Computing full set prototypes
		Z = X[protos, :]
		protos = self.bien_2011(X, y, Z)

		# Saving prototypes
		self.D = Z[protos, :].copy()
		self.G = self.compute_G(self.D)
		self.theta, cur_acc = self.find_theta(X, y)

		# Testing if verbose
		if self.verbose:

			# Print training accuracy
			print("Training accuracy: ", cur_acc)

	def estimate(self, X):
		"""
		Estimates input data using similarity model.

		@param X Input matrix [n_samples, n_features].

		@return Input data estimative [n_samples, n_features].
		"""

		# Extracting data and similarity class for the target class
		D = self.D
		G = self.G

		# Finding the number of samples
		if X.ndim > 1:
			n_samples = X.shape[0]
		else:
			n_samples = 1

		# Transforming input
		X = self.std_scl.transform(X)

		# Computing similarity between input and D data
		if n_samples > 2000: # Recursive

			# Breaking X in two sides
			Xl = X[:2000,:]
			Xr = X[2000:,:]

			# Computing left
			W = psimilarity(A=D, B=Xl, kernel=self.kernel, gamma=self.gamma,\
			norm=self.norm, icov_mtx=self.icov_mtx)

			# Recursive call
			W = np._r[W, self.estimate(Xr)]

		else: # Normal transformation
			W = psimilarity(A=D, B=X, kernel=self.kernel, gamma=self.gamma,\
			norm=self.norm, icov_mtx=self.icov_mtx)

		# Finding weights by transforming by G inverse
		W = np.dot(G, W)

		# Computing weights sum and regularizing
		sumW = np.sum(W, axis=0)
		sumW = sumW + 1e-9*(sumW < 1e-9)
		sumW = sumW[np.newaxis]

		# Normalizing
		W = W/sumW

		# Estimate by a weighted sum of prototypes
		Xhat = np.dot(W.T, D)

		# Estimate by searching prototype with max similarity
		#Xhat = D[W.argmax(axis=0)]

		# Return
		Xhat = self.std_scl.inverse_transform(Xhat)
		return Xhat

	def decision_function(self, X):
		"""
		Returns input data similarity using the similarity model.

		@param X Input matrix [n_samples, n_features].

		@return Similarity with the positive class [n_samples].
		"""

		# Computing estimative
		X = self.std_scl.transform(X)
		Xhat = self.std_scl.transform(self.estimate(X))

		# Computing similarity between the original and the estimate
		if self.norm == 'mahalanobis':
			Xsim = np.linalg.norm(X-Xhat, ord=2, axis=1)
		else:
			Xsim = np.linalg.norm(X-Xhat, ord=self.norm, axis=1)
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

	def get_w(self, X):
		"""
		Returns the decision vectors for each class.

		@param X Input matrix [n_samples, n_features].

		@return Similarity per prototype [n_samples, n_prototypes].
		"""

		# Extracting data and similarity class for the target class
		D = self.D
		G = self.G

		# Computing similarity between input and D data
		X = self.std_scl.transform(X)
		W = psimilarity(A=D, B=X, kernel=self.kernel, gamma=self.gamma,\
			norm=self.norm, icov_mtx=self.icov_mtx)

		# Finding weights by transforming by G inverse
		W = np.dot(G, W)

		# Return
		return W

	def get_proto(self):
		"""
		Returns current prototypes.

		@return Return original scale prototypes.
		"""

		# Returning
		return self.std_scl.inverse_transform(self.D)

	def transform(self, X):
		"""
		Returns the data residual against the estimated using the similarity
		model.

		@param X Input matrix [n_samples, n_features].

		@return Transformed data.
		"""

		# Computing the transformation
		if self.feat=="similarity":

			# Similarity score
			Xsim = self.decision_function(X)
			Xsim = Xsim[:, np.newaxis]

		elif self.feat=="residue":

			# Just the residue
			Xsim = X - self.estimate(X)

		elif self.feat=="all":

			# All the previous cases
			Xsim = self.decision_function(X)
			Xsim = Xsim[:, np.newaxis]
			Xsim = np.c_[Xsim, X - self.estimate(X)]

		else:

			# Error
			raise NameError("Invalid option. Valid options are \'similarity\'"+\
			"\'residue\' or \'all\'")

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
# Class BSBM
#*******************************************************************************

class BSBM(BaseEstimator, TransformerMixin, ClassifierMixin):
	"""
	Binary Similarity Based-Modeling from Wegerich2003.
	Wrapper for unary SBM with two class strategy for binary support.
	"""

	def __init__(self, model="SBM", method="kmeans", kernel='IES',\
		min_err=1e-2, max_itr=100, gamma=1e-2, tau=0.15, sampling=5,\
			norm=2, feat="similarity", scl=False, verbose=False, n_jobs=1):
		"""
		Inits SBM.

		@param model Model type.
		@param method Training method.
		@param kernel Similarity kernel.
		@param min_err Minimum error threshold stop criterium.
		@param max_itr Maximum number of iterations stop criterium.
		@param gamma Similarity kernel auxiliary parameter.
		@param tau Similarity score threshold value for prototype selection.
		@param sampling Subsampling factor for prototype selection.
		@param norm Norm applied on distance calculation for similarity.
		@param feat Transformation feat (similarity score or residual).
		@param scl Z-score scaler option.
		@param verbose Show training messages (verbose mode).
		@param n_jobs Number of parallel. Not implemented.
		"""

		# Model parameters
		self.model = model
		self.method = method
		self.kernel = kernel
		self.gamma = gamma

		# Setting training parameters
		self.min_err = min_err
		self.max_itr = max_itr
		self.tau = tau
		self.sampling = sampling
		self.norm = norm

		# Setting transformation feat
		self.feat = feat

		# Setting scaler
		self.scl = scl

		# Setting verbose mode
		self.verbose = verbose

		# Setting model
		self.label_binarizer_ = None
		self.estimators_ = []

		# Setting number of jobs
		self.n_jobs = n_jobs

	def fit(self, X, y=None):
		"""
		Computes SBM model.

		@param X Input matrix [n_samples, n_features].
		@param y Classes labels [n_samples].

		@return self
		"""

		# Find classes
		self.label_binarizer_ = LabelBinarizer(sparse_output=True)
		Y = self.label_binarizer_.fit_transform(y)
		Y = Y.tocsc()
		Y = Y.toarray().ravel()

		# Saving labels
		self.classes_ = self.label_binarizer_.classes_

		# Training models
		self.estimators_ = []

		# First model
		self.estimators_.append(USBM(**self.get_params()))
		self.estimators_[-1].fit(X, 1-Y)

		# Second model
		self.estimators_.append(USBM(**self.get_params()))
		self.estimators_[-1].fit(X, Y)

		# Return self for sklearn API
		return self

	def estimate(self, X):
		"""
		Estimates input data using similarity model.

		@param X Input matrix [n_samples, n_features].

		@return Input data estimative [n_samples, n_classes*n_features].
		"""

		# Extracting trained models
		estimators = self.estimators_

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

		# Softmax approach
		estimators = self.estimators_
		sim_matrix = [e.decision_function(X)[:, np.newaxis] for e in estimators]
		sim_matrix = np.concatenate(sim_matrix, axis=1)
		#sim_matrix = np.exp(sim_matrix) # Transforming it on exponentials
		#sim_matrix = sim_matrix/(sim_matrix.sum(axis=1, keepdims=True)+1e-9)

		return sim_matrix

	def predict(self, X):
		"""
		Predict class labels for samples in X.

		@param X Input matrix [n_samples, n_features].

		@return Predicted class label per sample [n_samples].
		"""

		# Just compute the predicted class and return
		y = np.argmax(self.decision_function(X), axis = 1)
		y = self.label_binarizer_.inverse_transform(y)
		return y

	def get_w(self, X):
		"""
		Returns the decision vectors for each class.

		@param X Input matrix [n_samples, n_features].

		@return Two elements:
			- Similarity per prototype [n_samples, n_prototypes].
			- Prototypes classes [n_prototypes].
		"""

		# Extracting trained models
		estimators = self.estimators_

		# For each class compute decision vectors.
		W = [e.get_w(X) for e in estimators]
		W = np.concatenate(W, axis=0)

		# Finding each prototype class
		classes = self.label_binarizer_.classes_
		classes = [classes[i]*np.ones(e.D.shape[0]) for i,e in enumerate(estimators)]
		classes = np.hstack(classes)
		classes = classes.tolist()

		# Returning
		return W, classes

	def get_proto(self):
		"""
		Returns current prototypes.

		@return Original scale prototypes [n_samples, n_prototypes].
		"""

		# Extracting trained models
		estimators = self.estimators_

		# Saving prototypes list
		D = [e.get_proto() for e in estimators]

		# Returning
		return D

	def transform(self, X):
		"""
		Returns the data residual against the estimated using the similarity
		model.

		@param X Input matrix [n_samples, n_features].

		@return Transformed data.
		"""

		# Computing the transformation
		if self.feat=="similarity":

			# Similarity score
			Xsim = self.decision_function(X)

		elif self.feat=="residue":

			# Extracting trained models
			estimators = self.estimators_

			# Computing the residue for each class
			Xsim = [X-e.estimate(X) for e in estimators]
			Xsim = np.concatenate(Xsim, axis=1)

		elif self.feat=="all":

			# All previous cases
			estimators = self.estimators_
			Xsim = [X-e.estimate(X) for e in estimators]
			Xsim.append(self.decision_function(X))
			Xsim = np.concatenate(Xsim, axis=1)

		else:

			# Error
			raise NameError("Invalid option. Valid options are \'similarity\'"+\
			"\'residue\' or \'all\'")

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
# Class MSBM
#*******************************************************************************

class MSBM(BaseEstimator, TransformerMixin, ClassifierMixin):
	"""
	Multiclass Similarity Based-Modeling from Wegerich2003.
	Wrapper for unary SBM with one-vs-the-rest strategy for multiclass support.
	"""

	def __init__(self, model="SBM", method="kmeans", kernel='IES',\
		min_err=1e-2, max_itr=100, gamma=1e-2, tau=0.15, sampling=5,\
			norm=2, feat="similarity", scl=False, verbose=False, n_jobs=1):
		"""
		Inits SBM.

		@param model Model type.
		@param method Training method.
		@param kernel Similarity kernel.
		@param min_err Minimum error threshold stop criterium.
		@param max_itr Maximum number of iterations stop criterium.
		@param gamma Similarity kernel auxiliary parameter.
		@param tau Similarity score threshold value for prototype selection.
		@param sampling Subsampling factor for prototype selection.
		@param norm Norm applied on distance calculation for similarity.
		@param feat Transformation feat (similarity score or residual).
		@param scl Z-score scaler option.
		@param verbose Show training messages (verbose mode).
		@param n_jobs Number of parallel. See OneVsRestClassifier.
		"""

		# Model parameters
		self.model = model
		self.method = method
		self.kernel = kernel
		self.gamma = gamma

		# Setting training parameters
		self.min_err = min_err
		self.max_itr = max_itr
		self.tau = tau
		self.sampling = sampling
		self.norm = norm

		# Setting transformation feat
		self.feat = feat

		# Setting scaler
		self.scl = scl

		# Setting verbose mode
		self.verbose = verbose

		# Setting model
		self.mdl = None

		# Setting number of jobs
		self.n_jobs = n_jobs

	def fit(self, X, y=None):
		"""
		Computes SBM model.

		@param X Input matrix [n_samples, n_features].
		@param y Classes labels [n_samples].

		@return self
		"""

		# Setting model
		if len(np.unique(y)) <= 2:

			# Use binary SBM
			self.mdl = BSBM(**self.get_params())

		else:
			# Multiclass
			self.mdl = OneVsRestClassifier(USBM(**self.get_params()),\
				self.n_jobs)

		# Fit
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

		# Testing softmax approach
		sim_matrix = self.mdl.decision_function(X) # Saving similarities

		if sim_matrix.shape[1] == 1:
			sim_matrix = np.c_[sim_matrix,(1 - sim_matrix)]

		# Normalizing
		#sim_matrix = np.exp(sim_matrix) # Transforming it on exponentials
		#sim_matrix = sim_matrix/(sim_matrix.sum(axis=1, keepdims=True)+1e-9)

		return sim_matrix

	def predict(self, X):
		"""
		Predict class labels for samples in X.

		@param X Input matrix [n_samples, n_features].

		@return Predicted class label per sample [n_samples].
		"""

		# Just compute the predicted class and return
		return self.mdl.predict(X)

	def get_w(self, X):
		"""
		Returns the decision vectors for each class.

		@param X Input matrix [n_samples, n_features].

		@return Two elements:
			- Similarity per prototype [n_samples, n_prototypes].
			- Prototypes classes [n_prototypes].
		"""

		# Extracting trained models
		estimators = self.mdl.estimators_

		# For each class compute decision vectors.
		W = [e.get_w(X) for e in estimators]
		W = np.concatenate(W, axis=0)

		# Finding each prototype class
		classes = self.mdl.classes_
		classes = [\
		classes[i]*np.ones(e.D.shape[0]) for i,e in enumerate(estimators)]
		classes = np.hstack(classes)
		classes = classes.tolist()

		# Returning
		return W, classes

	def get_proto(self):
		"""
		Returns current prototypes.

		@return Original scale prototypes [n_samples, n_prototypes].
		"""

		# Extracting trained models
		estimators = self.mdl.estimators_

		# Saving prototypes list
		D = [e.get_proto() for e in estimators]

		# Returning
		return D

	def transform(self, X):
		"""
		Returns the data residual against the estimated using the similarity
		model.

		@param X Input matrix [n_samples, n_features].

		@return Transformed data.
		"""

		# Computing the transformation
		if self.feat=="similarity":

			# Similarity score
			Xsim = self.decision_function(X)

		elif self.feat=="residue":

			# Extracting trained models
			estimators = self.mdl.estimators_

			# Computing the residue for each class
			Xsim = [X-e.estimate(X) for e in estimators]
			Xsim = np.concatenate(Xsim, axis=1)

		elif self.feat=="all":

			# All previous cases
			estimators = self.mdl.estimators_
			Xsim = [X-e.estimate(X) for e in estimators]
			Xsim.append(self.decision_function(X))
			Xsim = np.concatenate(Xsim, axis=1)

		else:

			# Error
			raise NameError("Invalid option. Valid options are \'similarity\'"+\
			"\'residue\' or \'all\'")

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
