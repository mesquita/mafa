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
# cluster.py
#
# Clustering methods.
#
# Created: 2015/10/09
# Modified: 2016/08/12
#
#*******************************************************************************

"""
cluster.py

Clustering methods.
"""

#*******************************************************************************
# Imports
#*******************************************************************************

import numpy as np # Numpy library (numeric algebra)
from scipy.spatial.distance import cdist # Distance computation

#*******************************************************************************
# Functions
#*******************************************************************************

def fast_median(X, step=0.1):
	"""
	Estimates the geometric median of a given set using the fast algorithm
	described in [Cenac20XX]
	
	@param X Input data set [n_samples, n_features].
	@param step Update stepsize value.
	
	@return Geometric median vector [n_features].
	"""
	
	# Using mean as initial value
	gmed_vec = X.mean(axis=0)
	
	# For each sample
	for i in range(X.shape[0]):
		
		# Extracting current sample
		x = X[i, :]
		
		# Computing distance to the median
		diff = x-gmed_vec
		dist = np.linalg.norm(diff)
		
		# Updating the median vector
		if dist > 1e-9:
			
			# Updating median
			gmed_vec = gmed_vec + (step/dist)*diff
			
	# Finding the nearest element in X
	pair_dist = cdist(gmed_vec[np.newaxis, :], X)
	min_idx = np.argmin(pair_dist)
	gmed_vec = X[min_idx, :]
	
	# Return median
	return gmed_vec

def pam(dst_mat, num_cls, max_itr=20):
	"""
	Computes k-medoids clustering using Partition Around Medoids (PAM)
	algorithm.
	
	@param dst_mat Distance matrix.
	@param num_cls Number of cluster centroids.
	@param max_itr Maximum number of iterations.
	
	@return Three elements:
			- Cluster center indices;
			- Data clusters indices;
			- Clustering error.
	"""
	
	# Building phase
	
	# Extracting number of elements
	num_smp = dst_mat.shape[0]
	
	# Computing distance sum
	sum_dst = np.sum(dst_mat, axis=1)
	
	# Finding smallest distance element as first medoid
	cur_mdd = [np.argmin(sum_dst)]
	
	# Unselected elements
	cur_uns = range(num_smp)
	
	# Auxiliary distance matrix ignoring diagonal
	aux_dst = np.copy(dst_mat)
	np.fill_diagonal(aux_dst, np.inf)
	
	# Next medoids
	for k in range(1, num_cls):
		
		# Updating unselected elements
		cur_uns = [i for i in cur_uns if i not in cur_mdd]
		
		# Finding closest distance medoid to each element
		cls_mdd = np.argmin(dst_mat[:, cur_mdd], axis=1)
		cls_dst = np.min(dst_mat[:, cur_mdd], axis=1)
		
		# Finding closest element to each sample and the respective distance
		nxt_ele = np.argmin(aux_dst[:, cur_uns], axis=1)
		nxt_dst = np.min(aux_dst[:, cur_uns], axis=1)
		
		# Computing selection cost
		cost = np.maximum(cls_dst-nxt_dst, 0.0)
		cost = np.bincount(nxt_ele, cost)
		
		# Finding next medoids
		aux_idx = np.argmax(cost)
		cur_mdd.append(cur_uns[aux_idx])
	
	# Sort vector
	cur_mdd.sort()
	
	# Swap phase
	
	# For each iteration
	for i in xrange(max_itr):
		
		# Setting previous iteration indices
		pre_mdd = list(cur_mdd)
		pre_uns = list(cur_uns)
		
		# For each medoid
		for k in range(num_cls):
			
			# Finding closest distance medoid to each element
			cls_mdd = np.argmin(dst_mat[:, cur_mdd], axis=1)
			cls_dst = np.min(dst_mat[:, cur_mdd], axis=1)
			
			# Setting elements from cluster
			csl_ele = (cls_mdd == k)
			
			# Finding second closest cluster distance
			aux_mdd = list(cur_mdd)
			aux_mdd.pop(k)
			aux_dst = np.min(dst_mat[:, aux_mdd], axis=1)
			
			# For each not selected element
			for u in pre_uns:
				
				# Settin auxiliary cluster index
				aux_mdd = list(cur_mdd)
				aux_mdd[k] = u
				
				# Updating unselected elements
				cur_uns = range(num_smp)
				cur_uns = [v for v in cur_uns if v not in aux_mdd]
				
				# Comopute distance
				swp_dst = dst_mat[:, u] - cls_dst
				
				# Outside the cluster cost
				cost = np.minimum(swp_dst, 0)
				
				# Inside cluster cost
				ins_dst = np.minimum(aux_dst, dst_mat[:, u])
				ins_dst = ins_dst-cls_dst
				cost[csl_ele] = ins_dst[csl_ele]
				
				# Computing the total cost
				tcost = np.sum(cost[cur_uns])
				
				# Testing cost
				if tcost < 0.0:
					
					# Swap everthing from the current cluster
					cur_mdd[k] = u
					cur_mdd.sort()
					cls_mdd = np.argmin(dst_mat[:, cur_mdd], axis=1)
					cls_dst = np.min(dst_mat[:, cur_mdd], axis=1)
					csl_ele = (cls_mdd == k)
					
					# And from the second cluster
					aux_mdd = list(cur_mdd)
					aux_mdd.pop(k)
					aux_dst = np.min(dst_mat[:, aux_mdd], axis=1)
					
		# Testing medoids index
		if np.all(np.equal(pre_mdd, cur_mdd)):
			
			# End iterations
			break
	
	# Clustering error
	cls_dst = np.min(dst_mat[:, cur_mdd], axis=1)
	cls_err = np.sum(cls_dst)
	
	# Returning
	return cur_mdd, cls_mdd, cls_err

def clara(dst_mat, num_cls, max_itr=20, num_init=10, num_rnd=80):
	"""
	Computes k-medoids clustering using Clustering LARge Applications (CLARA)
	algorithm.
	
	@param dst_mat Distance matrix.
	@param num_cls Number of cluster centroids.
	@param max_itr Maximum number of iterations.
	@param num_init Number of random initializations.
	@param num_rnd Number of base random samples (Total: num_rnd+2*num_cls).
	
	@return Three elements:
			- Cluster center indices;
			- Data clusters indices;
			- Clustering error.
	"""
	
	# initializing auxiliary variables
	bst_err = float("inf") # Best error
	bst_mdd = None
	cls_mdd = None
	
	# Number of random samples
	itr_smp = num_rnd+2*num_cls
	
	# Total number of samples
	num_smp = dst_mat.shape[0]
	
	# For each initialization
	for i in xrange(num_init):
		
		# Setting current random samples
		rnd_idx = np.sort(np.random.choice(num_smp, itr_smp, False)).tolist()
		
		# Reduced distance matrix with random draw samples
		rdc_dst = dst_mat[:, rnd_idx]
		rdc_dst = rdc_dst[rnd_idx, :]
		
		# Calling PAM
		cur_mdd, _, _ = pam(rdc_dst, num_cls, max_itr)
		
		# Updating current medoids using real indices
		cur_mdd = [rnd_idx[mdd_idx] for mdd_idx in cur_mdd]
		
		# Clustering error
		cls_dst = np.min(dst_mat[:, cur_mdd], axis=1)
		cls_err = np.sum(cls_dst)
		
		# Testing error
		if bst_err > cls_err:
			
			# Updating
			bst_err = cls_err
			bst_mdd = cur_mdd
			cls_mdd = np.argmin(dst_mat[:, bst_mdd], axis=1)
	
	# Returning
	return bst_mdd, cls_mdd, bst_err