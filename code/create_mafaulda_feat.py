#*******************************************************************************
# Universidade Federal do Rio de Janeiro
# Instituto Alberto Luiz Coimbra de Pos-Graduacao e Pesquisa de Engenharia
# Programa de Engenharia Eletrica
# Signal, Multimedia and Telecommunications Lab
#
# Author: Felipe M. L. Ribeiro, Matheus A. Marins
#
#*******************************************************************************

#*******************************************************************************
# create_mafaulda_feat.py
#
# Computes MaFaulDa database features.
#
# Created: 2016/12/05
# Modified: 2017/02/25
#
#*******************************************************************************

"""
create_mafaulda_feat.py

Computes MaFaulDa database features.
"""

#*******************************************************************************
# Imports
#*******************************************************************************

# Basic imports
import numpy as np # Numpy library (numeric algebra)
import random # Random generators
import pandas as pd # Pandas library (csv reader, etc...)
import glob # Unix style pathname pattern expansion
import fnmatch # Filtering filenames
import os # Parsing and walking through the data directory

# SBM
import prognosis.process.feature as feat # Import feature methods
from prognosis.process.data_balance import awgn # Additive white Gaussian noise
from prognosis.process.feature import find_peaks # Peak detection

# Statistics
from scipy.stats import skew, kurtosis, entropy # Kurtosis and Entropy
from sklearn.neighbors import KernelDensity # Kernel density estimator

#*******************************************************************************
# Functions
#*******************************************************************************

def read_mafaulda(path):
	"""
	Loads MaFaulDa database data.

	@param path Database path.

	@return Dataframe including all data features.
	"""

	# Number of frequencies
	num_frq = 3

	# Header names
	sign_nms = ['TCH', 'IAA', 'IRA', 'ITA', 'EAA', 'ERA', 'ETA', 'MIC']
	stat_nms = ["Ent_", "RMS_", "SRA_", "Krt_", "Skw_", "PPK_", "CF_", "ImF_",\
	"MrF_", "ShF_", "KrF_", "Mu_", "SD_"]
	freq_nms = ["F{:02d}_".format(i) for i in range(num_frq)]
	freq_nms.extend(["FC_", "RMSF_", "RVF_"])
	desc_lbs = ["Fr", "RotFreq", "Weigth", "Shift", "Bearing", "Class"]

	# Computing columns names
	head_nms = []
	head_nms.extend([p+s for p in stat_nms for s in sign_nms])
	head_nms.extend([p+s for p in freq_nms for s in sign_nms])
	head_nms.extend(desc_lbs)

	# Status types
	sts_type = ['normal', 'imbalance', 'horizontal-misalignment',\
	'vertical-misalignment', 'underhang', 'overhang']

	# Pillow faults types
	pll_type = ['cage_fault', 'outer_race', 'ball_fault']

	# Get filenames
	filenames = []
	for root, dirnames, fnames in os.walk(path):
		for fname in fnmatch.filter(fnames, '*.csv'):
			filenames.append(os.path.join(root, fname))

	# Output data
	out_data = []

	# Parsing filenames
	for fn in filenames:

		# Creating description vector
		desc_vec = np.zeros(5) # f, w, m, b, lbls

		# Parsing filename
		aux_fn = os.path.normpath(fn)
		aux_fn = aux_fn.split(os.sep)

		# Extracting frequency
		desc_vec[0] = float(aux_fn.pop()[:-4]) # Frequency

		# Finding class
		cur_clss = aux_fn[int(np.argwhere([s in sts_type for s in aux_fn]))]
		cur_clss = sts_type.index(cur_clss)
		desc_vec[4] = cur_clss

		# Finding the pillow faults
		cur_pll = np.argwhere([s in pll_type for s in aux_fn])
		if (cur_pll.size):
			cur_pll = aux_fn[int(cur_pll)]
			cur_pll = pll_type.index(cur_pll)

		# Testing for each type
		if cur_clss == 1: # Imbalance

			# Extracts weigth
			desc_vec[1] = float(aux_fn.pop()[:-1]) # Weigth

		elif cur_clss in [2,3]: # Misalignment

			# Extracts misalignment
			aux_val = aux_fn.pop()[:-2].replace(',', '.')
			desc_vec[2] = float(aux_val) # Misalignment

		elif cur_clss in [4,5]: # Bearing fault

			# Extracting weight and bearing fault
			desc_vec[1] = float(aux_fn.pop()[:-1]) # Weigth
			desc_vec[3] = cur_pll # Pillow fault

		else:

			# Do nothing
			pass

		#  Reading data and converting to float
		cur_raw = pd.read_csv(fn, header=None, names=sign_nms)
		cur_raw = cur_raw.values.astype(float)

		# Computing features
		proc_vec = pproc(cur_raw, num_frq=num_frq)
		proc_vec = np.r_[proc_vec, desc_vec]
		out_data.append(proc_vec.copy())

	# Concatenating, converting to dataframe and returning
	out_data = np.r_[out_data]
	out_data = pd.DataFrame(out_data, columns=head_nms)
	return out_data

def data_entropy(X, n_grid=1000, kernel=False, bandwidth=0.2, **kwargs):
	"""
	Computes unidimensional entropy from data points.

	@param X Input matrix [n_samples, n_features].
	@param n_grid Number of grid points. Integer.
	@param kernel Boolean to set kernel method on/off.
	@param bandwidth Kernel bandwidth. Scalar.

	@return Vector of [n_features], with the corresponding feature entropy.
	"""

	# Testing X dimension
	if (len(X.shape) > 1):

		# Computing per axis
		ent_h = np.apply_along_axis(data_entropy, 0, X, n_grid, kernel,\
			bandwidth, **kwargs)
		ent_h = np.array(ent_h)

	else:

		# Finding sample range
		x_max = X.max()
		x_min = X.min()

		# Testing for kernel method
		if kernel:

			# Computing random sampling
			rnd_idx = np.random.choice(X.shape[0], size=n_grid,\
				replace=False)

			# Kernel density estimation
			kde = KernelDensity(bandwidth=bandwidth, **kwargs)
			kde.fit(X[rnd_idx, np.newaxis])

			# Computing distro
			x_grid = np.linspace(x_min, x_max, n_grid)
			pdf = kde.score_samples(x_grid[:, np.newaxis]) # Log-likelihood
			pdf = np.exp(pdf) # Distribution estimation

		else:

			# Computing grid
			x_grid = np.arange(x_min, x_max, bandwidth)

			# Computing histogram
			pdf, _ = np.histogram(X, bins=x_grid, density=True)

		# Computing entropy
		ent_h = entropy(pdf)

	# Return entropy
	return ent_h

def stat_feat(data):
	"""
	Computes the statistical feats from the raw data. Based on the features
	found in [Rauber2015].

	@param data Input data. Numpy matrix.

	@retun Statistical features.
	"""

	# Computing auxiliary values
	rms = np.sqrt(np.mean(data**2, axis=0)) # RMS value
	adt = np.abs(data) # Data absolute value
	sra = np.mean(np.sqrt(adt), axis=0)**2 # Square root of the amplitude
	krt = kurtosis(data, axis=0)

	# Computing the statistical features
	sts_vec = []
	sts_vec.append(data_entropy(data)) # Entropy
	sts_vec.append(rms) # RMS value
	sts_vec.append(sra) #SRA
	sts_vec.append(krt) # Kurtosis
	sts_vec.append(skew(data, axis=0)) # Skew
	sts_vec.append(data.max(axis=0) - data.min(axis=0)) # Peak2peak value
	sts_vec.append(adt.max(axis=0)/rms) # Crest factor
	sts_vec.append(np.max(adt, axis=0)/np.mean(adt, axis=0)) # Impulse factor
	sts_vec.append(adt.max(axis=0)/sra) # Margin factor
	sts_vec.append(rms/np.mean(adt, axis=0)) # Shape factor
	sts_vec.append(krt/rms) # Kurtosis factor
	sts_vec.append(data.mean(axis=0)) # Mean value
	sts_vec.append(data.std(axis=0)) # Standard Deviation

	# Converting statistical features to numpy array
	sts_vec = np.r_[sts_vec].ravel()

	# Returning
	return sts_vec

def freq_feat(data, smp_frq=5e+4, frq_min=5, frq_max=700, num_frq=3, max_hrm=3):
	"""
	Computes features from the raw data.

	@param data Input data list. Numpy matrices list.
	@param smp_frq Number of points. Scalar.
	@param frq_min Minimum target frequency. Scalar.
	@param frq_max Maximum target frequency. Scalar.
	@param num_frq Number of frequency on the interval.
	@param max_hrm Maximum harmonic on the interval.

	@return Preprocessed data.
	"""

	# Finding the number of signals and samples
	signals = range(data.shape[1])
	num_smp = data.shape[0]

	# Computing real DFT transform for each signal
	aux_data = [np.fft.rfft(data[:, sidx]) for sidx in signals]

	# Setting target frequency range
	frq_vls = np.linspace(0, smp_frq/2.0, num=(num_smp/2)+1)
	frq_idx = np.logical_and(frq_vls >= frq_min, frq_vls <= frq_max)
	frq_vls = frq_vls[frq_idx]

	# Removing invalid frequencies
	aux_data = [np.real(np.abs(a[frq_idx])) for a in aux_data]

	# Detect peaks from the tachometer
	pidx, _ = find_peaks(aux_data[0], mpd=15)

	# Extract smallest frequency index and value (first harmonic)
	fidx = np.min(pidx)
	frot = frq_vls[fidx]

	# Setting the maximum harmonic
	max_fidx = min(max_hrm*fidx, aux_data[0].shape[0]-4)

	# Harmonics index
	hfidx = np.int64(np.round(np.linspace(fidx, max_fidx, num_frq)))
	hfidx = hfidx.tolist()

	# Initializing the features vector
	frq_vec = []

	# For each signal
	for sidx, csgn in enumerate(aux_data):

		# Extracting harmonic components
		for ih, hf in enumerate(hfidx):

			# Extracting harmonic components
			rhf = range(hf-3, hf+4)
			frq_vec.append(np.max(csgn[rhf]))

	# Computing auxiliary values
	fmu = [a.mean() for a in aux_data]

	# Computing the statistical features
	frq_vec.extend(fmu)
	frq_vec.extend([np.sqrt(np.mean(a**2)) for a in aux_data])
	frq_vec.extend([a.std() for a in aux_data])
	frq_vec.append(frot)
	frq_vec = np.r_[frq_vec].ravel()

	# Output
	return frq_vec

def pproc(data, smp_frq=5e+4, frq_min=5, frq_max=700, num_frq=3, max_hrm=3):
	"""
	@param data Input data list. Numpy matrices list.
	@param smp_frq Number of points. Scalar.
	@param frq_min Minimum target frequency. Scalar.
	@param frq_max Maximum target frequency. Scalar.
	@param num_frq Number of frequency on the interval.
	@param max_hrm Maximum harmonic on the interval.

	@return Preprocessed data.
	"""

	# Normalizing
	data = data/data.std(axis=0)

	# Computing the features
	sts_vec = stat_feat(data)
	frq_vec = freq_feat(data, smp_frq, frq_min, frq_max, num_frq, max_hrm)

	# Computing output and returning
	output = np.r_[sts_vec, frq_vec]

	# Return feature vector
	return(output)

#*******************************************************************************
# Main
#*******************************************************************************

if __name__ == '__main__':

	# Main parameters
	path = '../../data/rkit/MaFaulDa/' # Main path
	feat_path = "../../data/rkit/data.csv" # Output feature path

	# Frequency Parameters
	frq_min = 5 # Lowest frequency
	frq_max = 700 # Highest frequency
	frq_smp = 5e+4 # Sampling frequency

	# Reading data and saving
	data = read_mafaulda(path)
	data.to_csv(feat_path, index_label='Sample')
