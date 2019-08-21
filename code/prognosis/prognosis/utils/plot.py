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
# plot.py
#
# Auxiliary plot methods
#
# Created: 2015/11/12
# Modified: 2015/11/17
#
#*******************************************************************************

"""
plot.py

Plot functions.
"""

#*******************************************************************************
# Imports
#*******************************************************************************
import numpy as np # Numpy library (numeric algebra)
import matplotlib.pyplot as plt # Plotting library

#*******************************************************************************
# Functions
#*******************************************************************************

def plot_spec(X, show=False):
	"""
	Plot a spectrogram of a time-frequency matrix.
	
	@param X time-frequency matrix [2-D array].
	@param show Show flag.
	"""
	timebins, freqbins = np.shape(X)
	
	plt.close('all')
	plt.figure(figsize=(15, 9))
	
	# plt.imshow(np.transpose(10*np.log10(np.absolute(X))), origin="lower",
	# aspect="auto", cmap="jet", interpolation="spline36")
	plt.imshow(10*np.log10(X), origin="lower",aspect="auto", cmap="jet")
			#interpolation="spline36")
	cbar = plt.colorbar()
	cbar.set_label('Power (dB)')
	plt.xlabel("Frequency (samples)")
	plt.ylabel("Time (window)")
	plt.xlim([0, freqbins-1])
	plt.ylim([0, timebins-1])
	
	# Show plot
	if show:
		plt.show()