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
# interp.py
#
# Missing data interpolation/extrapolation methods.
#
# Created:		2015/10/15
# Modified:		2015/10/19
#
#*******************************************************************************

"""
interp.py

Missing data interpolation/extrapolation methods.
"""

#*******************************************************************************
# Imports
#*******************************************************************************

import numpy as np # Numpy library (numeric algebra)

#*******************************************************************************
# Functions
#*******************************************************************************

def linear_interp(x, indices=None):
	"""
	Interpolates the given series missing data using linear interpolation.
	
	@param x Input data series. Unidimensional.
	@param indices Sample indices. If none, considers equally spaced series.
	
	@return Interpolated series.
	"""
	
	# Testing indices
	if indices is None:
		
		# Extract indices from the series
		indices = np.arange(len(x))
	
	# Finding valid elements
	not_nan = np.logical_not(np.isnan(x))
	
	# Interpolating
	xi = np.interp(indices, indices[not_nan], x[not_nan])
	
	# Returning
	return xi