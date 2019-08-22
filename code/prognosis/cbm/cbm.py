#*******************************************************************************
# Universidade Federal do Rio de Janeiro
# Instituto Alberto Luiz Coimbra de Pos-Graduacao e Pesquisa de Engenharia
# Programa de Engenharia Eletrica
# Signal, Multimedia and Telecommunications Lab
#
# Author: Felipe Moreira Lopes Ribeiro, Luiz Gustavo Cardoso Tavares
#
#*******************************************************************************

#*******************************************************************************
# cbm.py
#
# Condition-Based Maintenance
#
# Created: 2016/05/03
# Modified: 2016/09/28
#
#*******************************************************************************

"""
cbm.py

Condition-Based Maintenance
"""

#*******************************************************************************
# Imports
#*******************************************************************************

# For pickle
try:
	import cPickle as pickle
except:
	import pickle

#*******************************************************************************
# Class CBM
#*******************************************************************************

class CBM(object):
	"""
	Condition-Based Maintenance system with modular components wrapper.
	"""
	
	def __init__(self, models_dict={}):
		"""
		Inits CBM.
		
		@param models_dict System models list.
		"""
		
		# Setting models list
		self.models_dict = models_dict
	
	def __setitem__(self, model_key, model):
		"""
		Sets model to the dictionary.
		
		@param model_key Model key.
		@param model New model.
		"""
		
		# Adding
		self.models_dict[model_key] = model
	
	def __getitem__(self, model_key):
		"""
		Get model with a given key.
		
		@param model_key Current model key.
		"""
		
		# Setting model
		cur_mdl = None
		
		# Testing if key exist
		if model_key in model_key:
			
			# Get model
			cur_mdl = self.models_dict[model_key]
		
		# Returning
		return cur_mdl
	
	def save(self, path):
		"""
		Saves current model.
		
		@param path File path.
		"""
		# Opening file
		with open(path, 'wb') as fp:
			
			# Saving on disk
			pickle.dump(self.__dict__, fp, 0)
	
	def load(self, path):
		"""
		Loads a saved model.
		
		@param path File path.
		"""
		# Opening file
		with open(path, 'rb') as fp:
			
			# Loading from disk
			tmp_dict = pickle.load(fp)
			self.__dict__.update(tmp_dict)