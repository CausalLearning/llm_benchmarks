'''
Model configuration.
Adapted from http://blender.cs.illinois.edu/software/oneie.
'''

import copy
import json
import os
from typing import Dict

class Config(object):
	def __init__(self, **kwargs):
		# dataset parameters
		self.dataset_name = None # name of evaluation dataset
		self.split = None # split of evaluation dataset

		# core parameters
		self.prompt_name = kwargs.get('prompt_name', None) # name of the prompt; should be one of the names from the `prompt/` folder, e.g. "subqeustion_dependency"
		self.LM = kwargs.get('LM', "code-davinci-002") # the underlying LM Translator
		self.max_tokens = kwargs.get('max_tokens', None) # maximum number of tokens to generate

		# decoding parameters
		self.n_votes = kwargs.get('n_votes', 1)  # number of reasoning chains to generate; default to 1 (greedy decoding)
		self.temperature = kwargs.get('temperature', 0.0)  # temperature for the LM ; default to 0.0 (greedy decoding)
		self.batch_size = 1 # number of examples to query the LM at a time

		# analysis-related parameters
		self.no_solver = kwargs.get('no_solver', False) # whether to use the LM to solve the answer instead of calling the solver

		# API keys; default to empty
		self.api_keys = []
		self.org_ids = []

	@classmethod
	def from_dict(cls, dict_obj):
		"""Creates a Config object from a dictionary.
		Args:
			dict_obj (Dict[str, Any]): a dict where keys are
		"""
		config = cls()
		for k, v in dict_obj.items():
			setattr(config, k, v)
		return config

	@classmethod
	def from_json_file(cls, path):
		with open(path, 'r', encoding='utf-8') as r:
			return cls.from_dict(json.load(r))

	def to_dict(self):
		output = copy.deepcopy(self.__dict__)
		return output

	def save_config(self, path):
		"""Save a configuration object to a file.
		:param path (str): path to the output file or its parent directory.
		"""
		if os.path.isdir(path):
			path = os.path.join(path, 'config.json')
		print('Save config to {}'.format(path))
		with open(path, 'w', encoding='utf-8') as w:
			w.write(json.dumps(self.to_dict(), indent=2, sort_keys=True))