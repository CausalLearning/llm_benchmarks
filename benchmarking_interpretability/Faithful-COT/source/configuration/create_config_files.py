'''Automatically generate the configuration files for a specified range of hyperparameters.'''
import os
if os.getcwd().endswith('configuration'):
	os.chdir('../..')
import sys
sys.path.append("source")
from itertools import product
import argparse
import json

'''
TO RUN:
--dataset_names "['AQUA', 'ASDiv', 'GSM8K', 'MultiArith', 'SVAMP', 'StrategyQA', 'date', 'sports', 'saycan', 'CLUTRR']"
--LMs "['gpt4']" 
--prompt_names "['NL+SL']"
--ns_votes "[40]"
--batch_sizes "[10]"
'''

if __name__ == "__main__":
	# config
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_names", help="the names of the datasets")
	Parser.add_argument("--LMs", help="the underlying LM")
	Parser.add_argument("--prompt_names", help="the name of the prompt (and template)")
	Parser.add_argument("--ns_votes", help="the numbers of votes")
	Parser.add_argument("--batch_sizes", help="number of completions in one query.")

	model_name_mapping = {"code001": "code-davinci-001",
	                      "code002": "code-davinci-002",
	                      "text001": "text-davinci-001",
	                      "text002": "text-davinci-002",
	                      "text003": "text-davinci-003",
	                      "gpt4": "gpt-4",
	                      "gpt-3.5-turbo": "gpt-3.5-turbo"
	                      }

	# get the arguments
	args = Parser.parse_args()
	LMs = eval(args.LMs)
	dataset_names = eval(args.dataset_names)
	prompt_names = eval(args.prompt_names)
	ns_votes = eval(args.ns_votes)
	batch_sizes = eval(args.batch_sizes)

	# enumerate all combinations of hyperparameters
	combos = list(product(dataset_names, LMs, prompt_names, ns_votes, batch_sizes))
	for i, combo in enumerate(combos):
		dataset_name, LM, prompt_name, n_votes, batch_size = combo
		config_fn = f"{LM}_{prompt_name}"
		if n_votes != 1:
			config_fn += f"_n:{n_votes}"
		config_fn += ".json"
		fwn = f"source/configuration/config_files/{dataset_name}/{config_fn}"
		# if os.path.exists(fwn):
		# 	continue

		full_model_name = model_name_mapping[LM]
		config = {
			"LM": full_model_name,
			"prompt_name": prompt_name,
			"n_votes": n_votes,
			"batch_size": batch_size,
			"no_solver": True if prompt_name in ["noNL","COT","standard", "LtM"] else False
		}

		with open(fwn, "w") as fw:
			json.dump(config, fw, indent=2)
