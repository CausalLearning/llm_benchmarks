'''Dataset utilities.'''

import json
import re
import csv
from collections import Counter
from fractions import Fraction
import math

INVALID_ANS = "[invalid]"

# Stop token for each dataset (faithful COT prompt)
CODE_STOP_TOKEN = {"GSM8K": "# Q:",
                   "SVAMP": "# Q:",
                   "MultiArith": "# Q:",
                   "ASDiv": "# Q:",
                   "AQUA": "# Question:",
                   "saycan": "User",
                   "StrategyQA": "// Q:",
                   "date": "# Q:",
                   "sports": "# Q:",
                   "CLUTRR": "Final answer: "
                  }

# Max number of tokens needed for each dataset (faithful COT prompt)
CODE_MAX_TOKEN = { "GSM8K": 1000,
                   "SVAMP": 1000,
                   "MultiArith": 1000,
                   "ASDiv": 1000,
                   "AQUA": 1000,
                   "saycan": 300,
                   "StrategyQA": 800,
                   "date": 500,
                   "sports": 500,
				   "CLUTRR": 70 # for each step
                  }

NO_CODE_STOP_TOKEN = {"GSM8K": "Q:",
                      "SVAMP": "Q:",
                      "MultiArith": "Q:",
                      "ASDiv": "Q:",
                      "AQUA": "Q: ",
                      "StrategyQA": "Q:",
                      "date": "Q:",
                      "sports": "Q:",
                      "saycan": "Human:",
                      "CLUTRR": "Context:",
                      }

NO_CODE_MAX_TOKEN = {"GSM8K": 500,
                     "SVAMP": 500,
                     "MultiArith": 500,
                     "ASDiv": 500,
                     "AQUA": 500,
                     "saycan": 300,
                     "date": 500,
                     "sports": 500,
                     "StrategyQA": 500,
                     "CLUTRR": 70, # for each step
                     }

# NO_CODE_MAX_TOKEN = {"GSM8K": 1000,
#                      "SVAMP": 1000,
#                      "MultiArith": 1000,
#                      "ASDiv": 1000,
#                      "AQUA": 1000,
#                      "saycan": 300,
#                      "date": 500,
#                      "sports": 500,
#                      "StrategyQA": 800,
#                      "CLUTRR": 70, # for each step
#                      }



def load_data(frn):
	'''Load data from a file.
	:param frn (str): The dataset file name.

	:return: The dataset (a list of examples, each as a dictionary).
	'''
	if frn.endswith(".jsonl"):
		with open(frn, 'r') as fr:
			lines = []
			for i, line in enumerate(fr):
				if line.strip() == "":
					continue
				try:
					lines.append(json.loads(line))
				except json.decoder.JSONDecodeError as e:
					print(f"Error in line {i}: {line}\n {e}")
					exit(-1)
		return lines
	elif frn.endswith(".csv"):
		with open(frn) as fr:
			reader = csv.DictReader(fr)
			return [line for line in reader]

def str2num(answer_str, rounding="int", abs_val=True):
	'''Convert a string to a number.
	@:param answer_str (str): The string to convert.
	@:param rounding (str): The rounding method for the answer. Can be "int", "ceil", or "floor".
	@:param abs_val (bool): Whether to take the absolute value of the answer.

	@:return The converted number.
	'''
	if "/" in answer_str:
		answer_str =  float(sum(Fraction(s) for s in answer_str.split()))
	answer_str = float(answer_str)

	if rounding == "int":
		answer_str = int(answer_str)
	elif rounding == "ceil":
		answer_str = math.ceil(answer_str)
	elif rounding == "floor":
		answer_str = math.floor(answer_str)

	if abs_val:
		answer_str = abs(answer_str)

	return answer_str


def extract_gold_answer(dataset_name, gold_completion):
	'''Extract the gold answer from a completion.
	:param dataset_name (str): The name of the dataset.
	:param gold_completion (str): The gold completion.

	:return: The gold answer.
	'''
	if dataset_name in ["GSM8K", "SVAMP", "MultiArith"]:
		ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
		match = ANS_RE.search(gold_completion)
		if match:
			match_str = match.group(1).strip()
			match_str = match_str.replace(",", "")
			return int(match_str)
		else:
			return INVALID_ANS
	elif dataset_name == "ASDiv":
		# ASDiv has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
		if type(gold_completion) in [tuple, list]: # the gold answer has multiple values
			answer = dict(Counter([int(ans) for ans in gold_completion]))
		else: # the gold answer has a single value
			answer = int(gold_completion)
		return answer
	elif dataset_name in ["date", "CLUTRR"]:
		answer = gold_completion.split("#### ")[-1]
		return answer
	elif dataset_name == "saycan":
		answer = eval(gold_completion)
		return answer
	elif dataset_name in ["StrategyQA"]:
		answer = bool(gold_completion)
		return answer
	elif dataset_name in ["sports"]:
		answer = bool(int(gold_completion))
		return answer
	else:
		return gold_completion

def extract_pred_answer(dataset_name, pred_completion, rounding="int", abs_val=True):
	'''Extract the predicted answer from a completion.
	:param dataset_name (str): The name of the dataset.
	:param pred_completion (str): The predicted completion.
	:param rounding (str): The rounding method for the predicted answer. Can be "int", "ceil", or "floor".
	:param abs_val (bool): Whether to take the absolute value of the predicted answer.

	:return: The predicted answer.
	'''
	if INVALID_ANS in str(pred_completion):
		return INVALID_ANS

	if dataset_name in ["GSM8K", "SVAMP", "MultiArith"]:
		# GSM8K, SVAMP, and MultiArith all have a single-value integer answer
		if type(pred_completion) == int:
			pred_answer = pred_completion
		elif type(pred_completion) == str:
			ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
			match = ANS_RE.search(pred_completion)
			if match:
				match_str = match.group(1).strip()
				match_str = match_str.replace(",", "")
				try:
					pred_answer = str2num(match_str, rounding, abs_val)
				except:
					pred_answer = INVALID_ANS
			else:
				pred_answer = INVALID_ANS
		return pred_answer

	elif dataset_name in ["ASDiv"]:
		# ASDiv has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
		if type(pred_completion) == int:
			return pred_completion
		elif type(pred_completion) == str:
			pred_completion = pred_completion.lstrip("{([").rstrip("]})")
			pred_answers = pred_completion.split(",")
			final_pred_answers = []
			for pred_answer in pred_answers:
				pred_answer = pred_answer.strip().split(":")[-1].strip("'\"")
				try:
					pred_answer = str2num(pred_answer, rounding, abs_val)
					final_pred_answers.append(pred_answer)
				except ValueError:
					continue
			if len(final_pred_answers) > 1:
				return dict(Counter(final_pred_answers))
			elif len(final_pred_answers) == 1:
				return final_pred_answers[0]
			else:
				return INVALID_ANS
		elif type(pred_completion) == dict:
			new_dict = {}
			for key, value in pred_completion.items():
				new_key = str(key)
				new_key = str2num(new_key, rounding, abs_val)
				new_dict[new_key] = value
			return new_dict

	elif dataset_name in ["StrategyQA"]:
		answer = bool(pred_completion)
		return answer

	elif dataset_name in ["sports"]:
		answer = bool(int(pred_completion))
		return answer

	elif dataset_name in ["saycan"]:
		answer = pred_completion.strip()
		return answer

	else:
		return pred_completion