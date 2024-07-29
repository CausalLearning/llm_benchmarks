import os
cwd = os.getcwd()
if cwd.endswith("source/model"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS
from dataset.utils import CODE_STOP_TOKEN, CODE_MAX_TOKEN, NO_CODE_STOP_TOKEN, NO_CODE_MAX_TOKEN
import sys
from io import StringIO
import openai
import itertools
from model.solver.MWP import math_solver
from model.solver.CLUTRR import CLUTRR_solver
from model.solver.StrategyQA import datalog_solver
from model.solver.saycan import pddl_planner
import errno
import os
import signal
import functools
import re

# The following are packages/funtions for exponential backoff
# (ref. https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff)
# in order to deal with OpenAI API "rate limit reached" errors
from tenacity import (
	retry,
	stop_after_attempt,
	wait_random_exponential,
)

class TimeoutError(Exception):
	pass

def log_retry(state):
	print(f"Retrying: {state.attempt_number}...")

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
	def decorator(func):
		def _handle_timeout(signum, frame):
			raise TimeoutError(error_message)

		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			signal.signal(signal.SIGALRM, _handle_timeout)
			signal.alarm(seconds)
			try:
				result = func(*args, **kwargs)
			finally:
				signal.alarm(0)
			return result
		return wrapper
	return decorator


class Model():
	'''The model class.'''
	def __init__(self, config):
		'''Initialize the model with the given configuration.
		@:param config: the configuration object, see source/configuration/configuration.py for details
		'''
		super(Model, self).__init__()

		# dataset parameters
		self.dataset_name = config.dataset_name # name of evaluation dataset

		# core parameters
		self.LM = config.LM
		self.prompt_name = config.prompt_name
		self.max_tokens = config.max_tokens

		# decoding parameters
		self.n_votes = config.n_votes  # number of programs to generate
		self.temperature = config.temperature  # temperature for the solver LM
		self.batch_size = config.batch_size  # batch size for querying the LM

		# analysis-related parameters
		self.no_solver = config.no_solver # whether to use the LM to solve the answer instead of calling the solver

		# load the prompt and template
		prompt_path = f"source/prompt/{config.dataset_name}/{self.prompt_name}_prompt.txt" # the prompt containing few-shot examples
		template_path = f"source/prompt/{config.dataset_name}/{self.prompt_name}_template.txt" # the template to convert a new example
		with open(prompt_path, 'r', encoding='utf-8') as fr:
			self.prompt = fr.read()
		with open(template_path, 'r', encoding='utf-8') as fr:
			self.template = fr.read()

		# load the API keys
		self.api_keys = itertools.cycle(config.api_keys)

	def predict(self, example_dict: dict, completion_only: bool = False):
		'''Predict the answer to a example.
		@:param example_dict (dict): a dict containing the example, in the format of {"question": question, (and other dataset-specific fields)}
		@:param completion_only (bool): whether to only return the completion, but not the answer

		@:return (dict): a output dict, in the format of {"answer": answer,
														  "completion": the final completion,
														  "completions": a list of all completions
												          }
		'''
		question = example_dict["question"]

		# apply the template to the question
		templated_example = self._apply_template(template=self.template, example=example_dict)
		# concatenate the few-shot prompt and the example
		prompt_and_example = f"{self.prompt}\n\n{templated_example}"

		# get the stop token for the current dataset
		if self.no_solver:
			stop_token = NO_CODE_STOP_TOKEN[self.dataset_name] # use the stop token for no-code prompts
		else:
			stop_token = CODE_STOP_TOKEN[self.dataset_name] # use the stop token for with-code prompts

		# get the max token for the current dataset
		if self.max_tokens: # if max_tokens is specified, use it
			max_token = self.max_tokens
		else: # otherwise, use the default max_tokens for the current dataset
			max_token = self.get_max_token(self.dataset_name, example_dict)

		# query the LM to get the completions
		n_iters = self.n_votes // self.batch_size # number of iterations to query the LM
		completions = []
		for iter_id in range(n_iters):
			new_completions = self._query(prompt=prompt_and_example,
								   n=self.batch_size,
								   stop=[stop_token],
								   max_tokens=max_token,
								   LM=self.LM,
								   temperature=self.temperature)
			completions += new_completions

		if completion_only: # return only the completions, but not the answer
			output = {"answer": "",
			          "completion": "",
			          "completions": completions
			          }
			return output

		answer, final_completion = self.derive_answer_from_completions(example=example_dict, completions=completions)

		output = {"answer": answer,
				  "completion": final_completion,
				  "completions": completions
		          }
		return output

	def _apply_template(self, template: str, example: dict):
		'''Apply the template to a new example.
		@:param template (str): the template to be applied to the example.
		@:param example (str): the example to be converted into the template format.

		@:return (str): the example converted into the template format.
		'''
		# for every [{FIELD}] in the template, replace it with the corresponding value of the key "{field}" in the example dict
		example_in_template = template
		for field in re.findall(r"\[.*?\]", template):
			field_name = field[1:-1]
			field_name = field_name.lower()
			if field_name in example:
				example_in_template = example_in_template.replace(field, str(example[field_name]))
		return example_in_template

	def get_max_token(self, dataset_name, example):
		'''Get the max token for the current dataset.
		@:param dataset_name (str): the name of the dataset
		@:param example (dict): the example dict

		@:return (int): the max token
		'''
		if self.no_solver:
			max_token_dict = NO_CODE_MAX_TOKEN
		else:
			max_token_dict = CODE_MAX_TOKEN

		if dataset_name == "CLUTRR": # for CLUTRR, the max token depends on the number of steps required (example["k"])
			return max_token_dict[self.dataset_name] * example["k"] # multiply the max token for each step by the number of steps
		else: # for other datasets, the max token is static for each dataset
			return max_token_dict[self.dataset_name]

	@timeout(200)
	def _execute(self, example: dict, completion: str):
		'''Execute the code in the model completion.
		@:param example (str): the example
		@:param completion (str): the model completion

		@:return: the answer (the type depends on the dataset)
		'''
		if self.no_solver: # no solver, use the LM to generate the answer from the completion
			if self.dataset_name == "AQUA":
				if "answer is " not in completion:
					answer = "[invalid]"
				else:
					answer = completion.split("answer is ")[-1].strip("\n().").upper()
			elif self.dataset_name in ["GSM8K", "SVAMP", "MultiArith", "ASDiv"]:
				if "answer is " not in completion:
					answer = "[invalid]"
				else:
					answer = completion.split("answer is ")[-1].strip("\n.")
			elif self.dataset_name == "date":
				if "answer is " not in completion:
					answer = "[invalid]"
				else:
					answer = completion.split("answer is ")[-1].strip()
					answer = re.sub(pattern="[\s\.#]", repl="", string=answer)
			elif self.dataset_name == "sports":
				if "answer is " not in completion:
					answer = "[invalid]"
				else:
					answer = completion.split("answer is ")[-1].split()[0].strip(".")
					if answer == "yes":
						answer = "1"
					elif answer == "no":
						answer = "0"
					else:
						answer = "[invalid]"
			elif self.dataset_name == "saycan":
				completion = completion.strip()
				lines = completion.split("\n")
				if len(lines) == 1:
					answer = lines[0].strip()
				else:
					answer_line = [line for line in lines if line.startswith("Plan:")][0]
					answer = answer_line.split("Plan: ")[1].strip()
			elif self.dataset_name == "CLUTRR":
				answer = "[invalid]"
				lines = completion.split("\n")
				lines = [line.strip() for line in lines if line.strip() != ""]
				answer_line = lines[-1]
				# look for patterns like "A is B's xx (relation name)", "A is the xx (relation name) of B"
				patterns = ["(\[?\w+\]?) is (\[?\w+\]?)'s (\w+)",
				            "(\[?\w+\]?) is the (\w+) of (\[?\w+\]?)"]
				relation_position = [3, 2] # where the relation name is in the matched pattern
				for pattern_id, pattern in enumerate(patterns):
					matched_pattern = re.search(pattern=pattern, string=answer_line)
					if matched_pattern is not None:
						# extract the relation name
						relation_name = matched_pattern.group(relation_position[pattern_id])
						answer = relation_name
						break
					else:
						continue
				answer = answer.strip(".")
			elif self.dataset_name == "StrategyQA":
				if "answer is" not in completion:
					answer = "[invalid]"
				else:
					answer = completion.split("answer is ")[-1].split()[0].strip("\n.").lower()
					if answer == "yes":
						answer =  True
					elif answer == "no":
						answer = False
					else:
						answer = "[invalid]"
			else:
				for line in completion.split("\n"):
					if line.startswith("Answer: "):
						answer = line[8:].strip('"')
			return answer

		else: # use the solver to derive the answer by executing the completion
			if self.dataset_name in ["GSM8K", "SVAMP", "MultiArith", "ASDiv"]:
				answer = math_solver.solve_mwp(completion)
				return answer

			elif self.dataset_name == "AQUA":
				answer = str(math_solver.solve_mwp(completion)).strip()
				# AQUA requires the answer to be in the form of a choice
				# we convert the answer to a choice by querying the LM again
				with open(f"source/prompt/AQUA/{self.prompt_name}_choice_prompt.txt", "r") as fr:
					choice_prompt = fr.read()
				with open(f"source/prompt/AQUA/{self.prompt_name}_choice_template.txt", "r") as fr:
					choice_template = fr.read()
				example["answer"] = answer
				templated_example = self._apply_template(template=choice_template, example=example)
				prompt_and_example = f"{choice_prompt}\n\n{templated_example}"
				completions = self._query(prompt=prompt_and_example,
										stop=[')', '\n'],
										LM=self.LM)
				final_answer = completions[0]
				return final_answer

			elif self.dataset_name == "date":
				completion = completion.rstrip("#")
				old_stdout = sys.stdout
				redirected_output = sys.stdout = StringIO()
				exec(completion)
				sys.stdout = old_stdout
				return redirected_output.getvalue()

			elif self.dataset_name == "sports":
				completion = completion.rstrip("#")
				completion += "\nprint(answer)"
				old_stdout = sys.stdout
				redirected_output = sys.stdout = StringIO()
				exec(completion)
				sys.stdout = old_stdout
				return redirected_output.getvalue()

			elif self.dataset_name == "StrategyQA":
				answer = datalog_solver.solve(completion, self.prompt_name)
				return answer

			elif self.dataset_name == "saycan":
				goal = []
				for line in completion.split("\n"):
					if not line:
						continue
					if not line.lstrip().startswith(";"):
						goal.append(line.rstrip())
				goal = "\n".join(goal)
				pddl_plan = pddl_planner.generate_plan_for_goal(goal=goal, prompt_name=self.prompt_name)
				nl_plan = pddl_planner.convert_plan_to_nl(plan=pddl_plan, goal=goal)
				return nl_plan

			elif self.dataset_name == "CLUTRR":
				answer = CLUTRR_solver.solve(completion)
				return answer

			else:
				raise NotImplementedError(f"Solver for dataset {self.dataset_name} is not implemented.")

	def derive_answer_from_completions(self, example, completions):
		'''Derive the answer from a list of completions.
		@:param example (dict): the example
		@:param completions (List[str]): the list of completions

		@:return (tuple): answer (type depends on dataset), final_completion (str)
		'''

		# execute the completions to get the answers
		completion_lists = {}  # a dict of lists of completions; each item is {answer: [completions that result in the same answer after execution]}
		for completion in completions:
			try:
				answer = self._execute(example=example, completion=completion)  # execute the completion
			except Exception as e:
				print(f"Error executing completion: {completion}.\n Error: {e}")
				continue

			if type(answer) == str and "invalid" in answer:
				continue

			answer = self.postprocess_answer(answer)

			# check for answer equivalence
			equivalent_found = False
			for existing_answer in completion_lists.keys():
				if existing_answer == answer:  # if the answer is equivalent to an existing answer
					completion_lists[existing_answer].append(
						completion)  # add the completion to list of completions corresponding to the existing answer
					equivalent_found = True
					break
			if not equivalent_found:  # if the answer is not equivalent to any existing answer
				completion_lists[answer] = [completion]  # create a new list of completions corresponding to the answer

		# get the top-voted answer as the final answer
		if len(completion_lists) == 0:  # if no valid completion is found
			return "[invalid]", completions[0]

		completion_lists = sorted(completion_lists.items(), key=lambda x: len(x[1]),
		                          reverse=True)  # vote for the majority answer
		final_completion = completion_lists[0][1][0]
		answer = completion_lists[0][0]

		return answer, final_completion

	def postprocess_answer(self, answer):
		'''Postprocess the answer based on the dataset.
		@:param answer: the answer to be postprocessed

		@:return: the postprocessed answer
		'''
		if self.dataset_name in ["GSM8K", "SVAMP", "MultiArith", "ASDiv"]:
			answer = str(answer).strip()
			answer = answer.split("\n")[-1]  # only get the last output
			return answer

		elif self.dataset_name == "AQUA":
			answer = str(answer).strip()[0]
			return answer

		elif self.dataset_name == "date":
			answer = str(answer).strip()
			answer = answer.split("\n")[-1] # only get the last output
			answer = answer.rstrip("Y") # strip the trailing "Y"s if it exists
			return answer

		elif self.dataset_name == "sports":
			answer = str(answer).strip()
			answer = answer.split("\n")[-1] # only get the last output
			return answer

		elif self.dataset_name == "StrategyQA":
			return answer
			
		elif self.dataset_name == "saycan":
			return answer
			
		elif self.dataset_name == "CLUTRR":
			answer = str(answer).strip()
			return answer

		else:
			raise NotImplementedError(f"Postprocessing function for dataset {self.dataset_name} is not implemented.")

	@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), after=log_retry)
	def _query(self, prompt, stop, LM, n=1, logprobs=None, temperature=0.0, max_tokens=1024):
		'''Query an OpenAI model.
		@:param prompt (str): the prompt to be fed to the model
		@:param stop (list): the stop tokens
		@:param LM (str): the LM to be queried
		@:param n (int): the number of completions to be returned
		@:param logprobs (int): the number of most likely tokens whose logprobs are to be returned
		@:param temperature (float): the temperature of the model
		@:param max_tokens (int): the maximum number of tokens to be returned

		@:return (dict): the response from the model
		'''
		api_key = next(self.api_keys)
		openai.api_base = 'https://api.bltcy.ai/v1'
		openai.api_key = api_key

		if LM in ["code-davinci-001", "code-davinci-002", "text-davinci-001", "text-davinci-002", "text-davinci-003"]: # models that support "completion"
			response = openai.Completion.create(
				model=LM,
				prompt=prompt,
				temperature=temperature,
				max_tokens=max_tokens,
				n=n,
				frequency_penalty=0,
				presence_penalty=0,
				stop=stop
			)
			choices = response["choices"]
			completions = [choice["text"] for choice in choices]
		elif LM in ["gpt-3.5-turbo", "gpt-4", "qwen-turbo"]: # models that support "chat"
			response = openai.ChatCompletion.create(
				model=LM,
				messages=[
					{"role": "user", "content": prompt},
				],
				temperature=temperature,
				n=n,
				frequency_penalty=0,
				presence_penalty=0,
				stop=stop
			)
			choices = response["choices"]
			completion_objs = [choice.message for choice in choices]
			completions = [completion.content for completion in completion_objs]
		else:
			raise NotImplementedError(f"Model {LM} is not supported.")
		return completions


if __name__ == "__main__":
	'''Run a simple test.'''

	dataset_name = ["AQUA", "ASDiv", "GSM8K", "MultiArith", "SVAMP", "StrategyQA", "date", "sports", "saycan", "CLUTRR"][2]

	config_frn = f"source/configuration/config_files/{dataset_name}/qwen-turbo_COT.json"
	config = Config.from_json_file(config_frn)
	api_keys = list(API_KEYS.values())
	config.api_keys = api_keys
	config.dataset_name = dataset_name

	model = Model(config)

	example = {"question": "Randy has 9 oatmeal cookies, 4 chocolate chip cookies, and 5 sugar cookies. He ate 3 cookies for an early day snack, one of each flavor. He ate 2 oatmeal cookies for lunch. He gives 2 sugar cookies to his friends. Then, he bakes 4 of each flavor for dinner. How many cookies does he have now?",
	           "answer": "Randy originally has 9+4+5 = <<9+4+5=18>>18 cookies in total.\nHe has 18-3 = <<18-3=15>>15 cookies left.\nHe has 15-2 = <<15-2=13>>13 cookies left.\nHe has 13-2 = <<13-2=11>>11 cookies left.\nRandy bakes 4*3 = <<4*3=12>>12 cookies.\nRandy has 11+12 = <<11+12=23>>23 cookies.\n#### 23", "id": 1301}

	output = model.predict(example)
	answer = output["answer"]
	completion = output["completion"]
	print("Answer:", [answer])
	print("Completion:", [completion])
