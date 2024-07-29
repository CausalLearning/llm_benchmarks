'''Evaluate the answer accuracy of a model output file.'''

import os
if os.getcwd().endswith("evaluate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data, extract_gold_answer, extract_pred_answer
from configuration.configuration import Config
import argparse
import jsonlines
import regex as re

def is_correct(dataset_name, gold_answers, pred_answer):
	'''Check if a predicted answer is correct.
	:param dataset_name (str): The name of the dataset.
	:param gold_answers: The gold answer(s).
	:param pred_answer: The predicted answer.

	:return: Whether the prediction is correct (True) or not (False).
	'''

	# saycan has multiple correct plans, so we need to check if the predicted plan is in the list of correct plans
	if dataset_name == "saycan":
		assert type(gold_answers) == list
		assert type(pred_answer) == str
		if pred_answer in ["[error]", "[invalid]"]:
			return False
		else:
			pred_answer = pred_answer.replace("\\n", "\n")
			pred_plan_list = []
			step_count = 0
			steps = re.split(r", |\n", pred_answer.strip())
			for step in steps:
				step_cols = step.split(". ")
				if len(step_cols) != 2:
					return "[invalid]"
				step_action = step_cols[1]
				if "find(initial)" in step_action:
					continue
				step_count += 1
				new_step = f"{step_count}. {step_action}"
				pred_plan_list.append(new_step)
			for gold_answer in gold_answers:
				gold_plan_list = gold_answer.strip().split("\n")
				if pred_plan_list == gold_plan_list:
					return True
		return False

	else:	# all other datasets have a single correct answer
		gold_answer = gold_answers
		return pred_answer == gold_answer

def evaluate_acc(dataset, predictions, dataset_name, non_empty_only=False, valid_only=False, debug=False):
	correct_count, total_count = 0, 0
	for example, prediction in zip(dataset, predictions):
		gold_id = int(example["id"])
		if prediction == {}:
			continue
		pred_id = int(prediction["id"])

		try:
			assert gold_id == pred_id
		except:
			raise AssertionError(f"Gold id {gold_id} doesn't match pred id {pred_id}.")

		try:
			gold_answer = extract_gold_answer(dataset_name, example["answer"])
		except SyntaxError as e:
			print("Error: ", e)
			print(gold_id)
			exit(-1)
		pred_answer = extract_pred_answer(dataset_name, prediction["answer"])

		if non_empty_only and pred_answer == "":
			continue

		if valid_only:
			if type(pred_answer)==str and ("invalid" in pred_answer or "error" in pred_answer):
				continue

		total_count += 1

		try:
			correct = is_correct(dataset_name, gold_answer, pred_answer)
		except Exception as e:
			print("Error: ", e)
			print("Example: ", gold_id)
			print("Question: ", example["question"])
			print("Gold answer: ", gold_answer, type(gold_answer))
			print("Pred answer: ", pred_answer, type(pred_answer))
			print("Completion: ", prediction["completion"])
			print("\n")
			exit(-1)

		if correct:
			correct_count += 1
		# else:
		# 	if pred_answer not in ["[error]", "[invalid]"]:
		# 		print("Example: ", gold_id)
		# 		print("Question: ", example["question"])
		# 		print("Gold answer: ", gold_answer)
		# 		print("Pred answer: ", ['\"'+f"{pred_answer}"+'\"'])
		# 		print("Completion:\n ", prediction["completion"])
		# 		print("\n")

		if debug and total_count >= 10:
			break

	acc = round(correct_count / total_count * 100, 1)
	return acc

if __name__ == "__main__":
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", choices=["GSM8K", "ASDiv", "MultiArith", "SVAMP", "AQUA", "date", "StrategyQA", "sports", "saycan", "CLUTRR"])
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--model_name", help="The name of the model (should have a corresponding config file under `configuration/config_files/dataset_name` called `{model_name}.json`.)")
	Parser.add_argument("--non_empty_only", help="If true, only evaluate on non-empty answers.", action="store_true")
	Parser.add_argument("--valid_only", help="If true, only evaluate on valid answers.", action="store_true")
	Parser.add_argument("--debug", help="If true, only run on the first 10 examples.", action="store_true")

	args = Parser.parse_args()
	model_name = args.model_name
	dataset_name = args.dataset_name
	split = args.split
	debug = args.debug
	non_empty_only = args.non_empty_only
	valid_only = args.valid_only

	config_frn = f"source/configuration/config_files/{dataset_name}/{model_name}.json"
	config = Config.from_json_file(config_frn)

	# load the dataset
	dataset_frn = f"data/{dataset_name}/{split}.jsonl"
	dataset = load_data(dataset_frn)

	output_dir = f"output_dir/{dataset_name}/{split}/{model_name}"
	pred_frn = f"{output_dir}/predictions.jsonl"

	with open(pred_frn) as fr:
		reader = jsonlines.Reader(fr)
		predictions = [line for line in reader]

	acc = evaluate_acc(dataset=dataset,
	                   predictions=predictions,
	                   dataset_name=dataset_name,
	                   non_empty_only=non_empty_only,
	                   valid_only=valid_only,
	                   debug=debug)

	print(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}")
	if valid_only:
		print("--valid_only")
	if non_empty_only:
		print("--non_empty_only")
	if debug:
		print("--debug")
	print(f"Answer accuracy: {acc}")



