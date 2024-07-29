from subprocess import PIPE, Popen

def solve(completion, prompt_name):
	''' Execute the completion to solve the question (for multi-hop QA problems).
	:param completion (str): the model completion
	:param prompt_name (str): the prompt name (to be used in the temporary output file name)

	:return (str): the final answer
	'''
	datalog_fwn = f"source/model/solver/StrategyQA/datalog_files/test_{prompt_name}.dl"
	with open(datalog_fwn, 'w') as fw:
		fw.write(completion)
	p = Popen(f'~/.local/bin/souffle -D - {datalog_fwn}', shell=True, stdout=PIPE, stderr=PIPE)
	stdout, stderr = p.communicate()
	stdout = stdout.decode('utf-8')
	if stdout == "":
		answer = False
	else:
		if "===============\n()\n===============\n" in stdout:
			answer = True
		else:
			answer = False

	return answer

