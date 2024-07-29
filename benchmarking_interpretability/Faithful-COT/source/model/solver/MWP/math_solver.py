import sys
from io import StringIO
import os
cwd = os.getcwd()
if cwd.endswith("source/model/solver/MWP"):
	os.chdir("../../../..")  # change the working directory to the root directory

def solve_mwp(completion):
	''' Execute the completion to solve the question (for math word problems).
	:param completion (str): the model completion

	:return (str): the final relation
	'''

	# get the relation from every step

	prefix_frn = "source/model/solver/MWP/prefix.txt"
	with open(prefix_frn, "r") as fr:
		prefix = fr.read()

	completion = completion.rstrip("#")

	code = f"{prefix}\n{completion}"

	try:
		locs = {}
		exec(code, locs, locs)
		answer = locs["answer"]
	except Exception as e:
		answer = "[invalid]"

	return answer


if __name__ == "__main__":
	completion = '''# 1. What is the unit digit? (independent, support: [])
unit_digit = Symbol("unit_digit")
# 2. What is the tens digit? (depends on 1, support: ["the tens digit is three times the unit digit"])
tens_digit = 3 * unit_digit
# 3. What is the number? (depends on 1 and 2, support: [])
number = tens_digit * 10 + unit_digit
# 4. What is the number decreased by 54? (depends on 3, support: ["When the number is decreased by 54"])
number_decreased_54 = number - 54
# 5. What is the number decreased by 54, with the digits reversed? (depends on 4, support: ["the digits are reversed"])
number_decreased_54_reversed = unit_digit * 10 + tens_digit
# 6. What is the number decreased by 54, with the digits reversed, given the number decreased by 54? (depends on 5, support: [])
number_decreased_54_reversed_eq = Eq(number_decreased_54_reversed, number_decreased_54)
# 7. What is the unit digit given the number decreased by 54, with the digits reversed? (depends on 6, support: [])
unit_digit_val = solve_it(number_decreased_54_reversed_eq, unit_digit)[unit_digit]
# 8. What is the tens digit given the unit digit? (depends on 1 and 7, support: [])
tens_digit_val = tens_digit.subs(unit_digit, unit_digit_val)
# 9. What is the number given the tens digit and the unit digit? (depends on 1 and 8, support: [])
number_val = number.subs(tens_digit, tens_digit_val).subs(unit_digit, unit_digit_val)
# 10. Final Answer: Find the number. (depends on 9, support: [])
answer = number_val
'''
	answer = solve_mwp(completion)
	print(answer)
