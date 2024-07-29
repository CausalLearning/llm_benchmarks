import re
from pddl.parser.domain import DomainParser
import os
cwd = os.getcwd()
if cwd.endswith("source/model/solver/saycan"):
	os.chdir("../../../..")  # change the working directory to the root directory

def normalize_plan_to_list(plan):
	"""Normalize a plan to a list of actions."""
	plan = plan.strip()
	steps = re.split(r"\n|, ", plan)
	new_steps = []
	for step in steps:
		if not step:
			continue
		new_step = step.strip(".")
		new_step = new_step.split(".")[-1]
		new_steps.append(new_step)
	return new_steps

def embed_goal_in_action(goal: str):
	"""Embed a goal in an action."""
	action_str = f'''(:action test
	      :parameters (?itm - item)
	      :precondition {goal}
	      :effect ()
	   )'''
	return action_str

def add_action_to_domain(action_str: str, domain_str: str):
	"""Add an action to a domain."""
	# validate the domain str
	new_domain_str = domain_str.rstrip()
	assert new_domain_str.endswith(")")
	new_domain_str = new_domain_str[:-1]

	new_domain_str += f"\n{action_str}\n)"
	return new_domain_str

def strip_goal(goal):
	"""Strip the goal of its prefix and suffix."""
	new_goal = goal.strip()
	try:
		assert new_goal.startswith("(:goal")
		assert new_goal.endswith(")")
	except AssertionError:
		print("Invalid goal: ", goal)
		return None
	new_goal = new_goal[6:-1]
	return new_goal

def check_goal_validity(goal):
	"""Check if a goal is valid."""
	# create a domain
	domain_frn = "source/model/solver/saycan/pddl_files/domain/domain_for_eval.pddl"
	with open(domain_frn, "r") as f:
		domain_str = f.read()
	domain_parser = DomainParser()
	assert domain_parser(domain_str)

	# embed the goals in actions
	goal = strip_goal(goal)
	if goal == None:
		return False
	action = embed_goal_in_action(goal)

	# add the actions to the domain
	new_domain_str = add_action_to_domain(action, domain_str)
	try:
		new_domain = domain_parser(new_domain_str)
		condition = list(new_domain.actions)[0].precondition
		return True
	except Exception as e:
		print(e)
		return False


def check_goals_equivalence(goal1, goal2):
	"""Check if two goals are equivalent.
	1. Should be insensitive to argument names.
	2. Should be insensitive to the order of arguments.
	3. Should be able to flatten the statements.
	4. Should be insensitive to the order of statements in conjunction/disjunction.
	5. Should simplify quantifications.
	"""
	# create a domain
	domain_frn = "source/model/solver/saycan/pddl_files/domain/domain_for_eval.pddl"
	with open(domain_frn, "r") as f:
		domain_str = f.read()
	domain_parser = DomainParser()
	assert domain_parser(domain_str)

	# embed the goals in actions
	goal1 = strip_goal(goal1)
	goal2 = strip_goal(goal2)
	action1 = embed_goal_in_action(goal1)
	action2 = embed_goal_in_action(goal2)

	# add the actions to the domain
	domain1_str = add_action_to_domain(action1, domain_str)
	domain2_str = add_action_to_domain(action2, domain_str)

	try:
		domain1 = domain_parser(domain1_str)
	except Exception as e:
		print(f"Invalid goal: {goal1}")
		print(e)
		return False
	try:
		domain2 = domain_parser(domain2_str)
	except Exception as e:
		print(f"Invalid goal: {goal2}")
		print(e)
		return False

	condition1 = list(domain1.actions)[0].precondition
	condition2 = list(domain2.actions)[0].precondition
	return condition1 == condition2


if __name__ == "__main__":
	'''Run a test.'''

	goal1 = '''(:goal
					(exists (?s - snack ?d1 - drink ?i - item ?s1 - soda)
						(and
							(is-salty ?s)
							(at ?s user)
							(is-salty ?d)
							(at ?d user)
						)
					)
				)'''
	goal2 = '''(:goal
					(exists (?d1 - drink ?s - snack ?s1 - soda ?i6 - item)
						(and
							(is-salty ?d)
							(at ?d user)
							(is-salty ?s)
							(at ?s user)
						)
					)
				)'''
	is_equivalent = check_goals_equivalence(goal1, goal2)
	print(is_equivalent)

	is_valid = check_goal_validity(goal1)
	print(is_valid)
