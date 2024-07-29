'''Reformat a goal with inner "exists" conditions to a goal with a single outermost "exists" condition, as only the latter format is accepted by the parse we use.'''

import re
from pprint import pprint

def reformat_goal(goal_str):
    '''Reformat the goal with inner "exists" conditions to a goal with a single outermost "exists" condition, as only the latter format is accepted by the parse we use.

    :param goal_str (str): The goal string to reformat
    :return (str): The reformatted goal string

    Example:
    Input:
    "(:goal
        (and
            (exists (?f - fruit)
                (at ?f user)
            )
            (exists (?s - soda)
                (at ?s user)
            )
        )
    )"

    Output:
    "(:goal
        (exists (?f - fruit ?s - soda)
          (and
            (at ?f user)
            (at ?s user)
          )
        )
    )"
    '''
    if "exists" not in goal_str and "initial" not in goal_str: # no need to reformat
        return goal_str
    else:
        # convert the goal string to a dictionary
        goal_dict = goal_str2dict(goal_str)
        # reformat goal_dict
        new_goal_dict = reformat_goal_dict(goal_dict)
        # convert the reformatted goal dict back to a string
        new_goal_str = goal_dict2str(new_goal_dict)
        return new_goal_str

def goal_str2dict(goal_str):
    '''Parse a PDDL goal string into a dictionary representation.
    @:param goal_str: The PDDL goal string to parse.

    @:return: A dictionary representation of the goal.

    Example:
    Input:
    "(:goal
        (and
            (exists (?f - fruit)
                (at ?f user)
            )
            (exists (?s - soda)
                (at ?s user)
            )
        )
    )"
    Output:
    {'goal': [{'and': [{'exists': [{'expr': [{'at': ['?f', 'user']}],
                                'params': [{'name': '?f', 'type': 'fruit'}]}]},
                   {'exists': [{'expr': [{'at': ['?s', 'user']}],
                                'params': [{'name': '?s',
                                            'type': 'soda'}]}]}]}]}
    '''
    def extract_params(param_str):
        params = []
        for match in re.finditer(r'\?([a-zA-Z0-9\-]+)\s*-\s*([a-zA-Z0-9\-]+)', param_str):
            name, type = match.groups()
            params.append({'name': f'?{name}', 'type': type})
        return params

    def parse_expression(exp_str):
        op_match = re.match(r'^\(([\-\w=]+)\s+(.*)\)$', exp_str.strip())
        if not op_match:
            return exp_str.strip()
        op, content = op_match.groups()
        if op in ('exists', 'forall'):
            params_end = content.index(')')
            params_str = content[:params_end + 1]
            params = extract_params(params_str)
            rest = content[params_end + 1:].strip()
            sub_expr = parse_expression(rest)
            return {op: [{'params': params, 'expr': [sub_expr]}]}
        else:
            expressions = []
            stack = 0
            start = 0
            for i, char in enumerate(content):
                if char == '(':
                    stack += 1
                elif char == ')':
                    stack -= 1
                elif char == ' ' and stack == 0:
                    sub_str = content[start:i].strip()
                    if sub_str:
                        expressions.append(parse_expression(sub_str))
                    start = i + 1
            if start < len(content):
                expressions.append(parse_expression(content[start:]))

            return {op: expressions}

    goal_str = goal_str.replace("\t", " ").replace("\n", " ")
    goal_str = re.sub(r'\s+', ' ', goal_str).strip()
    goal_match = re.match(r'\(:goal\s+(.*)\)', goal_str.strip(), re.DOTALL)
    if goal_match:
        goal_content = goal_match.group(1)
        goal_dict = {'goal': [parse_expression(goal_content)]}

    return goal_dict

def reformat_goal_dict(goal):
    '''Reformat the goal by combining all "exists" into a single one with all the variables, and connecting all the other conditions with "and".
    @:param goal (dict): The goal dictionary to reformat.

    @:return (dict): The reformatted goal dictionary.

    Example:
    Input:
    {'goal': [{'and': [{'exists': [{'expr': [{'at': ['?f', 'user']}],
                                'params': [{'name': '?f', 'type': 'fruit'}]}]},
                   {'exists': [{'expr': [{'at': ['?s', 'user']}],
                                'params': [{'name': '?s',
                                            'type': 'soda'}]}]}]}]}
    Output:
    {'goal': [{'exists': [{'expr': [{'and': [{'at': ['?f', 'user']},
                                         {'at': ['?s', 'user']}]}],
                       'params': [{'name': '?f', 'type': 'fruit'},
                                  {'name': '?s', 'type': 'soda'}]}]}]}
    '''
    # Find all the exists conditions
    exists_conditions = []
    all_conditions = []
    if 'and' not in goal['goal'][0]:
        return goal
    for expr in goal['goal'][0]['and']:
        if isinstance(expr, dict) and expr.get('exists'):
            exists_conditions += expr['exists']
        all_conditions.append(expr)

    if len(exists_conditions) > 0:
        # Merge the exists conditions and create a new and condition
        merged_exist_params = []
        for exists_condition in exists_conditions:
            for var in exists_condition['params']:
                merged_exist_params.append(var)

        merged_and = []
        for condition in all_conditions:
            if isinstance(condition, dict) and condition.get('exists'):
                for subcondition in condition['exists']:
                    merged_and += subcondition['expr']
            else:
                merged_and.append(condition)

        new_goal = {'exists': [{'params': merged_exist_params, 'expr': [{'and': merged_and}]}]}

    else:
        new_goal = goal['goal'][0]

    # Return the reformatted goal
    return {'goal': [new_goal]}

def goal_dict2str(goal_dict, indent=2):
    '''Convert a goal dictionary to a string.
    @:param goal_dict (dict): The goal dictionary to convert to a string.
    @:param indent (int): The number of spaces to indent the goal string.

    @:return (str): The goal string.

    Example:
    Input:
    {'goal': [{'exists': [{'expr': [{'and': [{'at': ['?f', 'user']},
                                         {'at': ['?s', 'user']}]}],
                       'params': [{'name': '?f', 'type': 'fruit'},
                                  {'name': '?s', 'type': 'soda'}]}]}]}
    Output:
    (:goal
        (exists (?f - fruit ?s - soda)
          (and
            (at ?f user)
            (at ?s user)
          )
        )
    )
    '''
    def format_params(params):
        formatted_params = " ".join([f"{p['name']} - {p['type']}" for p in params])
        return f"({formatted_params})"

    if isinstance(goal_dict, list):
        return "\n".join([goal_dict2str(g, indent) for g in goal_dict])

    if isinstance(goal_dict, dict):
        for key, value in goal_dict.items():
            if key == "goal":
                return f"(:{key}\n{goal_dict2str(value, indent + 1)}\n)"
            elif key in ["and", "or"]:
                return f"{indent * ' '}({key}\n{goal_dict2str(value, indent + 1)}\n{indent * ' '})"
            elif key == "not":
                inner_result = goal_dict2str(value, indent + 1)
                if inner_result == "":
                    return ""
                else:
                    return f"{indent * ' '}({key} {inner_result})"
            elif key == "exists":
                formatted_params = format_params(value[0]["params"])
                formatted_expr = goal_dict2str(value[0]["expr"], indent + 1)
                return f"{indent * ' '}({key} {formatted_params}\n{formatted_expr}\n{indent * ' '})"
            elif key == "at":
                if value[-1] == "initial":
                    return ""
                else:
                    value_str = ' '.join([v for v in value])
                    return f"{indent * ' '}({key} {value_str})"
            else:
                value_str = ' '.join([v for v in value])
                return f"{indent * ' '}({key} {value_str})"

    raise ValueError("Invalid input for PDDL goal conversion")


if __name__ == "__main__":
    # run a simple test
    goal_str = '''
(:goal
	(and
		(at rice-chips counter)
		(not (at rice-chips table))
		(not (at rice-chips trash))
		(not (at rice-chips bowl))
		(not (at rice-chips initial))
	)
)
	'''

    new_goal_str = reformat_goal(goal_str)
    print(new_goal_str)