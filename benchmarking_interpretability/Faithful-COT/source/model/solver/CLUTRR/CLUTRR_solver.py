import pickle

def solve(completion):
    ''' Execute the completion to solve the question (for CLUTRR).
    :param completion (str): the model completion

    :return (str): the final relation
    '''

    # get the relation from every step
    relations = []
    for line in completion.split('\n')[1:]:
        if ' = ' in line and not '@' in line: # it's a single relation line, not a comment line or the final relation deduction line
            try:
                relation = line.split(' = ')[1] # get the relation
            except IndexError as e:
                print(f"Error: {e}, line: {line}")
                continue
            relations.append(relation)

    # load the transitive rules
    with open("source/model/solver/CLUTRR/trans_rules.pkl", 'rb') as f:
        trans_dict = pickle.load(f)
      
     # apply the transitive rules to get the final relation
    if not relations:
        return "[invalid]"

    final_relation = ""
    for relation in reversed(relations):
        if not final_relation:
            # first relation
            final_relation = relation
        else:
            # apply transitive rules
            try:
                final_relation = trans_dict[(final_relation, relation)]
            except KeyError:
                return "[invalid]"
    return final_relation

if __name__ == "__main__":
    # run a simple test
    import os
    os.chdir("../../../..")
    blob = '''[James]? (independent, support: \"[James] and [Dorothy] werent able to have children naturally, so they adopted [Aida] from sweden.\")\nrelation(Aida, James) = adopted\n# 2. How is [James] related to [Dorothy]? (independent, support: \"[James] and [Dorothy] werent able to have children naturally, so they adopted [Aida] from sweden.\")\nrelation(James, Dorothy) = husband\n# 3. '''
    print(solve(blob))
