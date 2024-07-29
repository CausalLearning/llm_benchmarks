from sympy import Symbol, Eq, simplify, solve
import math
from math import *

def lcm(a,b): # for backwards compatibility with Python<3.9
	return (a * b) // math.gcd(a,b)

def solve_it(equation, variable):
	solution = solve(equation, variable, dict=True)
	if not solution:
		if isinstance(variable, list):
			solution = {v: None for v in variable}
		else:
			solution = {variable: None}
		return solution
	else:
		solution = solution[0]
		return solution

# 1. How many books does Sofie have? (independent, support: ["Sofie has 25 more books than Anne"])
books_sofie = Symbol("books_sofie")
# 2. How many books does Anne have? (depends on 1, support: ["Sofie has 25 more books than Anne"])
books_anne = books_sofie - 25
# 3. How many books does Fawn have? (depends on 2, support: ["Anne has 12 fewer books than Fawn does"])
books_fawn = books_anne + 12
# 4. How many books do Sofie, Anne, and Fawn have together? (depends on 1, 2, and 3, support: ["Together, Sofie, Anne, and Fawn have 85 books"])
books_total_eq = Eq(books_sofie + books_anne + books_fawn, 85)
# 5. How many books does Sophie have based on this equation? (depends on 1, 2, 3, and 4, support: [])
books_sofie_val = solve_it(books_total_eq, books_sofie)[books_sofie]
# 6. How many books does Fawn have? (depends on 5, support: [])
books_fawn_val = books_fawn.subs(books_sofie, books_sofie_val)
# 7. Final Answer: How many books does Fawn have? (depends on 6, support: [])
answer = books_fawn_val
print(answer)