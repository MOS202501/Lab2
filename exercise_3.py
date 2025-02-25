from pyomo.environ import *
import numpy as np
from help_functions import *

model = ConcreteModel()

N = 10

model.x = Var(range(N), range(N), within=Binary)

model.coverage = ConstraintList()

for i in range(N):
    for j in range(N):
        model.coverage.add(
            sum(model.x[i, k] for k in range(N)) +
            sum(model.x[k, j] for k in range(N)) +
            # Cobertura por diagonal principal (↘)
            sum(model.x[i + k, j + k] for k in range(-min(i, j), N - max(i, j)) if 0 <= i + k < N and 0 <= j + k < N) +
            # Cobertura por diagonal secundaria (↙)
            sum(model.x[i - k, j + k] for k in range(-min(N - 1 - i, j), min(i + 1,
                N - j)) if 0 <= i - k < N and 0 <= j + k < N)
            >= 1
        )

model.obj = Objective(
    expr=sum(model.x[i, j] for i in range(N) for j in range(N)), sense=minimize)

solver = SolverFactory('glpk').solve(model)

plot_queens_solution(model, N)
