from pprint import pprint
from pyomo.environ import *
from pyomo.opt import SolverFactory
import csv
from help_functions import *

nodes = 50
spread = '1.0'


def leer_matriz_csv(nombre_archivo=f'cost_matrix_{nodes}_nodes_{spread}_spread.csv', folder='Lab2_github/files_support/'):
    matriz = []
    with open(folder + nombre_archivo, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            matriz.append([float(x) for x in fila])
    return matriz[1:]


cost_matrix = leer_matriz_csv()
N = len(cost_matrix)

model = ConcreteModel()

model.N = RangeSet(N)
model.C = Param(model.N, model.N, initialize=lambda model,
                i, j: cost_matrix[i-1][j-1])

model.x = Var(model.N, model.N, within=Binary)
model.u = Var(model.N, within=NonNegativeReals)

# Función objetivo: Minimizar el costo total
model.obj = Objective(expr=sum(model.C[i, j] * model.x[i, j]
                      for i in model.N for j in model.N), sense=minimize)

model.visit_once = ConstraintList()
for i in model.N:
    model.visit_once.add(sum(model.x[i, j] for j in model.N if i != j) == 1)
    model.visit_once.add(sum(model.x[j, i] for j in model.N if i != j) == 1)

model.subtour = ConstraintList()
for i in model.N:
    for j in model.N:
        if i != j and i != 1 and j != 1:
            model.subtour.add(model.u[i] - model.u[j] +
                              N * model.x[i, j] <= N - 1)

solver = SolverFactory('glpk')
solver.solve(model, tee=True)

# Obtener la ruta óptima
route = []
current = 1
while len(route) < N:
    for j in model.N:
        if current != j and model.x[current, j].value == 1:
            route.append((current, j))
            current = j
            break

print("Ruta óptima:")
pprint(route)

dibujar_grafo_sin_ruta(cost_matrix, filename=f'grafo_{nodes}.png')
dibujar_grafo(cost_matrix, route,
              filename=f'grafo_solucionado_{nodes}_teams_1.png')
