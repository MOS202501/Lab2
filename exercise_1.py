from pyomo.environ import *
from help_functions import *

aviones = {
    1: {"peso": 30000, "volumen": 25},
    2: {"peso": 40000, "volumen": 30},
    3: {"peso": 50000, "volumen": 35}
}


recursos = {
    1: {"valor": 50, "peso": 15000, 'volumen': 8},
    2: {"valor": 100, "peso": 5000, 'volumen': 2},
    3: {"valor": 120, "peso": 20000, 'volumen': 10},
    4: {"valor": 60, "peso": 18000, 'volumen': 12},
    5: {"valor": 40, "peso": 10000, 'volumen': 6},
}

recursos_disponibles = RangeSet(1, len(recursos))
aviones_disponibles = RangeSet(1, len(aviones))

Model = ConcreteModel()

Model.x = Var(recursos_disponibles, aviones_disponibles,
              domain=NonNegativeReals)

Model.y_equipos_medicos = Var(aviones_disponibles, domain=NonNegativeIntegers)

# Luego modifica la restricción
Model.enteros = ConstraintList()
for a in aviones_disponibles:
    Model.enteros.add(
        Model.x[3, a] == 300 * Model.y_equipos_medicos[a]
    )


def maximizeFunction(model):
    return sum((Model.x[i, j] / recursos[i]['peso']) * recursos[i]['valor'] for i in recursos_disponibles for j in aviones_disponibles)


Model.obj = Objective(rule=maximizeFunction, sense=maximize)

Model.peso_constraint = ConstraintList()

for j in aviones_disponibles:
    Model.peso_constraint.add(
        sum(Model.x[i, j] for i in recursos_disponibles) <= aviones[j]['peso'])

Model.volumen_constraint = ConstraintList()

for j in aviones_disponibles:
    Model.volumen_constraint.add(
        sum((Model.x[i, j] / recursos[i]['peso']) * recursos[i]['volumen'] for i in recursos_disponibles) <= aviones[j]['volumen'])

Model.max_cuantity_constraint = ConstraintList()

for i in recursos_disponibles:
    Model.max_cuantity_constraint.add(
        sum(Model.x[i, j] for j in aviones_disponibles) <= recursos[i]['peso'])

Model.restriction_one = Constraint(
    expr=Model.x[2, 1] == 0)

Model.y_medicos = Var(aviones_disponibles, domain=Binary)
Model.y_agua = Var(aviones_disponibles, domain=Binary)

Model.recurso_medico_activado = ConstraintList()
Model.recurso_agua_activado = ConstraintList()

Model.compatibilidad = ConstraintList()
for j in aviones_disponibles:
    Model.compatibilidad.add(Model.y_medicos[j] + Model.y_agua[j] <= 1)

M = (sum(avion["peso"]
     for avion in aviones.values())**2)*100  # Un valor grande
for j in aviones_disponibles:
    Model.recurso_medico_activado.add(
        Model.x[3, j]
        <= M * Model.y_medicos[j]
    )
    Model.recurso_agua_activado.add(
        Model.x[4, j] <= M * Model.y_agua[j]
    )


Solver = SolverFactory('glpk')

Results = Solver.solve(Model)

print(Model.display())
# Print the matrix representation of X[i, j]
# Print the matrix in a structured table format
print("\nX Matrix:")

# Print header row
header = ["i \\ j"] + [f"{j}" for j in aviones_disponibles]
print("{:<8}{}".format(header[0], "  ".join(
    f"{col:>10}" for col in header[1:])))

# Print matrix rows
for i in recursos_disponibles:
    row_values = [Model.x[i, j].value for j in aviones_disponibles]
    print("{:<8}{}".format(i, "  ".join(f"{val:>10.1f}" for val in row_values)))

plot_pretty_heatmap(Model, recursos_disponibles, aviones_disponibles)
plot_allocation(Model, recursos_disponibles, aviones_disponibles)
