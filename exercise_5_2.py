from pyomo.environ import *
from pyomo.opt import SolverFactory
import csv
from help_functions import *


nodes = 5
spread = '2.5'
K = 5  # Puedes modificar este valor según necesites


def leer_matriz_csv(nombre_archivo=f'cost_matrix_{nodes}_nodes_{spread}_spread.csv', folder='Lab2_github/files_support/'):
    matriz = []
    with open(folder + nombre_archivo, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            matriz.append([float(x) for x in fila])
    return matriz[1:]  # se omite la primera fila si es header


cost_matrix = leer_matriz_csv()
N = len(cost_matrix)


model = ConcreteModel()
model.N = RangeSet(N)
model.C = Param(model.N, model.N, initialize=lambda model,
                i, j: cost_matrix[i-1][j-1])
model.x = Var(model.N, model.N, within=Binary)

# Variable auxiliar para eliminación de subtours (formulación MTZ)


def u_bounds(model, i):
    if i == 1:
        return (0, 0)   # Fijamos el depósito en 0
    else:
        return (1, N - K)  # Los demás nodos deben tener valores entre 1 y N-K


model.u = Var(model.N, bounds=u_bounds, within=NonNegativeReals)

# Función objetivo: Minimizar el costo total
model.obj = Objective(expr=sum(model.C[i, j] * model.x[i, j]
                      for i in model.N for j in model.N if i != j), sense=minimize)

# Restricciones de visita
model.visit_once = ConstraintList()
# Para el depósito (nodo 1): deben salir y llegar K rutas
model.visit_once.add(sum(model.x[1, j] for j in model.N if j != 1) == K)
model.visit_once.add(sum(model.x[j, 1] for j in model.N if j != 1) == K)
# Para cada nodo que no es el depósito: deben entrar y salir exactamente una vez
for i in model.N:
    if i != 1:
        model.visit_once.add(sum(model.x[i, j]
                             for j in model.N if j != i) == 1)
        model.visit_once.add(sum(model.x[j, i]
                             for j in model.N if j != i) == 1)

# Eliminación de subtours (restricción MTZ extendida para mTSP)
model.subtour = ConstraintList()
M = N - K  # coeficiente que se usa en la restricción MTZ
for i in model.N:
    for j in model.N:
        if i != j and i != 1 and j != 1:
            model.subtour.add(model.u[i] - model.u[j] +
                              M * model.x[i, j] <= M - 1)

# Resolver el modelo
solver = SolverFactory('glpk')
solver.solve(model, tee=True)

# Extracción de rutas para cada equipo
# Primero, obtenemos todos los arcos que forman parte de la solución
arcos = [(i, j) for i in model.N for j in model.N if i !=
         j and value(model.x[i, j]) == 1]

# Identificamos los ciclos que parten del depósito (nodo 1)
rutas = []
# Para cada arco saliente del depósito, se sigue la cadena hasta volver al depósito
for j in [j for j in model.N if j != 1 and value(model.x[1, j]) == 1]:
    ruta = [1, j]
    siguiente = j
    while True:
        # Buscamos el arco que inicia en el nodo 'siguiente' y que NO regresa inmediatamente al depósito
        siguiente_arco = [arc for arc in arcos if arc[0]
                          == siguiente and arc[1] != 1]
        if not siguiente_arco:
            # Si no hay más arcos, se asume que se retorna al depósito
            ruta.append(1)
            break
        siguiente = siguiente_arco[0][1]
        ruta.append(siguiente)
        if siguiente == 1:
            break
    rutas.append(ruta)

# Mostrar las rutas identificando a cada equipo
print("Rutas óptimas por equipo:")
for idx, ruta in enumerate(rutas, start=1):
    print(f"Equipo {idx}: {ruta}")

# Visualización
dibujar_grafo_sin_ruta(cost_matrix, filename=f'grafo_{nodes}.png')
dibujar_grafo_equipos(cost_matrix, rutas,
                      filename=f'grafo_solucionado_{nodes}_teams_{K}.png')
