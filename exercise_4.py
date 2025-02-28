from pyomo.environ import *
from help_functions import *

# Definir las coordenadas de los nodos
datos_nodos = {
    1: (20, 6),
    2: (22, 1),
    3: (9, 2),
    4: (3, 25),
    5: (21, 10),
    6: (29, 2),
    7: (14, 12)
}

# Número de nodos
num_nodos = len(datos_nodos)

# Inicializar la matriz de costos con infinito (sin conexión)
matriz_costos = np.full((num_nodos, num_nodos), np.inf)

# Calcular la distancia euclidiana y determinar enlaces
for i in range(1, num_nodos + 1):
    for j in range(1, num_nodos + 1):
        if i != j:
            xi, yi = datos_nodos[i]
            xj, yj = datos_nodos[j]
            distancia = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if distancia <= 20:
                matriz_costos[i-1, j-1] = round(distancia,1)  

# Definir el modelo en Pyomo
modelo = ConcreteModel()

# Conjuntos
modelo.N = RangeSet(num_nodos)
modelo.A = [(i+1, j+1) for i in range(num_nodos) for j in range(num_nodos) if matriz_costos[i, j] < np.inf]

# Parámetros
modelo.c = Param(modelo.A, initialize={(i, j): matriz_costos[i-1, j-1] for (i, j) in modelo.A})

# Variables de decisión
modelo.x = Var(modelo.A, within=Binary)

# Función objetivo: minimizar el costo total del recorrido
modelo.objetivo = Objective(expr=sum(modelo.c[i, j] * modelo.x[i, j] for (i, j) in modelo.A), sense=minimize)

# Restricciones
modelo.conservacion_flujo = ConstraintList()
for n in modelo.N:
    if n == 4:  # Nodo de origen
        modelo.conservacion_flujo.add(sum(modelo.x[i, j] for (i, j) in modelo.A if i == n) - sum(modelo.x[i, j] for (i, j) in modelo.A if j == n) == 1)
    elif n == 6:  # Nodo de destino
        modelo.conservacion_flujo.add(sum(modelo.x[i, j] for (i, j) in modelo.A if i == n) - sum(modelo.x[i, j] for (i, j) in modelo.A if j == n) == -1)
    else:  # Nodos intermedios
        modelo.conservacion_flujo.add(sum(modelo.x[i, j] for (i, j) in modelo.A if i == n) - sum(modelo.x[i, j] for (i, j) in modelo.A if j == n) == 0)

# Resolver el modelo
solver = SolverFactory('glpk')
resultado = solver.solve(modelo)
ruta_optima = []
# Mostrar la solución
print("Ruta de mínimo costo de 4 a 6:")
for (i, j) in modelo.A:
    if modelo.x[i, j].value == 1:
        ruta_optima.append((i,j))
        print(f"Nodo {i} -> Nodo {j} (Costo: {modelo.c[i, j]})")

graficar_red(datos_nodos, modelo, matriz_costos, ruta_optima)
