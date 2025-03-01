from pyomo.environ import *
from help_functions import *

# Definimos las coordenadas de los nodos
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

# Inicializamos la matriz de costos con infinito (sin conexión)
matriz_costos = np.full((num_nodos, num_nodos), np.inf)

# Calculamos la distancia euclidiana entre nodos
for i in range(1, num_nodos + 1): #Recorremos las filas
    for j in range(1, num_nodos + 1): #Recorremos las columnas
        if i != j:
            xi, yi = datos_nodos[i] #Guardamos las coordenadas del primer nodo del enlace
            xj, yj = datos_nodos[j] #Guardamos las coordenadas del segundo nodo del enlace
            distancia = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) #Calculamos la distancia entre los nodos del enlace
            if distancia <= 20:
                matriz_costos[i-1, j-1] = round(distancia,1) #Si la distancia es menor a 20 unidades la guardamos con un decimal

modelo = ConcreteModel()

# Creamos los conjuntos
modelo.N = RangeSet(num_nodos)
modelo.A = [(i+1, j+1) for i in range(num_nodos) for j in range(num_nodos) if matriz_costos[i, j] < np.inf]

# Inicializamos los parámetros
modelo.c = Param(modelo.A, initialize={(i, j): matriz_costos[i-1, j-1] for (i, j) in modelo.A})

# Creamos la variable de decisión binaria
modelo.x = Var(modelo.A, within=Binary)

# Función objetivo: buscamos minimizar el costo total del trayecto
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

# Resolvemos el modelo
solver = SolverFactory('glpk')
resultado = solver.solve(modelo)
ruta_optima = []
# Mostramos la solución
print("Ruta de mínimo costo de 4 a 6:")
for (i, j) in modelo.A:
    if modelo.x[i, j].value == 1:
        ruta_optima.append((i,j))
        print(f"Nodo {i} -> Nodo {j} (Costo: {modelo.c[i, j]})")

graficar_red(datos_nodos, modelo, matriz_costos, ruta_optima)
