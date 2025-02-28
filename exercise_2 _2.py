from pyomo.environ import *
from help_functions import *

# Definir el modelo
model = ConcreteModel()

# Conjuntos

origen = {
    1: {"nombre": "Bogotá", "oferta": 600},
    2: {"nombre": "Medellin","oferta": 650}
}

destinos = {
    1: {"nombre": "Cali", "costoBog": np.inf, "costoMed": 2.5, "demanda": 125},
    2: {"nombre": "Barranquilla", "costoBog": 2.5, "costoMed": np.inf, "demanda": 175},
    3: {"nombre": "Bucaramanga", "costoBog": 1.6, "costoMed": 2.0, "demanda": 225},
    4: {"nombre": "Cartagena", "costoBog": 1.4, "costoMed": 1.0, "demanda": 250},
    5: {"nombre": "Cúcuta", "costoBog": 0.8, "costoMed": 1.0, "demanda": 225},
    6: {"nombre": "Pereira", "costoBog": 1.4, "costoMed": 0.8, "demanda": 200},
}


oferta = range(1, len(origen) + 1)
demanda = range(1, len(destinos) + 1)

# Variables de decisión
model.x = Var(oferta, demanda, domain=NonNegativeReals)

# Función objetivo: minimizar el costo total de transporte
model.obj = Objective(
    expr=sum(
        (model.x[1, j] * destinos[j]["costoBog"] if destinos[j]["costoBog"] != np.inf else 0) +
        (model.x[2, j] * destinos[j]["costoMed"] if destinos[j]["costoMed"] != np.inf else 0)
        for j in demanda
    ),
    sense=minimize
)

# Restricciones de oferta
model.oferta_constraints = ConstraintList()
for i in oferta:
    model.oferta_constraints.add(sum(model.x[i, j] for j in demanda) <= origen[i]["oferta"])

# Restricciones de demanda
model.demanda_constraints = ConstraintList()
for j in demanda:
    model.demanda_constraints.add(sum(model.x[i, j] for i in oferta) == destinos[j]["demanda"])

model.restriccion_infinito = ConstraintList()
for i in oferta:
    for j in demanda:
        if destinos[j]["costoBog"] == np.inf and i == 1:
            model.restriccion_infinito.add(model.x[i, j] == 0)
        if destinos[j]["costoMed"] == np.inf and i == 2:
            model.restriccion_infinito.add(model.x[i, j] == 0)


# Resolver el modelo
solver = SolverFactory('glpk')
solver.solve(model)

# Crear DataFrame con los resultados
data_resultados = []
for j in destinos:  # Iteramos sobre destinos (filas)
        for i in origen:  # Iteramos sobre orígenes (columnas)
            cantidad = model.x[i, j].value
            costo_unitario = destinos[j]["costoBog"] if i == 1 else destinos[j]["costoMed"]
            costo = cantidad * costo_unitario if costo_unitario != np.inf else 0
            data_resultados.append([origen[i]["nombre"], destinos[j]["nombre"], cantidad, costo])
    
df = pd.DataFrame(data_resultados, columns=["Origen", "Destino", "Cantidad", "Costo"])

print(df)
# Obtener el costo total del transporte
costo_total = sum(
    (model.x[1, j].value * destinos[j]["costoBog"] if destinos[j]["costoBog"] != np.inf else 0) +
    (model.x[2, j].value * destinos[j]["costoMed"] if destinos[j]["costoMed"] != np.inf else 0)
    for j in demanda
)


print(f"Costo total del transporte: {costo_total:.2f} USD")
plot_transport_heatmap2(model, origen,destinos)