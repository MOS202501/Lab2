from pyomo.environ import *
from help_functions import *

# Definimos el modelo
model = ConcreteModel()

# Creamos los conjuntos

origen = {
    1: {"nombre": "Bogotá", "oferta": 550},
    2: {"nombre": "Medellin","oferta": 700}
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

# Creamos la variable de decisión
model.x = Var(oferta, demanda, domain=NonNegativeReals)

# Función objetivo: minimizar el costo de transporte
model.obj = Objective(
    expr=sum(
        (model.x[1, j] * destinos[j]["costoBog"] if destinos[j]["costoBog"] != np.inf else 0) +
        (model.x[2, j] * destinos[j]["costoMed"] if destinos[j]["costoMed"] != np.inf else 0)
        for j in demanda
    ),
    sense=minimize
)

# Restricción de oferta
model.oferta_constraint = ConstraintList()
for i in oferta:
    model.oferta_constraint.add(sum(model.x[i, j] for j in demanda) <= origen[i]["oferta"])

# Restricción de demanda
model.demanda_constraint = ConstraintList()
for j in demanda:
    model.demanda_constraint.add(sum(model.x[i, j] for i in oferta) == destinos[j]["demanda"])

model.infinito_constraint = ConstraintList()
for i in oferta:
    for j in demanda:
        if destinos[j]["costoBog"] == np.inf and i == 1:
            model.infinito_constraint.add(model.x[i, j] == 0)
        if destinos[j]["costoMed"] == np.inf and i == 2:
            model.infinito_constraint.add(model.x[i, j] == 0)


# Resolvemos el modelo
solver = SolverFactory('glpk')
solver.solve(model)

# Creamos DataFrame con los resultados
data_resultados = []
for j in destinos: 
        for i in origen:
            cantidad = model.x[i, j].value
            costo_unitario = destinos[j]["costoBog"] if i == 1 else destinos[j]["costoMed"]
            costo = cantidad * costo_unitario if costo_unitario != np.inf else 0
            data_resultados.append([origen[i]["nombre"], destinos[j]["nombre"], cantidad, costo])
    
df = pd.DataFrame(data_resultados, columns=["Origen", "Destino", "Cantidad", "Costo"])

print(df)

# Obtenemos el costo total del transporte para desplegarlo

costo_total = sum(
    (model.x[1, j].value * destinos[j]["costoBog"] if destinos[j]["costoBog"] != np.inf else 0) +
    (model.x[2, j].value * destinos[j]["costoMed"] if destinos[j]["costoMed"] != np.inf else 0)
    for j in demanda
)


print(f"Costo total del transporte: {costo_total:.2f} USD")
plot_transport_heatmap(model, origen,destinos, costo_total)