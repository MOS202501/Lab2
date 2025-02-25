from pyomo.environ import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_allocation(Model, recursos_disponibles, aviones_disponibles):
    recursos_labels = [f"Recurso {i}" for i in recursos_disponibles]
    aviones_labels = [f"Avión {j}" for j in aviones_disponibles]

    # Obtener valores de la matriz X[i, j]
    data = np.array([[Model.x[i, j].value for j in aviones_disponibles]
                    for i in recursos_disponibles])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Graficar las barras apiladas
    bottom = np.zeros(len(aviones_disponibles))
    colors = plt.cm.viridis(np.linspace(0, 1, len(recursos_disponibles)))

    for i, (recurso, color) in enumerate(zip(recursos_labels, colors)):
        ax.bar(aviones_labels, data[i],
               label=recurso, bottom=bottom, color=color)
        bottom += data[i]  # Acumular para la siguiente barra

    ax.set_ylabel("Cantidad Asignada")
    ax.set_xlabel("Aviones")
    ax.set_title("Distribución de Recursos en Aviones")
    ax.legend(title="Recursos", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('exercise_1_1.png')
    plt.close()


def plot_pretty_heatmap(Model, recursos_disponibles, aviones_disponibles):
    # Crear una matriz con los valores asignados
    data = np.array([[Model.x[i, j].value for j in aviones_disponibles]
                    for i in recursos_disponibles])

    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(data,
                      index=[f"Recurso {i}" for i in recursos_disponibles],
                      columns=[f"Avión {j}" for j in aviones_disponibles])

    # Configuración del heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".1f", linewidths=0.8,
                cbar_kws={'label': 'Cantidad Asignada'},
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                vmax=np.max(data) * 1.1)  # Ajuste de escala para mejor contraste

    plt.title("Asignación de Recursos a Aviones",
              fontsize=14, fontweight="bold")
    plt.xlabel("Aviones", fontsize=12)
    plt.ylabel("Recursos", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('exercise_1_2.png')
    plt.close()


def plot_queens_solution(model, N=8):
    """
    Función para visualizar la solución de cobertura mínima de un tablero de ajedrez con damas.

    Parámetros:
    - model: Modelo de Pyomo resuelto.
    - N: Tamaño del tablero (por defecto 8x8).
    """
    # Crear matriz del tablero
    board = np.zeros((N, N), dtype=int)

    # Extraer la solución del modelo
    for i in range(N):
        for j in range(N):
            if model.x[i, j].value == 1:
                board[i, j] = 1  # Dama en esa posición

    # Crear el heatmap con Seaborn
    plt.figure(figsize=(8, 8))
    sns.heatmap(board, annot=False, cmap="coolwarm",
                linewidths=0.5, linecolor='black', cbar=False)

    # Agregar las reinas en el tablero
    for i in range(N):
        for j in range(N):
            if board[i, j] == 1:
                plt.text(j + 0.5, i + 0.5, '♛', ha='center',
                         va='center', fontsize=30, color='black')

    # Estilizar el gráfico
    plt.xticks(np.arange(N) + 0.5, [chr(97 + i)
               for i in range(N)], fontsize=14)  # Etiquetas A-H
    plt.yticks(np.arange(N) + 0.5, np.arange(1, N+1)
               [::-1], fontsize=14)  # Etiquetas 8-1
    plt.xlabel("Columnas", fontsize=16)
    plt.ylabel("Filas", fontsize=16)
    plt.title("Cobertura Mínima del Tablero con Damas", fontsize=18)
    plt.gca().invert_yaxis()  # Invertir eje Y para que se vea como un tablero de ajedrez
    plt.savefig('exercise_3.png')
    plt.close()


def plot_transport_heatmap(model, origenes, destinos):
    """Grafica un heatmap de la solución del problema de transporte con los ejes invertidos e incluye el costo."""
    
    data = []
    for j in destinos:  # Iteramos sobre destinos (filas)
        for i in origenes:  # Iteramos sobre orígenes (columnas)
            cantidad = model.x[i, j].value
            costo_unitario = destinos[j]["costoBog"] if i == 1 else destinos[j]["costoMed"]
            costo = cantidad * costo_unitario if costo_unitario != np.inf else 0
            data.append([origenes[i]["nombre"], destinos[j]["nombre"], cantidad, costo])
    
    df = pd.DataFrame(data, columns=["Origen", "Destino", "Cantidad", "Costo"])
    df["Destino"] = pd.Categorical(df["Destino"], categories=[destinos[j]["nombre"] for j in destinos], ordered=True)


    # Convertimos a formato de tabla pivote para el heatmap
    pivot_cantidad = df.pivot(index="Destino", columns="Origen", values="Cantidad")
    pivot_costo = df.pivot(index="Destino", columns="Origen", values="Costo")
    pivot_df = df.pivot(index="Destino", columns="Origen", values="Cantidad")

    # Crear etiquetas combinadas para cada celda
    labels = np.array([[f"{pivot_cantidad.loc[d, o]:.1f}\n({pivot_costo.loc[d, o]:.1f})"
                        if not np.isnan(pivot_cantidad.loc[d, o]) else ""
                        for o in pivot_cantidad.columns] for d in pivot_cantidad.index])

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_cantidad, annot=labels, fmt='', cmap="Blues", linewidths=0.8, 
                xticklabels=pivot_cantidad.columns, yticklabels=pivot_cantidad.index, 
                cbar_kws={'label': 'Cantidad Transportada'})

    plt.title("Asignación de Transporte", fontsize=14, fontweight="bold")
    plt.xlabel("Orígenes", fontsize=12)
    plt.ylabel("Destinos", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('exercise_2.png')
    plt.close()