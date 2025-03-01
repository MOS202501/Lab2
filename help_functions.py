from pyomo.environ import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx


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


def plot_transport_heatmap(model, origenes, destinos, costo_total):
    """Grafica un heatmap de la solución del problema de transporte e incluye el costo."""
    
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

    plt.title(f"Asignación de Transporte\nCosto Total: {costo_total:.2f}", fontsize=14, fontweight="bold")
    plt.xlabel("Orígenes", fontsize=12)
    plt.ylabel("Destinos", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('exercise_2.png')
    plt.close()


def plot_transport_heatmap2(model, origenes, destinos, costo_total):
    """Grafica un heatmap de la solución del problema de transporte e incluye el costo."""
    
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

    plt.title(f"Asignación de Transporte\nCosto Total: {costo_total:.2f}", fontsize=14, fontweight="bold")
    plt.xlabel("Orígenes", fontsize=12)
    plt.ylabel("Destinos", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('exercise_2_2.png')
    plt.close()


def dibujar_grafo(cost_matrix, ruta, filename='tsp_solution.png'):
    G = nx.DiGraph()
    N = len(cost_matrix)

    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(i+1, j+1, weight=cost_matrix[i][j])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', node_size=700, font_size=10)

    edge_labels = {(i, j): f"{cost_matrix[i-1][j-1]:.1f}" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Dibujar ruta óptima en rojo
    nx.draw_networkx_edges(G, pos, edgelist=ruta, edge_color='red', width=2)
    plt.savefig(filename)
    plt.show()


def dibujar_grafo_equipos(cost_matrix, rutas, filename='tsp_solution_equipos.png'):
    G = nx.DiGraph()
    N = len(cost_matrix)

    # Agregar todos los arcos con su costo
    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(i+1, j+1, weight=cost_matrix[i][j])

    # Posiciones de los nodos usando un layout
    pos = nx.spring_layout(G)

    plt.figure(figsize=(8, 6))

    # Dibujar grafo base
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', node_size=700, font_size=10)

    # Etiquetas de los arcos con sus costos
    edge_labels = {(i, j): f"{cost_matrix[i-1][j-1]:.1f}" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Lista de colores para cada equipo (se cicla si hay más rutas que colores)
    colores = ['red', 'blue', 'green', 'orange', 'purple',
               'brown', 'pink', 'olive', 'cyan', 'magenta']

    # Dibujar cada ruta con un color distinto
    for idx, ruta in enumerate(rutas):
        # Convertir la lista de nodos en una lista de arcos
        edge_list = list(zip(ruta, ruta[1:]))
        color = colores[idx % len(colores)]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list,
                               edge_color=color, width=2, label=f'Equipo {idx+1}')

    plt.legend()
    plt.savefig(filename)
    plt.show()


def dibujar_grafo_sin_ruta(cost_matrix, filename='grafo.png'):
    G = nx.DiGraph()
    N = len(cost_matrix)

    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(i+1, j+1, weight=cost_matrix[i][j])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', node_size=700, font_size=10)

    edge_labels = {(i, j): f"{cost_matrix[i-1][j-1]:.1f}" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.savefig(filename)
    plt.show()

def graficar_red(datos_nodos, modelo, matriz_costos, ruta_optima=None):
    """
    Función para graficar la red con los pesos de las aristas y resaltar la ruta óptima.
    
    :param datos_nodos: Diccionario con coordenadas de los nodos {nodo: (x, y)}
    :param modelo: Modelo de Pyomo con las conexiones definidas en modelo.A
    :param matriz_costos: Matriz con los costos entre nodos
    :param ruta_optima: Lista de aristas [(i, j)] que representan la ruta óptima, opcional
    """
    G = nx.Graph()
    
    # Agregar nodos
    for nodo, (x, y) in datos_nodos.items():
        G.add_node(nodo, pos=(x, y))
    
    # Agregar aristas con pesos
    edges_with_weights = []
    for (i, j) in modelo.A:
        weight = matriz_costos[i-1, j-1]
        G.add_edge(i, j, weight=weight)
        edges_with_weights.append((i, j, weight))
    
    # Posiciones de los nodos
    pos = nx.get_node_attributes(G, 'pos')
    
    # Dibujar la red completa con etiquetas de peso
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    # Resaltar la ruta óptima si se proporciona
    if ruta_optima:
        nx.draw_networkx_edges(G, pos, edgelist=ruta_optima, edge_color='red', width=2)
    
    # Etiquetas de los pesos
    edge_labels = {(i, j): f"{w:.1f}" for i, j, w in edges_with_weights}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='black')
    
    
    plt.title("Red de conexiones con pesos entre nodos")
    plt.savefig('exercise_4.png')
    plt.close()