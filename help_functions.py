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
