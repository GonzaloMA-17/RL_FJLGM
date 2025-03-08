import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def plot(list_stats):
    """
    Grafica la proporción de recompensas obtenidas por episodio.

    Parámetros:
    -----------
    list_stats : list
        Lista con los valores de recompensas por episodio.
    """
    indices = list(range(len(list_stats)))

    plt.figure(figsize=(6, 3))
    plt.plot(indices, list_stats, label="Proporción de recompensas")
    plt.title("Proporción de recompensas")
    plt.xlabel("Episodio")
    plt.ylabel("Proporción")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_episode_lengths(episode_lengths, window=50):
    """
    Grafica la longitud de los episodios a lo largo del entrenamiento.

    Parámetros:
    -----------
    episode_lengths : list
        Lista con las longitudes de cada episodio.
    window : int, opcional (por defecto 50)
        Tamaño de la ventana para calcular la media móvil.
    """
    indices = list(range(len(episode_lengths)))

    plt.figure(figsize=(6, 3))
    plt.plot(indices, episode_lengths, label="Longitud de episodio", alpha=0.6)

    # Media móvil para suavizar la curva
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(episode_lengths)), moving_avg, label="Tendencia (Media Móvil)", color='red')

    plt.title("Longitud de los episodios")
    plt.xlabel("Episodio")
    plt.ylabel("Longitud")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_policy_arrows_no_grid(Q, env):
    """
    Dibuja la política aprendida en un mapa FrozenLake, coloreando las celdas según su tipo
    y mostrando flechas para indicar la acción óptima en cada celda.
    No muestra líneas internas y añade una leyenda con flechas y colores.

    Parámetros:
    -----------
    Q : np.ndarray
        Q-table con dimensiones [n_estados, n_acciones].
    env : gym.Env
        Entorno de Gymnasium (ej. FrozenLake-v1).
    """
    # Mapeo de índice de acción -> símbolo de flecha
    arrow_dict = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    # Dimensiones del mapa según la descripción interna (unwrapped)
    num_filas, num_columnas = env.unwrapped.desc.shape

    # Obtener la mejor acción para cada estado
    best_actions = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
    best_actions_2d = np.reshape(best_actions, (num_filas, num_columnas))

    # Crear figura
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, num_columnas - 0.5)
    ax.set_ylim(num_filas - 0.5, -0.5)

    # Pintar cada celda según el tipo de casilla
    for fila in range(num_filas):
        for columna in range(num_columnas):
            celda = env.unwrapped.desc[fila, columna].decode("utf-8")
            # Asignar color según el carácter
            if celda == 'S':  # Start
                color = "gray"
            elif celda == 'G':  # Goal
                color = "green"
            elif celda == 'F':  # Frozen
                color = "skyblue"
            elif celda == 'H':  # Hole (lo llamas 'Agua' si prefieres)
                color = "red"
            else:
                color = "white"

            # Dibujar rectángulo de fondo
            ax.add_patch(plt.Rectangle((columna - 0.5, fila - 0.5),
                                       1, 1, color=color, ec="black"))

    # Añadir flechas o texto en cada celda
    for fila in range(num_filas):
        for columna in range(num_columnas):
            celda = env.unwrapped.desc[fila, columna].decode("utf-8")
            if celda in ['H', 'G']:
                # H -> Agujero/Agua, G -> Meta
                ax.text(columna, fila, celda, ha="center", va="center",
                        fontsize=14, color="white", fontweight="bold")
            else:
                # Para 'S' y 'F', mostramos la flecha de la acción óptima
                accion_optima = best_actions_2d[fila, columna]
                flecha = arrow_dict.get(accion_optima, '?')
                # Elegir color de texto (blanco en S, negro en F)
                txt_color = "white" if celda == 'S' else "black"
                ax.text(columna, fila, flecha, ha="center", va="center",
                        fontsize=16, color=txt_color, fontweight="bold")

    # Ocultar los ejes y las divisiones internas
    ax.set_xticks([])
    ax.set_yticks([])

    # ───────────── CREAR LEYENDA ─────────────
    # 1. Flechas para las acciones
    left_arrow = mlines.Line2D([], [], color='black', marker=r'$\leftarrow$', 
                               linestyle='None', markersize=12, label='Acción 0: Izquierda')
    down_arrow = mlines.Line2D([], [], color='black', marker=r'$\downarrow$', 
                               linestyle='None', markersize=12, label='Acción 1: Abajo')
    right_arrow = mlines.Line2D([], [], color='black', marker=r'$\rightarrow$', 
                                linestyle='None', markersize=12, label='Acción 2: Derecha')
    up_arrow = mlines.Line2D([], [], color='black', marker=r'$\uparrow$', 
                             linestyle='None', markersize=12, label='Acción 3: Arriba')

    # 2. Colores para S, F, H, G
    patch_s = mpatches.Patch(color="gray", label="S (Inicio)")
    patch_f = mpatches.Patch(color="skyblue", label="F (Hielo)")
    patch_h = mpatches.Patch(color="red", label="H (Agua)")
    patch_g = mpatches.Patch(color="green", label="G (Meta)")

    # Unir todo en un solo listado
    legend_handles = [left_arrow, down_arrow, right_arrow, up_arrow,
                      patch_s, patch_f, patch_h, patch_g]

    ax.legend(handles=legend_handles,
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.set_title("Política Aprendida (Monte Carlo On-Policy)", pad=10)
    plt.show()


def plot_comparison(stats_list, labels, title="Comparación de Resultados de Entrenamiento"):
    """
    Compara la evolución de las recompensas de diferentes algoritmos.

    Parámetros:
    -----------
    stats_list : list
        Lista de listas con recompensas obtenidas por cada agente.
    labels : list
        Lista de nombres de cada agente.
    title : str, opcional
        Título del gráfico.
    """
    plt.figure(figsize=(10, 6))

    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c']

    for stats, label, line_style, color in zip(stats_list, labels, line_styles, colors):
        plt.plot(stats, label=label, linestyle=line_style, color=color, linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel("Episodios", fontsize=14)
    plt.ylabel("Recompensa Promedio", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()



def plot_all_three(list_stats, episode_lengths, Q, env, actions, window=50):
    """
    Muestra en una sola figura:
      1) Proporción de recompensas por episodio
      2) Longitud de los episodios (con media móvil)
      3) Política aprendida en el entorno (flechas)

    Además, se incluye un parámetro `actions` (lista de enteros 0..3) que
    se mostrará en la leyenda como "Política escogida por el agente".

    Parámetros:
    -----------
    list_stats : list
        Lista con los valores de recompensas por episodio.
    episode_lengths : list
        Lista con la longitud de cada episodio.
    Q : np.ndarray
        Q-table con dimensiones [n_estados, n_acciones].
    env : gym.Env
        Entorno de Gymnasium (ej. FrozenLake-v1).
    actions : list[int]
        Lista de enteros (0,1,2,3) que representan la política escogida por el agente.
    window : int
        Tamaño de la ventana para calcular la media móvil de la longitud de episodios.
    """
    # Si el entorno tiene el atributo 'env', usamos ese
    if hasattr(env, 'env'):
        env_inner = env.env
    else:
        env_inner = env

    # Crear la figura con 3 subplots en una sola fila
    # Crear la figura con 3 subplots en una sola fila
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ──────────────────────────────
    # 1) Proporción de recompensas
    # ──────────────────────────────
    indices = list(range(len(list_stats)))
    ax1.plot(indices, list_stats, label="Proporción de recompensas", color="blue")
    ax1.set_title("Proporción de Recompensas")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Proporción")
    ax1.grid(True)
    ax1.legend()

    # ─────────────────────────────────────────────────────
    # 2) Longitud de episodios (con media móvil opcional)
    # ─────────────────────────────────────────────────────
    indices = list(range(len(episode_lengths)))
    ax2.plot(indices, episode_lengths, label="Longitud de episodio", alpha=0.6, color="green")

    # Calcular y trazar la media móvil
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(episode_lengths)), moving_avg,
                 label=f"Media Móvil (window={window})", color='red', linewidth=2)

    ax2.set_title("Longitud de los Episodios")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Longitud")
    ax2.grid(True)
    ax2.legend()

    # ────────────────────────────────────────────────────────────
    # 3) Política aprendida (flechas en un mapa FrozenLake)
    # ────────────────────────────────────────────────────────────

    # Mapeo de índice de acción -> símbolo de flecha
    arrow_dict = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    # Dimensiones del mapa
    num_filas, num_columnas = env.unwrapped.desc.shape

    # Mejor acción para cada estado
    best_actions = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
    best_actions_2d = np.reshape(best_actions, (num_filas, num_columnas))

    # Ajustar límites del subplot
    ax3.set_xlim(-0.5, num_columnas - 0.5)
    ax3.set_ylim(num_filas - 0.5, -0.5)

    # Pintar cada celda
    for fila in range(num_filas):
        for columna in range(num_columnas):
            celda = env.unwrapped.desc[fila, columna].decode("utf-8")
            if celda == 'S':
                color = "gray"
            elif celda == 'G':
                color = "green"
            elif celda == 'F':
                color = "skyblue"
            elif celda == 'H':
                color = "red"
            else:
                color = "white"

            ax3.add_patch(plt.Rectangle((columna - 0.5, fila - 0.5),
                                        1, 1, color=color, ec="black"))

    # Añadir flechas o letras
    for fila in range(num_filas):
        for columna in range(num_columnas):
            celda = env.unwrapped.desc[fila, columna].decode("utf-8")
            if celda == 'S':
                # Mostramos "Start" en lugar de flecha
                ax3.text(columna, fila, "Start", ha="center", va="center",
                         fontsize=12, color="white", fontweight="bold")
            elif celda in ['H', 'G']:
                # H -> Agujero/Agua, G -> Meta
                ax3.text(columna, fila, celda, ha="center", va="center",
                         fontsize=14, color="white", fontweight="bold")
            else:
                # Para 'F', mostramos la flecha de la acción óptima
                accion_optima = best_actions_2d[fila, columna]
                flecha = arrow_dict.get(accion_optima, '?')
                txt_color = "black"
                ax3.text(columna, fila, flecha, ha="center", va="center",
                         fontsize=16, color=txt_color, fontweight="bold")

    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Política Aprendida (Monte Carlo On-Policy)", pad=10)

    # ────────────────────────────────────────────────────────────
    # Leyenda de flechas y colores
    # ────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    left_arrow = mlines.Line2D([], [], color='black', marker=r'$\leftarrow$', 
                               linestyle='None', markersize=12, label='Acción 0: Izquierda')
    down_arrow = mlines.Line2D([], [], color='black', marker=r'$\downarrow$', 
                               linestyle='None', markersize=12, label='Acción 1: Abajo')
    right_arrow = mlines.Line2D([], [], color='black', marker=r'$\rightarrow$', 
                                linestyle='None', markersize=12, label='Acción 2: Derecha')
    up_arrow = mlines.Line2D([], [], color='black', marker=r'$\uparrow$', 
                             linestyle='None', markersize=12, label='Acción 3: Arriba')
    patch_s = mpatches.Patch(color="gray", label="S (Inicio)")
    patch_f = mpatches.Patch(color="skyblue", label="F (Hielo)")
    patch_h = mpatches.Patch(color="red", label="H (Agua)")
    patch_g = mpatches.Patch(color="green", label="G (Meta)")

    # Añadir la política escogida por el agente a la leyenda
    # Convertimos 'actions' (lista de enteros) en cadena
    # policy_str = ', '.join(str(a) for a in actions)
    policy_str =  str(actions)
    policy_line = mlines.Line2D([], [], color='none', marker='',
                                linestyle='None', label=f"\n Política escogida: [{policy_str}]")

    # Unir todos los ítems de leyenda
    legend_handles = [
        left_arrow, down_arrow, right_arrow, up_arrow,
        patch_s, patch_f, patch_h, patch_g,
        policy_line  # <-- Aquí agregamos la línea con la lista de acciones
    ]

    ax3.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1),
               loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()


def plot_episode_lengths_comparison(episode_lengths_list, labels, window=50):
    """
    Compara las longitudes de los episodios de diferentes agentes o algoritmos en una sola figura.

    Parámetros:
    -----------
    episode_lengths_list : list of lists
        Lista que contiene varias listas de longitudes de episodios.
        Por ejemplo, [episodios_algo1, episodios_algo2, ...].
    labels : list of str
        Nombres de cada lista de episodios (para la leyenda).
    window : int, opcional (por defecto 50)
        Tamaño de la ventana para calcular la media móvil.
    """
    # Crear figura
    plt.figure(figsize=(8, 4))
    
    # Trazar cada lista de longitudes en la misma gráfica
    for episode_lengths, label in zip(episode_lengths_list, labels):
        indices = range(len(episode_lengths))
        plt.plot(indices, episode_lengths, alpha=0.6, label=f"{label}: Longitud de Episodio")
        
        # Media móvil
        if len(episode_lengths) >= window:
            moving_avg = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
            plt.plot(range(window - 1, len(episode_lengths)), moving_avg, 
                     label=f"{label}: Media Móvil (window={window})")
    
    plt.title("Comparación de Longitud de Episodios")
    plt.xlabel("Episodio")
    plt.ylabel("Longitud")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
