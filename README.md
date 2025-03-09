# Título del Trabajo 
## Información
- **Alumnos:** López, Francisco José; Marcos, Gonzalo; 
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** FJLGM

## Descripción 
Este trabajo se centra en el estudio y aplicación de técnicas de aprendizaje por refuerzo (RL) en dos contextos diferentes: el problema del bandido de k-brazos y los entornos complejos utilizando Gymnasium. La práctica 1 aborda el problema del bandido de k-brazos, explorando diferentes algoritmos como e-greedy, UCB y métodos de ascenso de gradiente para maximizar la recompensa acumulada y minimizar el rechazo (regret). En la práctica 2, se investigan entornos complejos mediante el uso de Gymnasium, implementando algoritmos de aprendizaje como Monte Carlo, SARSA y Q-Learning en entornos tabulares y de aproximación. El objetivo principal es comparar y analizar el rendimiento de estos algoritmos en diversos entornos y problemas, proporcionando una comprensión profunda y aplicada de las técnicas de RL.
## Estructura 

```plaintext
|-- 📂 src_agents                                    # Carpeta principal que contiene los agentes de Aprendizaje por Refuerzo
|   |-- 📄 __init__.py                               # Archivo que convierte el directorio en un paquete de Python
|   |-- 📄 agent.py                                  # Clase base para todos los agentes
|   |-- 📄 deepQLearning.py                          # Implementación del agente Deep Q-Learning (DQN)
|   |-- 📄 monteCarloOnPolicy.py                     # Implementación del agente Monte Carlo On-Policy
|   |-- 📄 monteCarloOffPolicy.py                    # Implementación del agente Monte Carlo Off-Policy
|   |-- 📄 qLearning.py                              # Implementación del agente Q-Learning
|   |-- 📄 sarsa.py                                  # Implementación del agente SARSA tabular
|   |-- 📄 sarsaSemiGradiente.py                     # Implementación del agente SARSA Semigradiente
|   |-- 📄 politicas.py                              # Definición de políticas de exploración como epsilon-greedy y softmax

|-- 📂 src_plotting                                  # Carpeta con herramientas de visualización de resultados
|   |-- 📄 __init__.py                               # Archivo que convierte el directorio en un paquete de Python
|   |-- 📄 plotting.py                               # Funciones para graficar recompensas y longitudes de episodios

|-- 📄 main.ipynb                                    # Cuaderno principal con la ejecución general del proyecto
|-- 📄 notebook_aproximaciones_4x4.ipynb             # Evaluación de métodos con aproximación en FrozenLake 4x4
|-- 📄 notebook_tabulares_4x4.ipynb                  # Evaluación de métodos tabulares en FrozenLake 4x4
|-- 📄 README.md                                     # Documentación general del proyecto
|-- 📄 requirements.txt                              # Lista de dependencias necesarias para ejecutar el proyecto
```

## Instalación y Uso 
Para instalar y utilizar este proyecto, sigue los siguientes pasos:
1º Clonar el repositorio
```bash
git clone https://github.com/GonzaloMA-17/RL_FJLGM.git
```  

2º Navega al directorio del proyecto:
```bash
cd RL_FJLGM
```  

3º Instalar todas las dependecias
```bash
pip install -r requirements.txt
```  

4º Ejecutar los scripts o notebooks necesarios

## Tecnologías Utilizadas 
Este proyecto utiliza las siguientes tecnologías:

- Lenguajes: Python
- Herramientas: Jupyter Notebook, NumPy, Pandas, Matplotlib
- Entornos: VSCode (entornos virtuales locales), Google Colab.
