#!/usr/bin/env python3

# @author:
# Este modulo contiene el codigo del experimento principal.

##################################################################
# Imports necesarios
##################################################################

import numpy as np
from distribuciones_elias import *
from algoritmos_elias import *
from scipy.stats import t
import matplotlib.pyplot as plt


##################################################################
# Funciones
##################################################################

# Parametros que vamos a pedir.
numero_instancias = [10, 50, 100, 200, 400]
distribucion = 'normal'
params = {'loc': 3, 'scale': 1}
NUMERO_PROCESADORES = 10
NUMERO_EXPERIMENTOS = 10

soluciones = {k: {} for k in ['Greedy_2Aprox', 'Greedy_15Aprox', 'Gurobi']}

for n in numero_instancias:
    # Crear listas vacías para cada tamaño
    soluciones['Greedy_2Aprox'][n] = []
    soluciones['Greedy_15Aprox'][n] = []
    soluciones['Gurobi'][n] = []

    # Hacer los experimentos
    for _ in range(NUMERO_EXPERIMENTOS):
        # Obtenemos el respectivo arreglo de instancias
        instancia = generador_instancias(distr=distribucion, n=n, **params) 

        # Hacmos el cálculo de la solución para cada uno de los algoritmos
        soluciones['Greedy_2Aprox'][n].append(greedy_2aproximacion(instancia, NUMERO_PROCESADORES))
        soluciones['Greedy_15Aprox'][n].append(greedy_1_5_aproximacion(instancia, NUMERO_PROCESADORES))
        soluciones['Gurobi'][n].append(solucion_gurobi(instancia, NUMERO_PROCESADORES))

# Diccionarios: media e IC para cada algoritmo
promedios = {k: [] for k in soluciones.keys()}
errores  = {k: [] for k in soluciones.keys()}

for n in numero_instancias:
    for k in soluciones.keys():
        media, ic = estadisticos(soluciones[k][n]) # Viene del archivo de distribuciones
        promedios[k].append(media)
        errores[k].append(ic)

# Obtenemos los ratios para hacer las gráficas
ratios = {k: [] for k in ['Greedy_2Aprox', 'Greedy_15Aprox']}
errores_ratio = {k: [] for k in ['Greedy_2Aprox', 'Greedy_15Aprox']}

for n in numero_instancias:
    # Convertimos a numpy array para facilitar operaciones vectoriales
    vals_gurobi = np.array(soluciones['Gurobi'][n])

    for alg in ['Greedy_2Aprox', 'Greedy_15Aprox']:
        vals_alg = np.array(soluciones[alg][n])

        # 1. Calculamos el ratio INSTANCIA POR INSTANCIA
        # Aquí comparamos "peras con peras": Greedy[i] vs Gurobi[i]
        ratios_individuales = vals_alg / vals_gurobi

        # 2. Calculamos el promedio de estos ratios
        media_ratio = np.mean(ratios_individuales)

        # 3. Calculamos el Intervalo de Confianza (95%) para los ratios
        # Error Estándar de la Media (SEM)
        std_dev = np.std(ratios_individuales, ddof=1) # ddof=1 para muestra insesgada
        sem = std_dev / np.sqrt(len(ratios_individuales)) # Error estándar
        
        # Valor crítico t de Student (para N=10 y 95% de confianza)
        # t.ppf(0.975, df) es el valor para dos colas con alpha=0.05
        t_critico = t.ppf(0.975, df=len(ratios_individuales)-1)
        
        ic = sem * t_critico

        ratios[alg].append(media_ratio)
        errores_ratio[alg].append(ic)


##################################################################
# GRÁFICAS
##################################################################

fig, axs = plt.subplots(1, 2, figsize=(15,9))

ax = axs[0]

ax.errorbar(
    numero_instancias, ratios['Greedy_2Aprox'], yerr=errores_ratio['Greedy_2Aprox'],
    fmt='s--', linewidth=2, markersize=7, label="Greedy 2-Aprox / Gurobi"
)

ax.errorbar(
    numero_instancias, ratios['Greedy_15Aprox'], yerr=errores_ratio['Greedy_15Aprox'],
    fmt='d--', linewidth=2, markersize=7, label="Greedy 1.5-Aprox / Gurobi"
)

# Líneas horizontales
ax.axhline(2, color='gray', linestyle='--', linewidth=1.5, label='Cota teórica 2')
ax.axhline(1.5, color='gray', linestyle=':', linewidth=1.5, label='Cota teórica 1.5')
ax.axhline(1, color='black', linestyle='-', linewidth=2, alpha=0.6, label='Gurobi (óptimo)')

ax.set_title("Comparación normalizada con cotas teóricas")
ax.set_xlabel("Tamaño de la instancia (n)")
ax.set_ylabel("Ratio respecto a Gurobi")
ax.grid(True)
ax.legend()

##############################################
#  GRÁFICA 2 — SIN COTAS
##############################################

ax = axs[1]

ax.errorbar(
    numero_instancias, ratios['Greedy_2Aprox'], yerr=errores_ratio['Greedy_2Aprox'],
    fmt='s--', linewidth=2, markersize=7, label="Greedy 2-Aprox / Gurobi"
)

ax.errorbar(
    numero_instancias, ratios['Greedy_15Aprox'], yerr=errores_ratio['Greedy_15Aprox'],
    fmt='d--', linewidth=2, markersiz=7, label="Greedy 1.5-Aprox / Gurobi"
)

ax.set_title("Comparación normalizada sin cotas")
ax.set_xlabel("Tamaño de la instancia (n)")
ax.grid(True)
ax.legend()

##############################################

plt.tight_layout()
plt.show()


