#!/usr/bin/env python3

# @author:
# Este modulo contiene el codigo del experimento principal.

##################################################################
# Imports necesarios
##################################################################

import numpy as np
from distribuciones_elias import *
from algoritmos import *
from scipy.stats import t
import matplotlib.pyplot as plt


##################################################################
# Funciones
##################################################################

# Parametros que vamos a pedir.
numero_instancias = [10, 50, 100, 200, 400]
distribucion = 'normal'
instancias = []
params = {'loc' : 3, 'scale': 1}

for n in numero_instancias:
    instancias.append(generador_instancias(distr = distribucion, n = n, **params))

# Evaluamos la solucion con los algoritmos del modulo.
#

soluciones = {k:{} for k in ['Greedy_2Aprox', 'Greedy_15Aprox', 'Gurobi']}

for instancia in instancias:
    len_instancia = len(instancia)
    print(len_instancia)
    soluciones['Greedy_2Aprox'][len_instancia] = greedy_2aproximacion(instancia, 10)
    soluciones['Greedy_15Aprox'][len_instancia] = greedy_1_5_aproximacion(instancia, 10)
    soluciones['Gurobi'][len_instancia] = solucion_gurobi(instancia, 10)

print(soluciones)
