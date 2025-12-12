#!/usr/bin/env python3

# @author:
# Este modulo contiene todas las distribuciones utilizadas para realizar el proyecto.

##################################################################
# Imports necesarios
##################################################################

import numpy as np
from scipy.stats import sem, t


##################################################################
# Funciones para calcular la distribucion.
##################################################################


# ----------------- DISTRIBUCIÓN NORMAL -----------------

def distribucion_normal(n: int, **params):

    """
    La distribucion normal.

    Toma de parametros de entrada:
    * n: el tamaño de el arreglo de salida.
    * params: diccionario con parámetros para la distribucion.
    
    Para la distribucion normal se utilizan:
        * loc: Para referirnos a la media.
        * scale: Para referirnos a la desviación estándar.

    """
    loc = params.get('mu', 0.0)
    scale = params.get('sigma', 1.0)
    return np.random.normal(loc = loc, scale = scale, size = n)

# ----------------- DISTRIBUCIÓN LOGNORMAL -----------------

def distribucion_lognormal(n: int, **params):

    """
    La distribucion lognormal.

    Toma de parametros de entrada:
    * n: el tamaño de el arreglo de salida.
    * params: diccionario con parámetros para la distribucion.

    Para la distribucion normal se utilizan:
        * param1: Para referirnos a la media.
        * param2: Para referirnos a la desviación estándar.

    """
    return np.random.lognormal(**params, size = n)


##################################################################
# Funcion para generar las instancias
##################################################################

def generador_instancias(distr: str, n: int, **params):
    """

    Esta función se encarga de generar las instancias que se utilizarán para hacer los experimentos.

    """
    if distr == 'normal':
        return distribucion_normal(n = n, **params)

    elif distr == 'lognormal':
        return distribucion_lognormal(n = n, **params)

    else:
        print('Distribción no válida.')


##################################################################
# Funciones para obtener estadísticos
##################################################################

def estadisticos(data):
    """
    
    Regresa (media, error_std) para barras verticales en matplotlib.
    
    """
    media = np.mean(data)
    error = t.ppf(0.975, len(data)-1) * sem(data)   # 95% CI
    return media, error

def main():
    pass

if __name__ == "__main__":
    main()