#!/usr/bin/env python3

# @author:
# Este modulo contiene todas las distribuciones utilizadas para realizar el proyecto.

##################################################################
# Imports necesarios
##################################################################

import numpy as np


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
    return np.random.normal(**params, size = n)

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


