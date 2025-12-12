#!/usr/bin/env python3

# @author:
# Este modulo contiene el codigo del experimento principal.

##################################################################
# Imports necesarios
##################################################################

import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging
import os
from pathlib import Path
import argparse

# Imports de modulos propios
from distribuciones_elias import *
from algoritmos_elias import *

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##################################################################
# Funciones
##################################################################

def parse_arguments():
    """
    Generada completamente por Gemini.
    """
    parser = argparse.ArgumentParser(description="Simulación de Algoritmos de Balanceo de Carga")
    
    # Argumentos configurables
    parser.add_argument('--instancias', nargs='+', type=int, default=[10, 50, 100, 200, 400], 
                        help='Lista de tamaños de instancias (ej: 10 50 100)')
    
    parser.add_argument('--procesadores', type=int, default=10, 
                        help='Número de procesadores (m)')
    
    parser.add_argument('--experimentos', type=int, default=10, 
                        help='Número de experimentos por tamaño de instancia')
    
    parser.add_argument('--distribucion', type=str, default='normal', choices=['normal', 'lognormal'],
                        help='Tipo de distribución a usar')
    
    parser.add_argument('--mu', type=float, default=3.0, help='Media (loc) para la distribución')
    parser.add_argument('--sigma', type=float, default=1.0, help='Desv. Estándar (scale)')

    return parser.parse_args()


def main():
    logging.info('Inicio de la ejecución del programa...')

    # --- PROCESAMIENTO DE ARGUMENTOS ---
    # Obtenemos los argumentos de la terminal
    args = parse_arguments()
    # Parametros actualizados con args..
    logging.info('Selección de parámetros...')
    numero_instancias = args.instancias
    distribucion = args.distribucion
    params = {'loc': args.mu, 'scale': args.sigma}
    logging.info(f"Parámetros de distribución: {params}")

    NUMERO_PROCESADORES = args.procesadores
    NUMERO_EXPERIMENTOS = args.experimentos


    # Creación de directorio y timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_instancias_str = "-".join(map(str, numero_instancias)) 

    nombre_directorio = f'Ejecucion_{distribucion}_NEXP_{NUMERO_EXPERIMENTOS}_NPROC_{NUMERO_PROCESADORES}_INSTANCIAS_{nombre_instancias_str}_{timestamp}'
    output_dir = Path(nombre_directorio)

    try:
        output_dir.mkdir(parents = True, exist_ok = True)
        logging.info(f'Directorio creado: {output_dir}')
    except OSError as e:
        logging.error(f'Error al crear el directorio: {e}')
        return # Salimos si no se puede crear carpeta

    soluciones = {k: {} for k in ['Greedy_2Aprox', 'Greedy_15Aprox', 'Gurobi']}
    tiempos    = {k: {} for k in ['Greedy_2Aprox', 'Greedy_15Aprox', 'Gurobi']}

    logging.info('Inicio de la ejecución de los algoritmos...')
    for n in numero_instancias:
        # Inicializar listas por tamaño de instancia
        logging.info(f'>>> Procesando instancias de tamaño n={n}...')
        for k in soluciones.keys():
            soluciones[k][n] = []
            tiempos[k][n] = []

        for i in range(NUMERO_EXPERIMENTOS):
            logging.info(f'Generando instancias (experimento {i})...')
            instancia = generador_instancias(distr=distribucion, n=n, **params) 

            logging.info(f'Ejecutando greedy 2a...')
            t_inicio = time.perf_counter()
            sol = greedy_2aproximacion(instancia, NUMERO_PROCESADORES)
            t_fin = time.perf_counter()
            
            soluciones['Greedy_2Aprox'][n].append(sol)
            tiempos['Greedy_2Aprox'][n].append(t_fin - t_inicio)

            logging.info(f'Ejecutando greedy 15a...')
            t_inicio = time.perf_counter()
            sol = greedy_1_5_aproximacion(instancia, NUMERO_PROCESADORES)
            t_fin = time.perf_counter()
            
            soluciones['Greedy_15Aprox'][n].append(sol)
            tiempos['Greedy_15Aprox'][n].append(t_fin - t_inicio)

            logging.info(f'Ejecutando gurobi...')
            t_inicio = time.perf_counter()
            sol = solucion_gurobi(instancia, NUMERO_PROCESADORES)
            t_fin = time.perf_counter()
            
            soluciones['Gurobi'][n].append(sol)
            tiempos['Gurobi'][n].append(t_fin - t_inicio)
    logging.info('Fin de la obtención de resultados.')

    # Diccionarios: media e IC para cada algoritmo

    logging.info('Inicio del cálculo de métricas estadísticas y normalización...')
    promedios = {k: [] for k in soluciones.keys()}
    errores  = {k: [] for k in soluciones.keys()}

    for n in numero_instancias:
        for k in soluciones.keys():
            media, ic = estadisticos(soluciones[k][n]) # Viene del archivo de distribuciones
            promedios[k].append(media)
            errores[k].append(ic)

    # Calculamos las mismas estadisticas para los tiempos
    promedios_t = {k: [] for k in tiempos.keys()}
    errores_t   = {k: [] for k in tiempos.keys()}

    for n in numero_instancias:
        for k in tiempos.keys():
            # Usamos tu misma función estadisticos que ya importa t-student
            media, ic = estadisticos(tiempos[k][n]) 
            promedios_t[k].append(media)
            errores_t[k].append(ic)
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
    logging.info('Fin del cálculo de estadísticas.')

    ##################################################################
    # GRÁFICAS RESPECTO A LAS INSTANCIAS
    ##################################################################

    logging.info('Inicio de la generación de gráficas...')

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
        fmt='d--', linewidth=2, markersize=7, label="Greedy 1.5-Aprox / Gurobi"
    )

    ax.set_title("Comparación normalizada sin cotas")
    ax.set_xlabel("Tamaño de la instancia (n)")
    ax.grid(True)
    ax.legend()

    ##############################################

    plt.tight_layout()
    # Modificación: Usamos output_dir para guardar dentro de la carpeta
    nombre_archivo_cotas = f'GráficaCota_{distribucion}_NEXP_{NUMERO_EXPERIMENTOS}_NPROC_{NUMERO_PROCESADORES}_INSTANCIAS_{nombre_instancias_str}_{timestamp}.png'
    plt.savefig(output_dir / nombre_archivo_cotas)
    logging.info('Se guardó con éxito la gráfica de cotas.')

    ##############################################
    #  GRÁFICA 3 — TIEMPOS DE EJECUCIÓN
    ##############################################
    plt.figure(figsize=(10, 6)) # Nueva figura independiente para limpieza

    # Graficar Greedy 2-Aprox
    plt.errorbar(
        numero_instancias, promedios_t['Greedy_2Aprox'], yerr=errores_t['Greedy_2Aprox'],
        fmt='s--', linewidth=2, markersize=7, label="Greedy 2-Aprox (O(n))"
    )

    # Graficar Greedy 1.5-Aprox
    plt.errorbar(
        numero_instancias, promedios_t['Greedy_15Aprox'], yerr=errores_t['Greedy_15Aprox'],
        fmt='d--', linewidth=2, markersize=7, label="Greedy 1.5-Aprox (O(n log n))"
    )

    # Graficar Gurobi
    plt.errorbar(
        numero_instancias, promedios_t['Gurobi'], yerr=errores_t['Gurobi'],
        fmt='o-', linewidth=2, markersize=7, color='black', label="Gurobi (Exponencial/Complejo)"
    )

    plt.yscale('log')  # <--- CLAVE: Escala logarítmica para ver las diferencias
    plt.title("Comparación de Tiempos de Ejecución (Escala Logarítmica)")
    plt.xlabel("Tamaño de la instancia (n)")
    plt.ylabel("Tiempo promedio (segundos)")
    plt.grid(True, which="both", ls="--", alpha=0.5) # Grid para escala log
    plt.legend()

    # Guardamos la figura de nuestros experimentos
    # Modificación: Usamos output_dir para guardar dentro de la carpeta
    nombre_archivo_tiempos = f'GráficaTiempo_{timestamp}.png'
    plt.savefig(output_dir / nombre_archivo_tiempos)
    logging.info('Se guardó con éxito la gráfica de tiempos de ejecución.')

if __name__ == "__main__":
    main()