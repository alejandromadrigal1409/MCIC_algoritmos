# NOMECLATURA:
# n = número de tareas
# m = número de procesadores
# l[] = duración de tareas

# CONSIDERACIONES:
# probar n = 50, 100, 200, 400
# m = 10 procesadores para todos los casos

# Para distribución exponencial: parametró importante "lambda" = número promedio de eventos por unidad de tiempo

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import heapq
from scipy.stats import t
import matplotlib.pyplot as plt
import time
import os

def generador_instancias(dist, n):
  #np.random.seed(69)
  mu_nom = 3
  sigma_nom = 1
  mu_lognom = 0
  sigma_lognom = 1

  if dist == 0:
    l = distribucion_normal(mu_nom, sigma_nom, n)
  else:
    l = distribucion_lognormal(mu_lognom, sigma_lognom, n)

  return l

# ----------------- DISTRIBUCIÓN NORMAL -----------------
def distribucion_normal(mu, sigma, n):
  l = []
  for i in n:
   l.append(np.random.normal(loc = mu, scale = sigma, size = i))

  return l 

# ----------------- DISTRIBUCIÓN LOGNORMAL -----------------
def distribucion_lognormal(mu, sigma, n):
  l = []
  for i in n:
   l.append(np.random.lognormal(mean = mu, sigma = sigma, size = i))

  return l

# ----------------- SOLUCIÓN CON GUROBI -----------------
def solucion_gurobi(m, N, L):
  makespans = []
  tiempo = []
  for aux in range(len(N)):
    inicio = time.perf_counter() 
    n = N[aux]
    l = L[aux]

    model = gp.Model("P||Cmax")

    # Variables
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")
    Makespan = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")

    # Objetivo: minimizar Cmax
    model.setObjective(Makespan, GRB.MINIMIZE)

    # Restriccion: cada tarea debe estar en exactamente un procesador
    for j in range(n):
        model.addConstr(sum(x[i,j] for i in range(m)) == 1)

    # Restricción: minimizar el makespan:
    for i in range(m):
        model.addConstr(sum(l[j] * x[i,j] for j in range(n)) <= Makespan)

    model.setParam('MIPGap', 0.000)
    model.setParam('TimeLimit', 2)
    model.optimize()

    fin = time.perf_counter() 

    makespans.append([Makespan.X])
    tiempo.append([fin - inicio])
  
  return np.array(makespans), np.array(tiempo)

# ----------------- SOLUCIÓN CON GREEDY 1.5 -----------------
def solucion_greedy_1_5(m, N, L):
  makespans = []
  tiempo = []
  for aux in range(len(N)):
    l = L[aux]
    inicio = time.perf_counter() 

    l = sorted(l, reverse=True) 

    heap = [0] * m  # crea una lista de ceros de tamaño m (carga de trabajo en cada procesador)

    for p in l:
        load = heapq.heappop(heap)  # selecciona el procesador con menos cargado 
        load += p # añade tarea nueva al procesador menos cargado
        heapq.heappush(heap, load) # creo min-heap con los dato nuevos

    Makespan = max(heap)

    fin = time.perf_counter() 
 
    makespans.append([Makespan])
    tiempo.append([fin - inicio])
  
  return np.array(makespans), np.array(tiempo)

# ----------------- SOLUCIÓN CON GREEDY 2 -----------------
def solucion_greedy_2(m, N, L):
  makespans = []
  tiempo = []
  for aux in range(len(N)):
    l = L[aux]
    inicio = time.perf_counter() 

    heap = [0] * m  # crea una lista de ceros de tamaño m (carga de trabajo en cada procesador)

    for p in l:
        load = heapq.heappop(heap)  # selecciona el procesador con menos cargado 
        load += p # añade tarea nueva al procesador menos cargado
        heapq.heappush(heap, load) # creo min-heap con los dato nuevos

    Makespan = max(heap)

    fin = time.perf_counter() 
  
    # Gudar soluciones
    makespans.append([Makespan])
    tiempo.append([fin - inicio])
  
  return np.array(makespans), np.array(tiempo)

# ----------------- PROMEDIO E INTERVALO DE CONFIANZA -----------------
def mediaIntervaloConfianza(experimentos):
  varStat = []
  for arr_10 in experimentos:
    aux = []
    n = len(arr_10)
    media = np.mean(arr_10)
    aux.append(media)

    s = np.std(arr_10, ddof=1)

    SE = s/np.sqrt(n)

    t_crit = t.ppf(0.975, df=n-1)

    inferior = media - t_crit * SE
    aux.append(inferior)
    superior = media + t_crit * SE
    aux.append(superior)

    varStat.append(aux)

  return np.array(varStat)

# ----------------- GRAFICA TIEMPOS DE EJECUCIÓN -----------------
def grafica(n, varStat,leyenda,ax,color,columna):
  media = []
  lb = []
  up = []
  for i in range(len(varStat)):
    aux = varStat[i]
    media.append(aux[0])
    lb.append(aux[1])
    up.append(aux[2])


  ax[columna].errorbar(n, media,
              yerr=[np.array(media) - np.array(lb),
                    np.array(up) - np.array(media)],  # [abajo, arriba]
              label=leyenda,
              fmt='-o',               # línea + círculos
              linewidth=2, markersize=8,
              capsize=4,              # "gorritos" en los extremos
              capthick=1.5,
              alpha=0.9,
              color=color) 
  

# ----------------- CÓDIGO PRINCIPAL -----------------
N = [50, 100, 200, 400] # vector con número de tareas
m = 10 # 10 procesadores

# Limpia pantalla (Windows / Linux)
os.system("cls" if os.name == "nt" else "clear")

# Colores ANSI
C_RESET = "\033[0m"
C_BLUE  = "\033[94m"
C_CYAN  = "\033[96m"
C_WHITE = "\033[97m"
C_BOLD  = "\033[1m"
C_RED   = "\033[91m"
C_UNDER = "\033[4m"

print(f"""
{C_CYAN}{C_BOLD}
╔══════════════════════════════════════════════════════╗
║                                                      ║
║           SIMULACIÓN DE PLANIFICACIÓN P||Cmax        ║
║                                                      ║
║      Comparación: Gurobi vs Algoritmos Greedy        ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
{C_RESET}
{C_WHITE}Generador de instancias para pruebas experimentales
Duraciones de tareas: seleccione la distribución{C_RESET}

{C_BLUE}    [0]{C_RESET}  Distribución Normal
{C_BLUE}    [1]{C_RESET}  Distribución Lognormal
""")

while True:
  dist = input(f"{C_BOLD}Seleccione una opción (0/1): {C_RESET}")
  if dist == '0' or dist == '1':
    break
  else:
    print(f"{C_RED}{C_BOLD}{C_UNDER}ERROR: OPCIÓN NO VÁLIDA !!!{C_RESET}")

makespans_gurobi     = [[] for _ in N]
makespans_greedy_1_5 = [[] for _ in N]
makespans_greedy_2   = [[] for _ in N]

tiempos_gurobi       = [[] for _ in N]
tiempos_greedy_1_5   = [[] for _ in N]
tiempos_greedy_2     = [[] for _ in N]

# Encontrar solución para 10 instancias
for _ in range(10):
  L = generador_instancias(dist, N)
  tmp_makespans_gurobi, tmp_tiempos_gurobi = solucion_gurobi(m, N, L)
  makespans_gurobi = np.hstack((makespans_gurobi, tmp_makespans_gurobi))
  tiempos_gurobi   = np.hstack((tiempos_gurobi, tmp_tiempos_gurobi))

  tmp_makespans_greedy_1_5, tmp_tiempos_greedy_1_5 = solucion_greedy_1_5(m, N, L)
  makespans_greedy_1_5 = np.hstack((makespans_greedy_1_5, tmp_makespans_greedy_1_5))
  tiempos_greedy_1_5   = np.hstack((tiempos_greedy_1_5, tmp_tiempos_greedy_1_5))

  tmp_makespans_greedy_2, tmp_tiempos_greedy_2 = solucion_greedy_2(m, N, L)
  makespans_greedy_2  = np.hstack((makespans_greedy_2, tmp_makespans_greedy_2))
  tiempos_greedy_2    = np.hstack((tiempos_greedy_2, tmp_tiempos_greedy_2))

# LLamada a función para calcular promedio e IC de los makespans
varStat_gurobi_makespans      = mediaIntervaloConfianza(makespans_gurobi/makespans_gurobi)
varStat_greedy_1_5_makespans  = mediaIntervaloConfianza(makespans_greedy_1_5/makespans_gurobi)
varStat_greedy_2_makespans    = mediaIntervaloConfianza(makespans_greedy_2/makespans_gurobi)

# LLamada a función para calcular promedio e IC de los tiempos de ejecución
varStat_gurobi_tiempos     = mediaIntervaloConfianza(tiempos_gurobi)
varStat_greedy_1_5_tiempos = mediaIntervaloConfianza(tiempos_greedy_1_5)
varStat_greedy_2_tiempos   = mediaIntervaloConfianza(tiempos_greedy_2)

# LLamada a función para gráficar Makespan normalizado
fig1, ax1 =plt.subplots(1, 2, figsize=(10, 8), num="Comparación: Gurobi vs Greedy") 
grafica(N, varStat_gurobi_makespans,     "Gurobi", ax1, color = "blue", columna = 0)
grafica(N, varStat_greedy_1_5_makespans, "Greedy 1.5 aprox", ax1, color="orange", columna = 0)
grafica(N, varStat_greedy_2_makespans,   "Greedy 2 aprox", ax1, color="green", columna = 0)
ax1[0].set_title("n vs Makespan") # Título
ax1[0].set_xlabel("n (número de tareas)") # Nombre del eje X
ax1[0].set_ylabel("Makespan normalizado") # Nombre del eje y
ax1[0].legend()  # Muestra la leyenda
ax1[0].grid(True)
grafica(N, varStat_gurobi_makespans,     "Gurobi", ax1, color = "blue", columna = 1)
grafica(N, varStat_greedy_1_5_makespans, "Greedy 1.5 aprox", ax1, color="orange", columna = 1)
ax1[1].set_title("n vs Makespan") # Título
ax1[1].set_xlabel("n (número de tareas)") # Nombre del eje X
ax1[1].set_ylabel("Makespan normalizado") # Nombre del eje y
ax1[1].legend()  # Muestra la leyenda
ax1[1].grid(True)

# LLamada a función para gráficar tiempos de ejecución
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 8), num="Comparación: Tiempo Gurobi vs Tiempo Greedy")
grafica(N, varStat_gurobi_tiempos*1000.0,     "Gurobi", ax2, color = "blue", columna = 0)
grafica(N, varStat_greedy_1_5_tiempos*1000.0, "Greedy 1.5 aprox", ax2, color="orange", columna = 0)
grafica(N, varStat_greedy_2_tiempos*1000.0,   "Greedy 2 aprox", ax2, color="green", columna = 0)
ax2[0].set_title("n vs Tiempo") # Título
ax2[0].set_xlabel("n (número de tareas)") # Nombre del eje X
ax2[0].set_ylabel("tiempo (milisegundos)") # Nombre del eje y
ax2[0].legend()
ax2[0].grid(True)
grafica(N, varStat_greedy_1_5_tiempos*1000.0, "Greedy 1.5 aprox", ax2, color="orange", columna = 1)
grafica(N, varStat_greedy_2_tiempos*1000.0,   "Greedy 2 aprox", ax2, color="green", columna = 1)
ax2[1].set_title("n vs Tiempo") # Título
ax2[1].set_xlabel("n (número de tareas)") # Nombre del eje X
ax2[1].set_ylabel("tiempo (milisegundos)") # Nombre del eje y
ax2[1].legend()
ax2[1].grid(True)

plt.tight_layout()
plt.show()