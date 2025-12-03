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
  #print(d)

# ----------------- DISTRIBUCIÓN LOGNORMAL -----------------
def distribucion_lognormal(mu, sigma, n):
  l = []
  for i in n:
   l.append(np.random.lognormal(mean = mu, sigma = sigma, size = i))

  return l
  #print(a)

# ----------------- SOLUCIÓN CON GUROBI -----------------
def solucion_gurobi(m, N, L):
  #print("SOLUCIÓN GUROBI")
  makespans = []
  for aux in range(len(N)):
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

    model.setParam('MIPGap', 0.001)
    model.optimize()

    # Mostrar solución
    #print("="*150)
    #print(f"\nSolución para {n} tareas")
    #print(f"\nMakespan óptimo = {Makespan.X}")
    # Gudar soluciones
    makespans.append([Makespan.X])
  
  return np.array(makespans)

# ----------------- SOLUCIÓN CON GREEDY 1.5 -----------------
def solucion_greedy_1_5(m, N, L):
  #print("\n\nSOLUCIÓN GREEDY 1.5")
  makespans = []
  for aux in range(len(N)):
    n = N[aux]
    l = L[aux]

    l = sorted(l, reverse=True) 

    heap = [0] * m  # crea una lista de ceros de tamaño m (carga de trabajo en cada procesador)
    #heapq.heapify(heap) # crea un min-heap 

    for p in l:
        load = heapq.heappop(heap)  # selecciona el procesador con menos cargado 
        load += p # añade tarea nueva al procesador menos cargado
        heapq.heappush(heap, load) # creo min-heap con los dato nuevos

    Makespan = max(heap)
    # Mostrar solución
    #print("="*150)
    #print(f"\nSolución para {n} tareas")
    #print(f"\nMakespan óptimo = {Makespan}")

    # Gudar soluciones
    makespans.append([Makespan])
  
  return np.array(makespans)

# ----------------- SOLUCIÓN CON GREEDY 2 -----------------
def solucion_greedy_2(m, N, L):
  #print("\n\nSOLUCIÓN GREEDY 2")
  makespans = []
  for aux in range(len(N)):
    n = N[aux]
    l = L[aux]

    heap = [0] * m  # crea una lista de ceros de tamaño m (carga de trabajo en cada procesador)

    for p in l:
        load = heapq.heappop(heap)  # selecciona el procesador con menos cargado 
        load += p # añade tarea nueva al procesador menos cargado
        heapq.heappush(heap, load) # creo min-heap con los dato nuevos

    Makespan = max(heap)
    # Mostrar solución
    #print("="*150)
    ##print(f"\nMakespan óptimo = {Makespan}")
  
    # Gudar soluciones
    makespans.append([Makespan])
  
  return np.array(makespans)

# ----------------- PROMEDIO E INTERVALO DE CONFIANZA -----------------
def mediaIntervaloConfianza(makespans):
  varStat = []
  for arr_10 in makespans:
    aux = []
    n = len(arr_10)
    media = np.mean(arr_10)
    aux.append(media)

    s = s = np.std(arr_10, ddof=1)

    SE = s/np.sqrt(n)

    t_crit = t.ppf(0.975, df=n-1)

    inferior = media - t_crit * SE
    aux.append(inferior)
    superior = media + t_crit * SE
    aux.append(superior)

    varStat.append(aux)

  return np.array(varStat)

# ----------------- GRAFICA MAKESPAN NORMALIZADO -----------------
def grafMakespanNorm(n, varStat, varStat_gurobi,leyenda):
  #
  makespan = []
  lb = []
  up = []
  for i in range(len(varStat)):
    z = varStat_gurobi[i][0]
    aux = varStat[i]/z #normalizar datos
    makespan.append(aux[0])
    lb.append(aux[1])
    up.append(aux[2])

  plt.errorbar(n, makespan,
             yerr=[np.array(makespan) - np.array(lb),
                   np.array(up) - np.array(makespan)],  # [abajo, arriba]
             label=leyenda,
             fmt='-o',               # línea + círculos
             linewidth=2, markersize=8,
             capsize=4,              # "gorritos" en los extremos
             capthick=1.5,
             alpha=0.9)

# ----------------- CÓDIGO PRINCIPAL -----------------
N = [50, 100, 200, 400] # vector con número de tareas
#N = [10, 20, 40, 80]
m = 10 # 10 procesadores

print("Distribución normal    (0)")
print("Distribución lognormal (1)")
dist = input("Selecione una distribución para generar las duraciónde las tareas: ")

makespans_gurobi = [[]] * len(N)
makespans_greedy_1_5 = [[]] * len(N)
makespans_greedy_2 = [[]] * len(N)

# Encontrar solución para 10 instancias
for _ in range(10):
  L = generador_instancias(dist, N)
  makespans_gurobi = np.hstack((makespans_gurobi,solucion_gurobi(m, N, L)) )
  makespans_greedy_1_5 = np.hstack((makespans_greedy_1_5,solucion_greedy_1_5(m, N, L)))
  makespans_greedy_2 = np.hstack((makespans_greedy_2,solucion_greedy_2(m, N, L)))

# LLamada a función para calcular promedio e IC
varStat_gurobi = mediaIntervaloConfianza(makespans_gurobi)
varStat_greedy_1_5 = mediaIntervaloConfianza(makespans_greedy_1_5)
varStat_greedy_2 = mediaIntervaloConfianza(makespans_greedy_2)

# LLamada a función para gráficar
plt.figure(figsize=(8, 5), num="Comparación: Gurobi vs Greedy") 
grafMakespanNorm(N, varStat_gurobi,     varStat_gurobi, "Gurobi")
grafMakespanNorm(N, varStat_greedy_1_5, varStat_gurobi, "Greedy 1.5 aprox")
grafMakespanNorm(N, varStat_greedy_2, varStat_gurobi,   "Greedy 2 aprox")


plt.title("n vs Makespan") # Título
plt.xlabel("n (número de tareas)") # Nombre del eje X
plt.ylabel("Makespan normalizado") # Nombre del eje y
plt.ylim(0, None)
plt.legend()  # Muestra la leyenda
plt.grid(True)
plt.show()








