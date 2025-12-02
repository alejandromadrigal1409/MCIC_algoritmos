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

def generador_instancias(dist, n):
  np.random.seed(69)
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
  #print("\nDuración de tareas\n")
  l = []
  for i in n:
   l.append(np.random.normal(loc = mu, scale = sigma, size = i))

  return l 
  #print(d)

# ----------------- DISTRIBUCIÓN LOGNORMAL -----------------
def distribucion_lognormal(mu, sigma, n):
  #print("\nTiempo de llegada de tareas\n")
  l = []
  for i in n:
   l.append(np.random.lognormal(mean = mu, sigma = sigma, size = i))

  return l
  #print(a)

# ----------------- SOLUCIÓN CON GUROBI -----------------
def solucion_gurobi(m, N, L):
  print("SOLUCIÓN GUROBI")
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
    print("="*150)
    print(f"\nSolución para {n} tareas")
    print(f"\nMakespan óptimo = {Makespan.X}")

# ----------------- SOLUCIÓN CON GREEDY 1.5 -----------------
def solucion_greedy_1_5(m, N, L):
  print("\n\nSOLUCIÓN GREEDY 1.5")
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
    print("="*150)
    print(f"\nSolución para {n} tareas")
    print(f"\nMakespan óptimo = {Makespan}")

# ----------------- SOLUCIÓN CON GREEDY 2 -----------------
def solucion_greedy_2(m, N, L):
  print("\n\nSOLUCIÓN GREEDY 2")
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
    print("="*150)
    print(f"\nSolución para {n} tareas")
    print(f"\nMakespan óptimo = {Makespan}")

# ----------------- CÓDIGO PRINCIPAL -----------------
N = [50, 100, 200, 400] # vector con número de tareas
m = 10 # 10 procesadores

print("Distribución normal    (0)")
print("Distribución lognormal (1)")
dist = input("Selecione una distribución para generar las duraciónde las tareas: ")
L = generador_instancias(dist, N)
solucion_gurobi(m, N, L)
solucion_greedy_1_5(m, N, L)
solucion_greedy_2(m, N, L)





