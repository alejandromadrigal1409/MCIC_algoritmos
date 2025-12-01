# NOMECLATURA:
# n = número de tareas
# m = número de procesadores
# l[] = duración de tareas
# a[] = tiempo de llegada de las tareas

# CONSIDERACIONES:
# probar n = 50, 100, 200, 400
# m = 10 procesadores para todos los casos

# Para distribución exponencial: parametró importante "lambda" = número promedio de eventos por unidad de tiempo

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def generador_instancias(n):
  np.random.seed(47)
  mu_nom = 3
  sigma_nom = 1
  mu_lognom = 0
  sigma_lognom = 1

  l = distribucion_normal(mu_nom, sigma_nom, n)
  a = distribucion_lognormal(mu_lognom, sigma_lognom, n)

  return l, a

# Distribución para duración de tareas
def distribucion_normal(mu, sigma, n):
  #print("\nDuración de tareas\n")
  l = []
  for i in n:
   l.append(np.random.normal(loc = mu, scale = sigma, size = i))

  return l 
  #print(d)

# Distribución para llegada de tareas
def distribucion_lognormal(mu, sigma, n):
  #print("\nTiempo de llegada de tareas\n")
  a = []
  for i in n:
   a.append(np.random.lognormal(mean = mu, sigma = sigma, size = i))

  return a
  #print(a)

# ----------------- SOLUCIÓN CON GUROBI -----------------
def solucion_gurobi(m, N, L, A):
  for aux in range(len(N)):
    n = N[aux]
    l = L[aux]
    a = A[aux]
    big_M = 1000000  # big-M

    model = gp.Model("P||Cmax")

    # Variables
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x") # 1 si la tarea j esta asignada al procesador i, 0 si esta en otro procesador
    s = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="s") # tiempo de inicio de la tarea j en le procesador i
    y = model.addVars(n, n, m, vtype=GRB.BINARY,name="y") # variable binaria que indica si la tarea j va antes que la tarea k en el procesador i
    Makespan = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")

    # Objetivo: minimizar Cmax
    model.setObjective(Makespan, GRB.MINIMIZE)

    # Restriccion: cada tarea debe estar en exactamente un procesador
    for j in range(n):
        model.addConstr(sum(x[i,j] for i in range(m)) == 1)

    # Restricción: no empezar una tarea antes de que llegue:
    for j in range(n):
        model.addConstr(s[j] >= a[j])

    # Restricción: respetar precedencia de tareas en el mismo procesador
    for i in range(m):
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                model.addConstr(s[j] + l[j] <= s[k] + big_M*(1 - y[j,k,i]) + big_M*(2 - x[i,j] - x[i,k]))
                model.addConstr(s[k] + l[k] <= s[j] + big_M*y[j,k,i] + big_M*(2 - x[i,j] - x[i,k]))

    # Restricción: minimizar el makespan:
    for j in range(n):
        model.addConstr(Makespan >= s[j] + l[j])

    model.optimize()

    # Mostrar solución
    print(f"="*100)
    print(f"MAKESPAN PARA: {n} TAREAS")
    print(f"\nMakespan óptimo = {Makespan.X}")

# ----------------- CÓDIGO PRINCIPAL -----------------
N = [50, 100, 200, 400] # vector con número de tareas
m = 10 # 10 procesadores
L, A = generador_instancias(N)
solucion_gurobi(m, N, L, A)


