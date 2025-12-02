
import gurobipy as gp
from gurobipy import GRB
import numpy as np

n = 7 # número de tareas
l = [4,2,6,5,9,4,7] # duración de cada tarea
m = 3 # número de procesadores

mu_nom = 3
sigma_nom = 1
mu_lognom = 1
sigma_lognom = 0.4
#n = 100

# FIJAR LA SEMILLA PARA REPRODUCIBILIDAD
#np.random.seed(42)
#l = np.random.normal(loc = mu_nom, scale = sigma_nom, size = n)
#l = np.random.lognormal(mean = mu_lognom, sigma = sigma_lognom, size = n)

print(l)
print("n =", n)
print("len(l) =", len(l))
print("max =", np.max(l))
print("min =", np.min(l))
print("sum =", np.sum(l))

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
print(f"\nMakespan óptimo = {Makespan.X}")

'''
for i in range(m):
    tareas = [j+1 for j in range(n) if x[i,j].X > 0.5]
    print(f"Máquina {i+1}: {tareas}")
'''