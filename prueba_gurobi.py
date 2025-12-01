import gurobipy as gp
from gurobipy import GRB
import numpy as np

n = 5 # número de tareas
l = [2, 4, 6, 3, 5] # duración de cada tarea
m = 2 # número de procesadores

mu_nom = 3
sigma_nom = 1
#n = 400
#l = np.random.normal(loc = mu_nom, scale = sigma_nom, size = n)

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

model.optimize()

# Mostrar solución
print(f"\nMakespan óptimo = {Makespan.X}")


for i in range(m):
    tareas = [j+1 for j in range(n) if x[i,j].X > 0.5]
    print(f"Máquina {i+1}: {tareas}")
