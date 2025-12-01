import gurobipy as gp
from gurobipy import GRB

n = 5 # número de tareas
l = [2, 4, 6, 3, 5] # duración de cada tarea
m = 2 # número de procesadores
a = [0, 1, 3, 2, 5]        # tiempos de llegada
big_M = 1000                   # big-M

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
print(f"\nMakespan óptimo = {Makespan.X}")

for i in range(m):
    tareas = [j+1 for j in range(n) if x[i,j].X > 0.5]
    print(f"Máquina {i+1}: {tareas}")
    for j in tareas:
        print(f"  Inicio tarea {j}: {s[j-1].X}")