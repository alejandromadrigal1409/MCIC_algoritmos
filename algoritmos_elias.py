# ------------------------------------------------------------
#!/usr/bin/env python3

# @author:
# Este modulo contiene todos los algoritmos utilizados para realizar el proyecto.


import heapq
import gurobipy as gp
from gurobipy import GRB

# ------------------------------------------------------------
# Algoritmo Greedy 2-Aproximación
# ------------------------------------------------------------

def greedy_2aproximacion(l, m):
    """
    Recibe:
        l: lista de tareas (duraciones)
        m: número de procesadores
    
    Retorna:
        Makespan estimado por el algoritmo Greedy 2-aproximación.
    """
    # Min-heap con cargas por procesador
    heap = [0] * m

    for p in l:
        load = heapq.heappop(heap)   # procesador menos cargado
        load += p
        heapq.heappush(heap, load)

    return max(heap)


# ------------------------------------------------------------
# Algoritmo Greedy 1.5-Aproximación
# (Implementación idéntica a la anterior excepto por heurística)
# ------------------------------------------------------------

def greedy_1_5_aproximacion(l, m):
    """
    Misma estructura que el greedy anterior.
    """
    tareas = sorted(l, reverse=True)

    heap = [0] * m

    for p in tareas:
        load = heapq.heappop(heap)
        load += p
        heapq.heappush(heap, load)

    return max(heap)


# ------------------------------------------------------------
# Solución exacta con Gurobi
# ------------------------------------------------------------

def solucion_gurobi(l, m):
    """
    Resuelve P||Cmax usando MILP con Gurobi.

    Recibe:
        l: lista de duraciones de tareas
        m: número de máquinas

    Retorna:
        Makespan óptimo (float)
    """

    n = len(l)

    model = gp.Model("P||Cmax")
    model.setParam('OutputFlag', 0)  # apagar impresión de Gurobi
    #model.setParam('TimeLimit', 60)  # Maximo 60 segundos por problema
    model.setParam("MIPGap", 0.001)

    # Variables: x[i,j] = 1 si tarea j va a máquina i, 0 si no.
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")

    # Makespan, este representa el maximo de carga que tienen los procesadores.
    Cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")

    # Objetivo, la idea es que el makespan sea lo más pequeño posible.
    model.setObjective(Cmax, GRB.MINIMIZE)

    # Cada tarea debe asignarse a exactamente a un procesador.
    for j in range(n):
        model.addConstr(sum(x[i, j] for i in range(m)) == 1)

    # Carga de cada máquina <= Cmax
    for i in range(m):
        model.addConstr(sum(l[j] * x[i, j] for j in range(n)) <= Cmax)
    
    # Resolver
    model.optimize()

    return Cmax.X

def main():
    pass

if __name__ == "__main__":
    main()