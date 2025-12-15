import heapq
l = [4,2,6,5,9,4,7]
m = 3

heap = [0] * m  # crea una lista de ceros de tamaño m (carga de trabajo en cada procesador)

for p in l:
    load = heapq.heappop(heap)  # selecciona el procesador con menos cargado 
    load += p # añade tarea nueva al procesador menos cargado
    heapq.heappush(heap, load) # creo min-heap con los dato nuevos

Makespan = max(heap)
print(Makespan)