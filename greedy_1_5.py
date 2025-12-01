import heapq

l = [4,2,6,5,9,4,7]
m = 3
M = [0] * 3

l.sort(reverse=True)

heap = [0] * 3  # crea una lista de ceros de tamaño m (carga de trabajo en cada procesador)
#heapq.heapify(heap) # crea un min-heap 

for p in l:
    load = heapq.heappop(heap)  # selecciona el procesador con menos cargado 
    load += p # añade tarea nueva al procesador menos cargado
    heapq.heappush(heap, load) # creo min-heap con los dato nuevos

Makespan = max(heap)
print(Makespan)

'''
for i in l:
    j = M.index(min(M))
    M[j] += i

print(M)
Makespan = max(M)    
print(Makespan)
'''
