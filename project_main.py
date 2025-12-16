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

    model.setParam('MIPGap', 0.001)
    #model.setParam('TimeLimit', 2)
    model.optimize()

    fin = time.perf_counter() 

    makespans.append(Makespan.X)
    tiempo.append(fin - inicio)
  
  return makespans, tiempo

# ----------------- SOLUCIÓN CON GREEDY 1.5 -----------------
def solucion_greedy_1_5(m, N, L):
  makespans = []
  tiempo = []
  for aux in range(len(N)):
    l = L[aux]
    inicio = time.perf_counter() 

    l = sorted(l, reverse=True) 

    Makespan, _ = solucion_greedy_2(m, N, [l])

    fin = time.perf_counter() 
 
    makespans.append(Makespan[0])
    tiempo.append(fin - inicio)
  
  return makespans, tiempo

# ----------------- SOLUCIÓN CON GREEDY 2 -----------------
def solucion_greedy_2(m, N, L):
  makespans = []
  tiempo = []
  for aux in range(len(L)):
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
    makespans.append(Makespan)
    tiempo.append(fin - inicio)
  
  return makespans, tiempo

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
#N = [50, 100, 200, 400] # vector con número de tareas
N = [400]
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
  #L = generador_instancias(dist, N)
  L = [np.array([
    4.47833085, 2.82736289, 1.95955016, 2.47228319, 3.68787147, 3.46332444,
    2.26041637, 3.55992451, 3.33824051, 0.87464342, 1.14353696, 1.38070934,
    3.58867799, 1.72323988, 3.45200345, 2.78868813, 5.61752534, 2.7912481,
    2.03341717, 4.05064669, 3.8195104, 2.25787326, 2.68581894, 4.49867244,
    1.6222624, 2.70364749, 3.59719818, 2.89854941, 2.51073075, 4.10273184,
    4.22193261, 2.22316781, 2.51305207, 1.97712085, 2.6869786, 1.98455507,
    3.07334644, 2.51069879, 4.55658667, 4.82183137, 3.99351282, 1.30468567,
    2.51460263, 4.53415117, 3.92256712, 3.05335777, 2.2373804, 1.49986335,
    3.11649695, 3.44140805, 4.95316741, 2.90326164, 2.20466725, 3.03496348,
    2.31147106, 4.73066413, 3.96234472, 1.9490104, 1.45416646, 4.32402828,
    4.08411641, 2.08646018, 2.42938524, 3.71504533, 2.71014337, 1.34611102,
    3.1856131, 3.07038829, 4.40801618, 2.02481258, 3.3916333, 3.43925013,
    3.13269672, 4.91260962, 4.36763443, 3.60029409, 2.09379999, 2.74005229,
    3.42756896, 2.65827541, 3.47721793, 2.73734586, 2.67126989, 3.78862609,
    2.02257251, 4.75974908, 2.4418854, 1.85530609, 4.63984297, 1.39279024,
    2.61628172, 1.85438887, 3.00608694, 5.3803998, 2.56028965, 3.83812072,
    4.16619363, 3.86047016, 4.83507795, 2.63619985, 1.84880736, 3.3829716,
    2.8452205, 2.48388161, 2.37084173, 0.80514436, 3.69443084, 3.28330012,
    2.34534602, 1.31471807, 3.01018655, 3.70308901, 2.92389319, 2.84274098,
    2.71393436, 3.5125707, 3.0160241, 2.61278695, 3.46869506, 2.29903102,
    4.12319126, 2.67825813, 3.63703729, 5.2872531, 2.67338178, 1.55615155,
    2.93056583, 5.12461786, 3.13615959, 3.23659842, 1.9035839, 4.75483419,
    3.38761242, 2.43816348, 2.41469405, 3.55601015, 3.06487495, 2.95264728,
    3.1683085, 3.03530111, 3.78140946, 3.63041503, 4.64150259, 3.41857367,
    1.79994012, 2.71265855, 3.69321391, 3.1678184, 2.41391597, 3.72905326,
    3.69692265, 2.52708201, 3.63705779, 1.34722957, 4.11616534, 1.65307815,
    2.39392475, 3.11100984, 4.78296427, 2.01834629, 2.91793023, 2.30925635,
    2.43520055, 3.76529375, 2.43076456, 2.5284917, 2.50255521, 2.19542626,
    4.24964723, 1.02328609, 2.92221586, 4.41745735, 2.21282969, 3.5551786,
    2.13549175, 3.58194422, 2.77699924, 1.22880708, 2.31866044, 2.81184425,
    3.76458367, 3.43666273, 3.42329925, 2.37492216, 2.532965, 2.97234241,
    2.385681, 2.56311399, 2.97109314, 4.29313323, 4.45821779, 4.20234842,
    1.91169106, 3.19607778, 2.13125833, 1.56061465, 2.6787853, 1.53674079,
    2.80264148, 1.73517499, 3.7485706, 1.90831413, 3.39642377, 4.32678901,
    1.76868609, 1.43594213, 2.15523956, 2.09749326, 2.47482414, 1.5581304,
    3.09558393, 3.55667294, 3.68602025, 2.10687594, 2.79009595, 4.27024737,
    4.06534236, 1.94239957, 3.67647051, 2.42051012, 2.29098489, 3.12656137,
    5.12938346, 2.89651525, 1.04220929, 0.71668963, -0.08306619, 0.64969786,
    1.23260156, 2.15063735, 5.35197222, 4.72149591, 2.55221049, 2.98387116,
    3.29726347, 2.94588692, 4.66175164, 2.47908835, 1.87833411, 2.6082433,
    2.9247147, 4.43811371, 0.69722606, 2.36229142, 3.83611662, 3.61387172,
    2.6167932, 3.83189654, 3.19805909, 1.18755867, 2.80698334, 4.23584271,
    3.72636597, 2.28715973, 3.09869773, 1.59608462, 4.34623478, 3.43348659,
    4.23130848, 1.46039276, 4.24631529, 1.86205161, 4.99015939, 3.46243794,
    2.64160337, 3.46165135, 3.81733109, 1.71023411, 1.960318, 3.6825412,
    1.84640018, 3.32629357, 0.34547026, 2.63431342, 4.23270137, 4.25096779,
    3.13527428, 1.8139081, 2.09749283, 3.09630615, 3.17580313, 1.54442138,
    2.96117992, 1.84708129, 2.60179872, 2.67524012, 2.32773612, 2.73630218,
    4.39702435, 2.13509416, 3.21800944, 1.80223595, 3.20166457, 1.7115003,
    4.34384877, 3.00440302, 4.20422938, 5.08982732, 1.77271301, 1.31590081,
    3.47155, 3.33296263, 4.1107572, 2.29115304, 4.05319939, 2.7449447,
    3.55312711, 1.62937948, 3.36474595, 4.99583209, 1.87487703, 3.77529762,
    2.97803459, 2.94162232, 1.81858418, 1.99391821, 4.63908131, 3.8924985,
    3.19982699, 3.99528752, 2.77102923, 3.94078099, 4.35848314, 3.41373193,
    2.43765903, 4.15927699, 1.38004316, 3.71300194, 3.46040579, 3.18887148,
    0.91802386, 2.41643355, 3.19202204, 2.1531811, 4.50806987, 3.40738025,
    2.51174648, 2.17584478, 2.72217009, 3.03667552, 0.75794208, 2.32525257,
    4.28702542, 3.70303605, 3.32402739, 2.57163649, 3.9314814, 1.51336774,
    3.16478844, 1.71997321, 2.70881456, 3.73765128, 3.44434981, 4.10222591,
    3.22857511, 2.1949212, 4.47325768, 2.95845896, 4.6374149, 3.51360061,
    1.17890904, 3.24354659, 0.40604744, 2.93337121, 3.02805047, 4.13561359,
    3.31839562, 3.71161027, 3.37004731, 2.21672367, 2.86238258, 2.47813427,
    3.1796092, 2.37535821, 2.66197412, 5.13714939, 2.75639568, 3.29987837,
    3.98098968, 2.88562836, 2.15343518, 3.78385206, 2.78641956, 2.63221509,
    3.64676693, 2.37679109, 2.81490965, 1.54492168, 2.94938392, 3.17631713,
    2.31265892, 1.90912073, 3.10858531, 3.17530046, 2.54291323, 3.97225545,
    2.11575941, 4.89448809, 2.68508998, 1.80756179
])]
  tmp_makespans_gurobi, tmp_tiempos_gurobi = solucion_gurobi(m, N, L)
  tmp_makespans_greedy_1_5, tmp_tiempos_greedy_1_5 = solucion_greedy_1_5(m, N, L)
  tmp_makespans_greedy_2, tmp_tiempos_greedy_2 = solucion_greedy_2(m, N, L)

  for i in range(len(N)):
    makespans_gurobi[i].append(tmp_makespans_gurobi[i])
    makespans_greedy_1_5[i].append(tmp_makespans_greedy_1_5[i])
    makespans_greedy_2[i].append(tmp_makespans_greedy_2[i])

    tiempos_gurobi[i].append(tmp_tiempos_gurobi[i])
    tiempos_greedy_1_5[i].append(tmp_tiempos_greedy_1_5[i])
    tiempos_greedy_2[i].append(tmp_tiempos_greedy_2[i])


makespans_gurobi = np.array(makespans_gurobi)
makespans_greedy_1_5 = np.array(makespans_greedy_1_5)
makespans_greedy_2 = np.array(makespans_greedy_2)

tiempos_gurobi = np.array(tiempos_gurobi)
tiempos_greedy_1_5 = np.array(tiempos_greedy_1_5)
tiempos_greedy_2 = np.array(tiempos_greedy_2)

print(makespans_gurobi)
print()
print(makespans_greedy_1_5)
print()
print(makespans_greedy_2)

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
