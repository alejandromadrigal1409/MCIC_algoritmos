# NOMECLATURA:
# n = número de tareas
# k = número de procesadores
# d[] = duración de tareas
# s[] = tiempo de llegada de las tareas

# CONSIDERACIONES:
# probar n = 50, 100, 200, 400
# k = 10 para todos los casos

# Para distribución exponencial: parametró importante "lambda" = número promedio de eventos por unidad de tiempo

import numpy as np

def generador_instancias(n):
  np.random.seed(47)
  mu_nom = 3
  sigma_nom = 1
  mu_lognom = 0
  sigma_lognom = 1

  distribucion_normal(mu_nom, sigma_nom, n)
  distribucion_lognormal(mu_lognom, sigma_lognom, n)

# Distribución para duración de tareas
def distribucion_normal(mu, sigma, n):
  print("\nDuración de tareas\n")
  d = []
  for i in n:
   d.append(np.random.normal(loc = mu, scale = sigma, size = i))
  print(d)

# Distribución para llegada de tareas
def distribucion_lognormal(mu, sigma, n):
  print("\nTiempo de llegada de tareas\n")
  s = []
  for i in n:
   s.append(np.random.lognormal(mean = mu, sigma = sigma, size = i))
  print(s)

generador_instancias([50, 100, 200, 400])

