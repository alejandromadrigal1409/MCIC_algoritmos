# NOMECLATURA:
# n = número de tareas
# k = número de procesadores
# t[] = duración de tareas

# CONSIDERACIONES:
# probar n = 50, 100, 200, 400
# k = 10 para todos los casos

# Para distribución exponencial: parametró importante "lambda" = número promedio de eventos por unidad de tiempo

import numpy as np

def generador_instancias(n):
  np.random.seed(47)
  mu_nom = 0
  sigma_nom = 1
  mu_lognom = 0
  sigma_lognom = 1

  distribucion_normal(mu_nom, sigma_nom, n)
  distribucion_lognormal(mu_lognom, sigma_lognom, n)

def distribucion_normal(mu, sigma, n):
  print("\nDistribución Normal\n")
  t = []
  for i in n:
   t.append(np.random.normal(mu, sigma, i))
  print(t)

def distribucion_lognormal(mu, sigma, n):
  print("\nDistribución LogNormal\n")
  t = []
  for i in n:
   t.append(np.random.lognormal(mu, sigma, i))
  print(t)
  #if flag == 1:
  #t = np.random.exponential(scale = 1/lam, size = n)
  #print(t)
  #u = np.random.random(size = n)
  #t = (-1/lam)*np.log(1 - u)
  #print(t)

generador_instancias([50, 100, 200, 400])
