import numpy as np
import time

def eratosthenes(n):
  prime = np.ones(n, dtype=np.bool_)
  prime[0] = False
  prime[1] = False
  for i in xrange(2, int(np.sqrt(n))):
    if prime[i]:
      prime[range(i*i, n, i)] = False
  return prime
