import numpy as np
import time

def eratosthenes(n):
  prime = np.ones(n, dtype=np.bool_)
  prime[0] = False
  prime[1] = False
  for i in range(2, int(np.sqrt(n)) + 1):
    if prime[i]:
      prime[i*i:n:i] = False
  return prime

for i in [3, 4, 5, 6, 7, 8, 9, 9.5]:
  n = int(10**i)
  t = time.time()
  primes = eratosthenes(n)
  print(f'handled 10^{i} in {time.time() - t}')
