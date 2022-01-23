function eratosthenes(n::Integer)
  prime = trues(n) 
  prime[1] = false
  for i in 2:floor(Integer, sqrt(n))
    if prime[i]
      prime[i*i:i:n] = false
    end
  end
  return prime
end
