using Dates

function eratosthenes(n::Integer)
  prime = trues(n) 
  prime[1] = false
  for i in 2:floor(Integer, sqrt(n))
    if prime[i]
      prime[i*i:i:n] .= false
    end
  end
  return prime
end

eratosthenes(10) # warmup
for i in [3, 4, 5, 6, 7, 8, 9, 9.5]
  n = trunc(Int, 10^i)
  t = now()
  eratosthenes(n)
  println("handled 10^", i, " in ", now() - t)
end
