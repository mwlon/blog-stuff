using Dates

const VALID_Y1 = Array[[3,5,7,13,15,17,23,25,27],[1,5,9,11,15,19,21,25,29],[1,5,11,19,25,29],[3,5,7,13,15,17,23,25,27],[1,3,7,9,11,13,17,19,21,23,27,29],[5,7,13,17,23,25],[1,5,9,11,15,19,21,25,29],[1,5,9,11,15,19,21,25,29],[5,7,13,17,23,25],[1,3,7,9,11,13,17,19,21,23,27,29],[3,5,7,13,15,17,23,25,27],[1,5,11,19,25,29],[1,5,9,11,15,19,21,25,29],[3,5,7,13,15,17,23,25,27],[1,7,11,13,17,19,23,29],[3,5,7,13,15,17,23,25,27],[1,5,9,11,15,19,21,25,29],[1,5,11,19,25,29],[3,5,7,13,15,17,23,25,27],[1,3,7,9,11,13,17,19,21,23,27,29],[5,7,13,17,23,25],[1,5,9,11,15,19,21,25,29],[1,5,9,11,15,19,21,25,29],[5,7,13,17,23,25],[1,3,7,9,11,13,17,19,21,23,27,29],[3,5,7,13,15,17,23,25,27],[1,5,11,19,25,29],[1,5,9,11,15,19,21,25,29],[3,5,7,13,15,17,23,25,27],[1,7,11,13,17,19,23,29]]
const VALID_Y2 = Array[[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,14,16,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,14,16,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,14,16,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[],[2,4,8,10,14,16,20,22,26,28],Int64[]]
const VALID_Y3 = Array[Int64[],[1],[2,4],[1,5],[2,4,8],[1,5,7],[2,4,8,10],[1,5,7,11,13],[2,4,8,10,14],[1,7,11,13,17],[2,4,8,10,14,16],[1,5,7,11,13,17,19],[2,4,8,10,14,16,20,22],[1,5,7,11,13,17,19,23],[2,4,8,14,16,22],[1,5,7,11,13,17,19,23,25],[2,4,8,10,14,16,20,22,26,28],[1,5,7,11,13,17,19,23,25,29],[2,4,8,10,14,16,20,22,26,28],[1,7,11,13,17,19,23,29],[2,4,8,10,14,16,20,22,26,28],[1,5,7,11,13,17,19,23,25,29],[2,4,8,10,14,16,20,22,26,28],[1,5,7,11,13,17,19,23,25,29],[2,4,8,14,16,22,26,28],[1,5,7,11,13,17,19,23,25,29],[2,4,8,10,14,16,20,22,26,28],[1,5,7,11,13,17,19,23,25,29],[2,4,8,10,14,16,20,22,26,28],[1,7,11,13,17,19,23,29]]

function atkin(n::Integer)
  prime = falses(n)
  sqn = sqrt(n)
  maxx1 = floor(Integer, sqn/sqrt(4.0))
  maxx2 = floor(Integer, sqn/sqrt(3.0))
  maxx3 = floor(Integer, sqn/sqrt(2.0))
  xrange = 1:maxx3
  xsq = xrange.^2
  modx = (xrange .- 1) .% 30 .+ 1
  for x in 1:maxx1
    xterm = 4*xsq[x]
    maxy = floor(Integer, sqrt(n - xterm))
    for j in VALID_Y1[modx[x]]
      mask = xterm .+ (j:30:maxy).^2
      prime[mask] .= .!prime[mask]
    end
  end
  for x in 1:2:maxx2
    xterm = 3*xsq[x]
    maxy = floor(Integer, sqrt(n - xterm))
    for j in VALID_Y2[modx[x]]
      mask = xterm .+ (j:30:maxy).^2
      prime[mask] .= .!prime[mask]
    end
  end
  for x in 1:maxx3
    xterm = 3*xsq[x]
    maxy = x-1
    min_anyy = ceil(Integer, sqrt(max(1, xterm - n)))
    for j in VALID_Y3[modx[x]]
      miny = cld(min_anyy - j, 30) * 30 + j
      mask = xterm .- (miny:30:maxy).^2
      prime[mask] .= .!prime[mask]
    end
  end
  for i in 7:floor(Integer, sqrt(n))
    if prime[i]
      iterm = i*i
      prime[iterm * 1:fld(n, iterm)] .= false
    end
  end
  prime[2] = true
  prime[3] = true
  prime[5] = true
  return prime
end

atkin(10) # warmup
for i in [3, 4, 5, 6, 7, 8, 9, 9.5]
  n = trunc(Int, 10^i)
  t = now()
  atkin(n)
  println("handled 10^", i, " in ", now() - t)
end
