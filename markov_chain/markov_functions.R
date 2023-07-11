library(expm)
library(matlib)


find.stationary = function(p) {
  if (length(p) == 1) {
    return(1)
  }
  n = nrow(p)
  A = p - diag(n)
  A[,n] = rep(1, n)
  pi = solve(A)[n,]
  return(pi)
}


check.dbc = function(p, pi=find.stationary(p)) {
  S = 1:nrow(p)
  for (x in S) {
    for (y in S) {
      lhs = pi[x]*p[x,y]
      rhs = pi[y]*p[y,x]
      equality = isTRUE(all.equal(lhs, rhs, tol=10^-6))
      if (!equality) {
        print(c(x,y))
        return(FALSE)
      }
    }
  } 
  return(TRUE)
}


limit.trans.mat = function(p, Tr, R.1, R.2) {
  # Tr is set (vector) of all transient states
  # R.k is a closed, irreducible set of (recurrent) states
  N = nrow(p)
  t = length(Tr)
  # p.inf[x,y] = lim p^n[x,y] as n -> inf. Initialized to zeros. 
  p.inf = matrix(rep(0,N^2), N, N)
  
  r = p[Tr, Tr]
  v.1 = p[Tr, R.1]
  if (length(R.1) > 1) {
    v.1 = rowSums(p[Tr, R.1])
  }
  
  r.inv = solve(diag(t) - r)
  
  h.1 = r.inv %*% v.1
  h.2 = 1 - h.1
  
  g = r.inv %*% rep(1,t)
  
  pi.1 = find.stationary(p[R.1, R.1])
  pi.2 = find.stationary(p[R.2, R.2])
  
  # restore indices as states
  h.1.full  = rep(0,N)
  h.2.full  = rep(0,N)
  pi.1.full = rep(0,N)
  pi.2.full = rep(0,N)
  for (i in 1:length(R.1)) {
    x = R.1[i]
    h.1.full[x] = 1
    pi.1.full[x] = pi.1[i]
  }
  for (i in 1:length(R.2)) {
    x = R.2[i]
    h.2.full[x] = 1
    pi.2.full[x] = pi.2[i]
  }
  for (i in 1:t) {
    x = Tr[i]
    h.1.full[x]  = h.1[i]
    h.2.full[x]  = h.2[i]
  }
  
  for (x in 1:N) {
    for (y in R.1) {
      p.inf[x,y] = h.1.full[x]*pi.1.full[y]
    }
    for (y in R.2) {
      p.inf[x,y] = h.2.full[x]*pi.2.full[y]
    }
  }
  
  return(list("p.inf"=p.inf,
              "r"=r, "v.1"=v.1,
              "h.1"=h.1, "h.2"=h.2,
              "g"=g,
              "pi.1"=pi.1, "pi.2"=pi.2))
}


find.stationary.CTMC = function(Q) {
  n = nrow(Q) # number of states
  b = c(rep(0, n-1),1)
  Q[,n] = rep(1, n)
  pi = Solve(t(Q), b, fractions=TRUE)
  return(pi)
}
