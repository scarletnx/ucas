# absalg.thy - the proof library: definitions, the arithmetic axioms of
# Z, and lemmas proved from them (kscript.c runs it; every lemma ends in
# kernel rules).  The only axioms are the "def" and "arith" lines.
#
# Model: one ambient abelian group (sort el) holds every group as a
# subgroup (sort set, membership "in"); homomorphisms are symbols marked
# hom (additive and Z-linear: the kernel's conversion uses that).  Any
# diagram of abelian groups embeds this way (in the direct sum of its
# groups), so nothing is lost.

sort set cpl cx

# ---- integers: the ordered ring Z ----
arith int_discrete: forall n:int: n <= 0 or 1 <= n
arith le_antisym: forall a:int: 0 <= a and 0 <= -a -> a = 0
arith mul_nonneg: forall a b:int: 0 <= a and 0 <= b -> 0 <= a*b
arith int_domain: forall a b:int: a*b = 0 -> a = 0 or b = 0
arith pow_zero: forall a:int: a^0 = 1
arith pow_succ: forall a m:int: 0 <= m -> a^(m+1) = a^m*a

lemma pow_pos: forall p:int: 1 <= p -> forall m:int: 0 <= m -> 1 <= p^m
proof
  fix p
  intro hp
  induct m from 0
  base
  proof
    show 1 <= p^0 by pow_zero[p]
  qed
  step
  proof
    fix m
    intro hm ih
    have e: p^(m+1) = p^m*p by pow_succ[p, m] hm
    show 1 <= p^(m+1) by e ih hp
  qed
qed

# ---- subgroups ----
pred subgroup(set)
def subgroup_def: forall S:set: subgroup(S) <-> (0 in S and (forall x y:el: x in S and y in S -> x + y in S) and (forall x:el: x in S -> -x in S))

lemma sub_zero: forall S:set: subgroup(S) -> 0 in S
by subgroup_def
lemma sub_add: forall S:set, x y:el: subgroup(S) and x in S and y in S -> x + y in S
by subgroup_def
lemma sub_neg: forall S:set, x:el: subgroup(S) and x in S -> -x in S
by subgroup_def

lemma sub_natmul: forall S:set, x:el: subgroup(S) and x in S -> forall n:int: 0 <= n -> n*x in S
proof
  fix S x
  intro hs hx
  induct n from 0
  base
  proof
    show 0*x in S by sub_zero hs
  qed
  step
  proof
    fix m
    intro hm ih
    show (m+1)*x in S by sub_add[S, m*x, x] hs ih hx
  qed
qed

lemma sub_smul: forall S:set, n:int, x:el: subgroup(S) and x in S -> n*x in S
proof
  fix S n x
  intro hs hx
  have nat: forall m:int: 0 <= m -> m*x in S by sub_natmul hs hx
  cases hc: n <= 0 or 1 <= n by int_discrete[n]
  case h1
  proof
    have a: (-n)*x in S by nat h1
    show n*x in S by sub_neg[S, (-n)*x] hs a
  qed
  case h2
  proof
    show n*x in S by nat h2
  qed
qed

# the sum of two subgroups (a set by comprehension)
func ssum(set, set): set
def ssum_def: forall A B:set, x:el: x in ssum(A,B) <-> (exists a b:el: a in A and b in B and x = a + b)

lemma ssum_sub: forall A B:set: subgroup(A) and subgroup(B) -> subgroup(ssum(A,B))
proof
  fix A B
  intro ha hb
  have a0: 0 in A by sub_zero ha
  have b0: 0 in B by sub_zero hb
  have z: 0 in ssum(A,B) by ssum_def a0 b0 with 0, 0
  have ad: forall x y:el: x in ssum(A,B) and y in ssum(A,B) -> x + y in ssum(A,B)
  proof
    fix x y
    intro hx hy
    obtain a1 b1:el st a1A b1B e1: a1 in A and b1 in B and x = a1 + b1 by ssum_def hx
    obtain a2 b2:el st a2A b2B e2: a2 in A and b2 in B and y = a2 + b2 by ssum_def hy
    have aa: a1 + a2 in A by ha a1A a2A
    have bb: b1 + b2 in B by hb b1B b2B
    show x + y in ssum(A,B) by ssum_def aa bb e1 e2 with a1 + a2, b1 + b2
  qed
  have ng: forall x:el: x in ssum(A,B) -> -x in ssum(A,B)
  proof
    fix x
    intro hx
    obtain a1 b1:el st a1A b1B e1: a1 in A and b1 in B and x = a1 + b1 by ssum_def hx
    have aa: -a1 in A by ha a1A
    have bb: -b1 in B by hb b1B
    show -x in ssum(A,B) by ssum_def aa bb e1 with -a1, -b1
  qed
  show subgroup(ssum(A,B)) by subgroup_def z ad ng
qed

# ---- direct summands ----
pred summand(set, set)
pred zsummand(el, set)
pred zpksummand(el, set, int, int)

# S is a direct summand of T: T = S (+) M
def summand_def: forall S T:set: summand(S,T) <-> (exists M:set: subgroup(S) and subgroup(T) and subgroup(M) and (forall x:el: x in S -> x in T) and (forall x:el: x in M -> x in T) and (forall x:el: x in T -> exists s m:el: s in S and m in M and x = s + m) and (forall s m:el: s in S and m in M and s + m = 0 -> s = 0))

# a generates a copy of Z as a direct summand of S: S = <a> (+) M, <a> = Z
def zsummand_def: forall a:el, S:set: zsummand(a,S) <-> (exists M:set: a in S and subgroup(S) and subgroup(M) and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> n = 0))

# a generates a copy of Z/q as a direct summand of S (q = p^k)
def zpksummand_def: forall a:el, S:set, p k:int: zpksummand(a,S,p,k) <-> (exists M:set: a in S and subgroup(S) and subgroup(M) and p^k*a = 0 and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> exists q:int: n = p^k*q))

lemma zsummand_trans: forall a:el, S T:set: zsummand(a,S) and summand(S,T) -> zsummand(a,T)
proof
  fix a S T
  intro hz hs
  obtain M:set st aS sS hM hMS hdec hind: a in S and subgroup(S) and subgroup(M) and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> n = 0) by zsummand_def hz
  obtain N:set st sS2 sT hN hST hNT hdec2 hind2: subgroup(S) and subgroup(T) and subgroup(N) and (forall x:el: x in S -> x in T) and (forall x:el: x in N -> x in T) and (forall x:el: x in T -> exists s m:el: s in S and m in N and x = s + m) and (forall s m:el: s in S and m in N and s + m = 0 -> s = 0) by summand_def hs
  have hMN: subgroup(ssum(M,N)) by ssum_sub hM hN
  have aT: a in T by hST aS
  have sub: forall x:el: x in ssum(M,N) -> x in T
  proof
    fix x
    intro hx
    obtain m n:el st mM nN e: m in M and n in N and x = m + n by ssum_def hx
    have mS: m in S by hMS mM
    have mT: m in T by hST mS
    have nT: n in T by hNT nN
    have mn: m + n in T by sT mT nT
    show x in T by e mn
  qed
  have dec: forall x:el: x in T -> exists q:int, w:el: w in ssum(M,N) and x = q*a + w
  proof
    fix x
    intro hx
    obtain s n:el st sS3 nN e: s in S and n in N and x = s + n by hdec2 hx
    obtain q:int, m:el st mM e2: m in M and s = q*a + m by hdec sS3
    have w: m + n in ssum(M,N) by ssum_def mM nN with m, n
    show by w e e2 with q, m + n
  qed
  have ind: forall q:int, w:el: w in ssum(M,N) and q*a + w = 0 -> q = 0
  proof
    fix q w
    intro hw h0
    obtain m n:el st mM nN e: m in M and n in N and w = m + n by ssum_def hw
    have mS: m in S by hMS mM
    have qa: q*a in S by sub_smul[S, q, a] sS aS
    have s1: q*a + m in S by sS qa mS
    have s2: (q*a + m) + n = 0 by h0 e
    have s3: q*a + m = 0 by hind2[q*a + m, n] s1 nN s2
    show q = 0 by hind[q, m] mM s3
  qed
  show zsummand(a,T) by zsummand_def aT sT hMN sub dec ind with ssum(M,N)
qed

# coordinates in S = <a> (+) M, a of infinite order: no element of the
# form q*a with q >= 1 lies in S[p] + p*q*S (the valuation argument)
lemma zsum_val: forall a:el, S:set: zsummand(a,S) -> forall p q:int, t x:el: 2 <= p and 1 <= q and t in S and x in S and p*t = 0 and q*a = t + (p*q)*x -> false
proof
  fix a S
  intro hz
  fix p q t x
  intro hp hq ht hx hpt hqa
  obtain M:set st aS sS hM hMS hdec hind: a in S and subgroup(S) and subgroup(M) and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> n = 0) by zsummand_def hz
  obtain s:int, m1:el st m1M te: m1 in M and t = s*a + m1 by hdec ht
  obtain u:int, m2:el st m2M xe: m2 in M and x = u*a + m2 by hdec hx
  have pm: p*m1 in M by hM m1M
  have ps0: (p*s)*a + p*m1 = 0 by hpt te
  have ps: p*s = 0 by hind[p*s, p*m1] pm ps0
  have s0: s = 0
  proof
    cases hc: p = 0 or s = 0 by int_domain[p, s] ps
    case h1
    proof
      show by h1 hp
    qed
    case h2
    proof
      show by h2
    qed
  qed
  have mm: -m1 - (p*q)*m2 in M by hM m1M m2M
  have co: (q - s - (p*q)*u)*a + (-m1 - (p*q)*m2) = 0 by hqa te xe
  have z: q - s - (p*q)*u = 0 by hind[q - s - (p*q)*u, -m1 - (p*q)*m2] mm co
  have z2: q*(1 - p*u) = 0 by z s0
  cases hc2: q = 0 or 1 - p*u = 0 by int_domain[q, 1 - p*u] z2
  case h1
  proof
    show by h1 hq
  qed
  case h2
  proof
    show by h2 hp
  qed
qed

# the same coordinates: a is not in S[w] + p*S for w >= 1 (a unit
# argument instead of a valuation)
lemma zsum_unit: forall a:el, S:set: zsummand(a,S) -> forall p w:int, x y:el: 2 <= p and 1 <= w and x in S and y in S and w*x = 0 and a = x + p*y -> false
proof
  fix a S
  intro hz
  fix p w x y
  intro hp hw hx hy hwx ha
  obtain M:set st aS sS hM hMS hdec hind: a in S and subgroup(S) and subgroup(M) and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> n = 0) by zsummand_def hz
  obtain s:int, m1:el st m1M xe: m1 in M and x = s*a + m1 by hdec hx
  obtain u:int, m2:el st m2M ye: m2 in M and y = u*a + m2 by hdec hy
  have wm: w*m1 in M by hM m1M
  have ws0: (w*s)*a + w*m1 = 0 by hwx xe
  have ws: w*s = 0 by hind[w*s, w*m1] wm ws0
  have s0: s = 0
  proof
    cases hc: w = 0 or s = 0 by int_domain[w, s] ws
    case h1
    proof
      show by h1 hw
    qed
    case h2
    proof
      show by h2
    qed
  qed
  have mm: -m1 - p*m2 in M by hM m1M m2M
  have co: (1 - s - p*u)*a + (-m1 - p*m2) = 0 by ha xe ye
  have z: 1 - s - p*u = 0 by hind[1 - s - p*u, -m1 - p*m2] mm co
  show false by z s0 hp
qed

# ---- exact couples ----
pred excouple(cpl)
func D(cpl): set
func E(cpl): set
func i(cpl, el): el hom
func j(cpl, el): el hom
func k(cpl, el): el hom

# D --i--> D --j--> E --k--> D, exact at every place
def excouple_def: forall c:cpl: excouple(c) <-> (subgroup(D(c)) and subgroup(E(c)) and (forall x:el: x in D(c) -> i(c,x) in D(c)) and (forall x:el: x in D(c) -> j(c,x) in E(c)) and (forall e:el: e in E(c) -> k(c,e) in D(c)) and (forall x:el: x in D(c) and j(c,x) = 0 -> exists y:el: y in D(c) and x = i(c,y)) and (forall x:el: x in D(c) -> j(c,i(c,x)) = 0) and (forall e:el: e in E(c) and k(c,e) = 0 -> exists x:el: x in D(c) and e = j(c,x)) and (forall x:el: x in D(c) -> k(c,j(c,x)) = 0) and (forall x:el: x in D(c) and i(c,x) = 0 -> exists e:el: e in E(c) and x = k(c,e)) and (forall e:el: e in E(c) -> i(c,k(c,e)) = 0))

lemma ec_subD: forall c:cpl: excouple(c) -> subgroup(D(c))
by excouple_def
lemma ec_subE: forall c:cpl: excouple(c) -> subgroup(E(c))
by excouple_def
lemma ec_iD: forall c:cpl: excouple(c) -> forall x:el: x in D(c) -> i(c,x) in D(c)
by excouple_def
lemma ec_jE: forall c:cpl: excouple(c) -> forall x:el: x in D(c) -> j(c,x) in E(c)
by excouple_def
lemma ec_kD: forall c:cpl: excouple(c) -> forall e:el: e in E(c) -> k(c,e) in D(c)
by excouple_def
lemma ec_kerj: forall c:cpl: excouple(c) -> forall x:el: x in D(c) and j(c,x) = 0 -> exists y:el: y in D(c) and x = i(c,y)
by excouple_def
lemma ec_ji: forall c:cpl: excouple(c) -> forall x:el: x in D(c) -> j(c,i(c,x)) = 0
by excouple_def
lemma ec_kerk: forall c:cpl: excouple(c) -> forall e:el: e in E(c) and k(c,e) = 0 -> exists x:el: x in D(c) and e = j(c,x)
by excouple_def
lemma ec_kj: forall c:cpl: excouple(c) -> forall x:el: x in D(c) -> k(c,j(c,x)) = 0
by excouple_def
lemma ec_keri: forall c:cpl: excouple(c) -> forall x:el: x in D(c) and i(c,x) = 0 -> exists e:el: e in E(c) and x = k(c,e)
by excouple_def
lemma ec_ik: forall c:cpl: excouple(c) -> forall e:el: e in E(c) -> i(c,k(c,e)) = 0
by excouple_def

# ---- iterates of i (i^m; 0 for m < 0 by convention) ----
func ipow(cpl, int, el): el
def ipow_zero: forall c:cpl, x:el: ipow(c,0,x) = x
def ipow_succ: forall c:cpl, m:int, x:el: 0 <= m -> ipow(c,m+1,x) = i(c,ipow(c,m,x))
def ipow_neg: forall c:cpl, m:int, x:el: m < 0 -> ipow(c,m,x) = 0

lemma ipow_add: forall c:cpl, m:int, x y:el: ipow(c,m,x+y) = ipow(c,m,x) + ipow(c,m,y)
proof
  fix c m x y
  have nat: forall n:int: 0 <= n -> ipow(c,n,x+y) = ipow(c,n,x) + ipow(c,n,y)
  proof
    induct n from 0
    base
    proof
      show by ipow_zero[c, x+y] ipow_zero[c, x] ipow_zero[c, y]
    qed
    step
    proof
      fix n
      intro hn ih
      show by ih ipow_succ[c, n, x+y] ipow_succ[c, n, x] ipow_succ[c, n, y] hn
    qed
  qed
  cases hc: m < 0 or 0 <= m by int_discrete[m+1]
  case h1
  proof
    show by ipow_neg[c, m, x+y] ipow_neg[c, m, x] ipow_neg[c, m, y] h1
  qed
  case h2
  proof
    show by nat h2
  qed
qed

lemma ipow_smul: forall c:cpl, m:int, n:int, x:el: ipow(c,m,n*x) = n*ipow(c,m,x)
proof
  fix c m n x
  have nat: forall q:int: 0 <= q -> ipow(c,q,n*x) = n*ipow(c,q,x)
  proof
    induct q from 0
    base
    proof
      show by ipow_zero[c, n*x] ipow_zero[c, x]
    qed
    step
    proof
      fix q
      intro hq ih
      show by ih ipow_succ[c, q, n*x] ipow_succ[c, q, x] hq
    qed
  qed
  cases hc: m < 0 or 0 <= m by int_discrete[m+1]
  case h1
  proof
    show by ipow_neg[c, m, n*x] ipow_neg[c, m, x] h1
  qed
  case h2
  proof
    show by nat h2
  qed
qed

homdecl ipow by ipow_add ipow_smul

lemma ipow_shift: forall c:cpl, m:int: 0 <= m -> forall x:el: ipow(c,m+1,x) = ipow(c,m,i(c,x))
proof
  fix c
  induct m from 0
  base
  proof
    fix x
    show by ipow_succ[c, 0, x] ipow_zero[c, x] ipow_zero[c, i(c,x)]
  qed
  step
  proof
    fix m
    intro hm ih
    fix x
    have a: ipow(c,m+1,x) = ipow(c,m,i(c,x)) by ih
    have b: ipow(c,m+2,x) = i(c,ipow(c,m+1,x)) by ipow_succ[c, m+1, x] hm
    have d: ipow(c,m+1,i(c,x)) = i(c,ipow(c,m,i(c,x))) by ipow_succ[c, m, i(c,x)] hm
    show by a b d
  qed
qed

lemma ipow_D: forall c:cpl: excouple(c) -> forall m:int, x:el: x in D(c) -> ipow(c,m,x) in D(c)
proof
  fix c
  intro ec
  have sD: subgroup(D(c)) by ec_subD ec
  have nat: forall m:int: 0 <= m -> forall x:el: x in D(c) -> ipow(c,m,x) in D(c)
  proof
    induct m from 0
    base
    proof
      fix x
      intro hx
      show by ipow_zero[c, x] hx
    qed
    step
    proof
      fix m
      intro hm ih
      fix x
      intro hx
      have a: ipow(c,m,x) in D(c) by ih hx
      have b: i(c,ipow(c,m,x)) in D(c) by ec_iD ec a
      show by b ipow_succ[c, m, x] hm
    qed
  qed
  fix m x
  intro hx
  cases hc: m < 0 or 0 <= m by int_discrete[m+1]
  case h1
  proof
    have z: 0 in D(c) by sub_zero sD
    show by z ipow_neg[c, m, x] h1
  qed
  case h2
  proof
    show by nat h2 hx
  qed
qed

# ---- derived couples (the construction, repeated) ----
# D_r (Dr), E_r = Z_r / B_r (cycles Zr, boundaries Br, as subgroups of
# E), i_r = i on D_r, k_r [e] = k e, and j_r as the relation Jr: Jr(y, e)
# says j_r(y) = [e].  D' = i D, E' = ker d / im d with d = j k,
# j'(i x) = [j x], k'[e] = k e (the third isomorphism theorem identifies
# (Z'/B)/(B'/B) with Z'/B').
pred Dr(cpl, int, el)
pred Zr(cpl, int, el)
pred Br(cpl, int, el)
pred Jr(cpl, int, el, el)
def D1_def: forall c:cpl, x:el: Dr(c,1,x) <-> x in D(c)
def Z1_def: forall c:cpl, e:el: Zr(c,1,e) <-> e in E(c)
def B1_def: forall c:cpl, e:el: Br(c,1,e) <-> e = 0
def J1_def: forall c:cpl, x e:el: Jr(c,1,x,e) <-> (x in D(c) and e = j(c,x))
def Dsucc_def: forall c:cpl, r:int, y:el: 1 <= r -> (Dr(c,r+1,y) <-> (exists x:el: Dr(c,r,x) and y = i(c,x)))
def Zsucc_def: forall c:cpl, r:int, e:el: 1 <= r -> (Zr(c,r+1,e) <-> (Zr(c,r,e) and exists f:el: Jr(c,r,k(c,e),f) and Br(c,r,f)))
def Bsucc_def: forall c:cpl, r:int, e:el: 1 <= r -> (Br(c,r+1,e) <-> (exists e0 f:el: Zr(c,r,e0) and Jr(c,r,k(c,e0),f) and Br(c,r,e - f)))
def Jsucc_def: forall c:cpl, r:int, y e:el: 1 <= r -> (Jr(c,r+1,y,e) <-> (exists x f:el: Dr(c,r,x) and y = i(c,x) and Jr(c,r,x,f) and Br(c,r+1,e - f)))

# d_r [e] = 0, and [e] in the image of d_r on degree m
pred Edeg(cpl, int, el)
pred dzero(cpl, int, el)
pred dimg(cpl, int, int, el)
def dzero_def: forall c:cpl, r:int, e:el: dzero(c,r,e) <-> (exists f:el: Jr(c,r,k(c,e),f) and Br(c,r,f))
def dimg_def: forall c:cpl, r m:int, e:el: dimg(c,r,m,e) <-> (exists e0 f:el: Zr(c,r,e0) and Edeg(c,m,e0) and Jr(c,r,k(c,e0),f) and Br(c,r,e - f))

# the closed forms of the r-th derived couple, by induction on r:
# D_r = i^(r-1) D, Z_r = k^-1(i^(r-1) D), B_r = j(ker i^(r-1)),
# j_r(i^(r-1) x) = [j x]
lemma closed_forms: forall c:cpl: excouple(c) -> forall r:int: 1 <= r -> (forall y:el: Dr(c,r,y) <-> (exists x:el: x in D(c) and y = ipow(c,r-1,x))) and (forall e:el: Zr(c,r,e) <-> (e in E(c) and exists x:el: x in D(c) and k(c,e) = ipow(c,r-1,x))) and (forall e:el: Br(c,r,e) <-> (exists x:el: x in D(c) and ipow(c,r-1,x) = 0 and e = j(c,x))) and (forall y e:el: Jr(c,r,y,e) <-> (exists x:el: x in D(c) and y = ipow(c,r-1,x) and Br(c,r,e - j(c,x))))
proof
  fix c
  intro ec
  have sD: subgroup(D(c)) by ec_subD ec
  have sE: subgroup(E(c)) by ec_subE ec
  have iz: forall x:el: ipow(c,0,x) = x by ipow_zero
  induct r from 1
  base
  proof
    have pD: forall y:el: Dr(c,1,y) <-> (exists x:el: x in D(c) and y = ipow(c,0,x))
    proof
      fix y
      have l: Dr(c,1,y) -> exists x:el: x in D(c) and y = ipow(c,0,x)
      proof
        intro h
        have yD: y in D(c) by D1_def h
        show by yD iz[y] with y
      qed
      have rr: (exists x:el: x in D(c) and y = ipow(c,0,x)) -> Dr(c,1,y)
      proof
        intro h
        obtain x:el st xD e: x in D(c) and y = ipow(c,0,x) by h
        have yD: y in D(c) by xD e iz[x]
        show by D1_def yD
      qed
      show by l rr
    qed
    have pZ: forall e:el: Zr(c,1,e) <-> (e in E(c) and exists x:el: x in D(c) and k(c,e) = ipow(c,0,x))
    proof
      fix e
      have l: Zr(c,1,e) -> e in E(c) and exists x:el: x in D(c) and k(c,e) = ipow(c,0,x)
      proof
        intro h
        have eE: e in E(c) by Z1_def h
        have kD: k(c,e) in D(c) by ec_kD ec eE
        have ex: exists x:el: x in D(c) and k(c,e) = ipow(c,0,x) by kD iz[k(c,e)] with k(c,e)
        show by eE ex
      qed
      have rr: e in E(c) and (exists x:el: x in D(c) and k(c,e) = ipow(c,0,x)) -> Zr(c,1,e)
      proof
        intro eE h
        show by Z1_def eE
      qed
      show by l rr
    qed
    have pB: forall e:el: Br(c,1,e) <-> (exists x:el: x in D(c) and ipow(c,0,x) = 0 and e = j(c,x))
    proof
      fix e
      have l: Br(c,1,e) -> exists x:el: x in D(c) and ipow(c,0,x) = 0 and e = j(c,x)
      proof
        intro h
        have e0: e = 0 by B1_def h
        have z: 0 in D(c) by sub_zero sD
        show by z e0 iz[0] with 0
      qed
      have rr: (exists x:el: x in D(c) and ipow(c,0,x) = 0 and e = j(c,x)) -> Br(c,1,e)
      proof
        intro h
        obtain x:el st xD x0 ex: x in D(c) and ipow(c,0,x) = 0 and e = j(c,x) by h
        have xz: x = 0 by x0 iz[x]
        have e0: e = 0 by ex xz
        show by B1_def e0
      qed
      show by l rr
    qed
    have pJ: forall y e:el: Jr(c,1,y,e) <-> (exists x:el: x in D(c) and y = ipow(c,0,x) and Br(c,1,e - j(c,x)))
    proof
      fix y e
      have l: Jr(c,1,y,e) -> exists x:el: x in D(c) and y = ipow(c,0,x) and Br(c,1,e - j(c,x))
      proof
        intro h
        have yD: y in D(c) by J1_def h
        have ey: e = j(c,y) by J1_def h
        have b: Br(c,1,e - j(c,y)) by B1_def ey
        show by yD b iz[y] with y
      qed
      have rr: (exists x:el: x in D(c) and y = ipow(c,0,x) and Br(c,1,e - j(c,x))) -> Jr(c,1,y,e)
      proof
        intro h
        obtain x:el st xD yx b: x in D(c) and y = ipow(c,0,x) and Br(c,1,e - j(c,x)) by h
        have yx2: y = x by yx iz[x]
        have yD: y in D(c) by xD yx2
        have e0: e - j(c,x) = 0 by B1_def b
        have ey: e = j(c,y) by e0 yx2
        show by J1_def yD ey
      qed
      show by l rr
    qed
    show by pD pZ pB pJ
  qed
  step
  proof
    fix r
    intro hr ih
    have hD: forall y:el: Dr(c,r,y) <-> (exists x:el: x in D(c) and y = ipow(c,r-1,x)) by ih
    have hZ: forall e:el: Zr(c,r,e) <-> (e in E(c) and exists x:el: x in D(c) and k(c,e) = ipow(c,r-1,x)) by ih
    have hB: forall e:el: Br(c,r,e) <-> (exists x:el: x in D(c) and ipow(c,r-1,x) = 0 and e = j(c,x)) by ih
    have hJ: forall y e:el: Jr(c,r,y,e) <-> (exists x:el: x in D(c) and y = ipow(c,r-1,x) and Br(c,r,e - j(c,x))) by ih
    have r0: 0 <= r - 1 by hr
    have isucc: forall x:el: ipow(c,r,x) = i(c,ipow(c,r-1,x)) by ipow_succ r0
    have ishift: forall x:el: ipow(c,r,x) = ipow(c,r-1,i(c,x)) by ipow_shift r0
    have B0: Br(c,r,0)
    proof
      have z: 0 in D(c) by sub_zero sD
      have i0: ipow(c,r-1,0) = 0 by z
      show by hB z i0 with 0
    qed
    # D_(r+1) = i^r D
    have pD: forall y:el: Dr(c,r+1,y) <-> (exists x:el: x in D(c) and y = ipow(c,r,x))
    proof
      fix y
      have l: Dr(c,r+1,y) -> exists x:el: x in D(c) and y = ipow(c,r,x)
      proof
        intro h
        obtain x1:el st h1 h2: Dr(c,r,x1) and y = i(c,x1) by Dsucc_def h hr
        obtain x:el st xD h4: x in D(c) and x1 = ipow(c,r-1,x) by hD h1
        have h5: ipow(c,r,x) = i(c,ipow(c,r-1,x)) by isucc
        show by xD h2 h4 h5 with x
      qed
      have rr: (exists x:el: x in D(c) and y = ipow(c,r,x)) -> Dr(c,r+1,y)
      proof
        intro h
        obtain x:el st xD h2: x in D(c) and y = ipow(c,r,x) by h
        have h3: Dr(c,r,ipow(c,r-1,x)) by hD xD with x
        have h5: y = i(c,ipow(c,r-1,x)) by h2 isucc[x]
        show by Dsucc_def hr h3 h5 with ipow(c,r-1,x)
      qed
      show by l rr
    qed
    # B_(r+1) = j(ker i^r)
    have pB: forall e:el: Br(c,r+1,e) <-> (exists x:el: x in D(c) and ipow(c,r,x) = 0 and e = j(c,x))
    proof
      fix e
      have l: Br(c,r+1,e) -> exists x:el: x in D(c) and ipow(c,r,x) = 0 and e = j(c,x)
      proof
        intro h
        obtain e0 f:el st z0 jf bf: Zr(c,r,e0) and Jr(c,r,k(c,e0),f) and Br(c,r,e - f) by Bsucc_def h hr
        have e0E: e0 in E(c) by hZ z0
        obtain x1:el st x1D k1 b1: x1 in D(c) and k(c,e0) = ipow(c,r-1,x1) and Br(c,r,f - j(c,x1)) by hJ jf
        obtain x2:el st x2D i2 e2: x2 in D(c) and ipow(c,r-1,x2) = 0 and f - j(c,x1) = j(c,x2) by hB b1
        obtain x3:el st x3D i3 e3: x3 in D(c) and ipow(c,r-1,x3) = 0 and e - f = j(c,x3) by hB bf
        have xD: x1 + x2 + x3 in D(c) by sD x1D x2D x3D
        have ik: i(c,k(c,e0)) = 0 by ec_ik ec e0E
        have iz: ipow(c,r,x1 + x2 + x3) = 0 by isucc[x1 + x2 + x3] k1 i2 i3 ik
        have ee: e = j(c,x1 + x2 + x3) by e2 e3
        show by xD iz ee with x1 + x2 + x3
      qed
      have rr: (exists x:el: x in D(c) and ipow(c,r,x) = 0 and e = j(c,x)) -> Br(c,r+1,e)
      proof
        intro h
        obtain x:el st xD iz ex: x in D(c) and ipow(c,r,x) = 0 and e = j(c,x) by h
        have yD: ipow(c,r-1,x) in D(c) by ipow_D ec xD
        have iy: i(c,ipow(c,r-1,x)) = 0 by iz isucc[x]
        obtain e0:el st e0E ke: e0 in E(c) and ipow(c,r-1,x) = k(c,e0) by ec_keri ec yD iy
        have z0: Zr(c,r,e0) by hZ e0E xD ke with x
        have jj: Br(c,r,j(c,x) - j(c,x)) by B0
        have jf: Jr(c,r,k(c,e0),j(c,x)) by hJ xD ke jj with x
        have bf: Br(c,r,e - j(c,x)) by B0 ex
        show by Bsucc_def hr z0 jf bf with e0, j(c,x)
      qed
      show by l rr
    qed
    # Z_(r+1) = k^-1(i^r D)
    have pZ: forall e:el: Zr(c,r+1,e) <-> (e in E(c) and exists x:el: x in D(c) and k(c,e) = ipow(c,r,x))
    proof
      fix e
      have l: Zr(c,r+1,e) -> e in E(c) and exists x:el: x in D(c) and k(c,e) = ipow(c,r,x)
      proof
        intro h
        have z: Zr(c,r,e) by Zsucc_def h hr
        have eE: e in E(c) by hZ z
        obtain f:el st jf bf: Jr(c,r,k(c,e),f) and Br(c,r,f) by Zsucc_def h hr
        obtain x1:el st x1D k1 b1: x1 in D(c) and k(c,e) = ipow(c,r-1,x1) and Br(c,r,f - j(c,x1)) by hJ jf
        obtain x2:el st x2D i2 e2: x2 in D(c) and ipow(c,r-1,x2) = 0 and f = j(c,x2) by hB bf
        obtain x3:el st x3D i3 e3: x3 in D(c) and ipow(c,r-1,x3) = 0 and f - j(c,x1) = j(c,x3) by hB b1
        have zD: x1 - x2 + x3 in D(c) by sD x1D x2D x3D
        have jz: j(c,x1 - x2 + x3) = 0 by e2 e3
        obtain w:el st wD wi: w in D(c) and x1 - x2 + x3 = i(c,w) by ec_kerj ec zD jz
        have kw: k(c,e) = ipow(c,r,w) by ishift[w] wi k1 i2 i3
        have ex: exists x:el: x in D(c) and k(c,e) = ipow(c,r,x) by wD kw with w
        show by eE ex
      qed
      have rr: e in E(c) and (exists x:el: x in D(c) and k(c,e) = ipow(c,r,x)) -> Zr(c,r+1,e)
      proof
        intro eE h
        obtain x:el st xD kx: x in D(c) and k(c,e) = ipow(c,r,x) by h
        have ixD: i(c,x) in D(c) by ec_iD ec xD
        have kx2: k(c,e) = ipow(c,r-1,i(c,x)) by kx ishift[x]
        have z: Zr(c,r,e) by hZ eE ixD kx2 with i(c,x)
        have ji: j(c,i(c,x)) = 0 by ec_ji ec xD
        have bb: Br(c,r,0 - j(c,i(c,x))) by B0 ji
        have jf: Jr(c,r,k(c,e),0) by hJ ixD kx2 bb with i(c,x)
        show by Zsucc_def hr z jf B0 with 0
      qed
      show by l rr
    qed
    # j_(r+1)(i^r x) = [j x]
    have pJ: forall y e:el: Jr(c,r+1,y,e) <-> (exists x:el: x in D(c) and y = ipow(c,r,x) and Br(c,r+1,e - j(c,x)))
    proof
      fix y e
      have l: Jr(c,r+1,y,e) -> exists x:el: x in D(c) and y = ipow(c,r,x) and Br(c,r+1,e - j(c,x))
      proof
        intro h
        obtain x0 f:el st d0 y0 jf bf: Dr(c,r,x0) and y = i(c,x0) and Jr(c,r,x0,f) and Br(c,r+1,e - f) by Jsucc_def h hr
        obtain x1:el st x1D k1 b1: x1 in D(c) and x0 = ipow(c,r-1,x1) and Br(c,r,f - j(c,x1)) by hJ jf
        obtain x2:el st x2D i2 e2: x2 in D(c) and ipow(c,r-1,x2) = 0 and f - j(c,x1) = j(c,x2) by hB b1
        obtain x3:el st x3D i3 e3: x3 in D(c) and ipow(c,r,x3) = 0 and e - f = j(c,x3) by pB bf
        have xD: x3 + x2 in D(c) by sD x2D x3D
        have iz: ipow(c,r,x3 + x2) = 0 by i3 i2 isucc[x2]
        have ee: e - j(c,x1) = j(c,x3 + x2) by e2 e3
        have b: Br(c,r+1,e - j(c,x1)) by pB xD iz ee with x3 + x2
        have yy: y = ipow(c,r,x1) by y0 k1 isucc[x1]
        show by x1D yy b with x1
      qed
      have rr: (exists x:el: x in D(c) and y = ipow(c,r,x) and Br(c,r+1,e - j(c,x))) -> Jr(c,r+1,y,e)
      proof
        intro h
        obtain x:el st xD yx b: x in D(c) and y = ipow(c,r,x) and Br(c,r+1,e - j(c,x)) by h
        have d0: Dr(c,r,ipow(c,r-1,x)) by hD xD with x
        have yy: y = i(c,ipow(c,r-1,x)) by yx isucc[x]
        have jj: Br(c,r,j(c,x) - j(c,x)) by B0
        have jf: Jr(c,r,ipow(c,r-1,x),j(c,x)) by hJ xD jj with x
        show by Jsucc_def hr d0 yy jf b with ipow(c,r-1,x), j(c,x)
      qed
      show by l rr
    qed
    show by pD pZ pB pJ
  qed
qed
