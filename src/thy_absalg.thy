# SPDX-License-Identifier: GPL-3.0-or-later
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
lemma zsum_val: forall a:el, S:set, p q:int, t x:el: zsummand(a,S) and 2 <= p and 1 <= q and t in S and x in S and p*t = 0 and q*a = t + (p*q)*x -> false
proof
  fix a S p q t x
  intro hz hp hq ht hx hpt hqa
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
lemma zsum_unit: forall a:el, S:set, p w:int, x y:el: zsummand(a,S) and 2 <= p and 1 <= w and x in S and y in S and w*x = 0 and a = x + p*y -> false
proof
  fix a S p w x y
  intro hz hp hw hx hy hwx ha
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
lemma ec_iD: forall c:cpl, x:el: excouple(c) and x in D(c) -> i(c,x) in D(c)
by excouple_def
lemma ec_jE: forall c:cpl, x:el: excouple(c) and x in D(c) -> j(c,x) in E(c)
by excouple_def
lemma ec_kD: forall c:cpl, e:el: excouple(c) and e in E(c) -> k(c,e) in D(c)
by excouple_def
lemma ec_kerj: forall c:cpl, x:el: excouple(c) and x in D(c) and j(c,x) = 0 -> exists y:el: y in D(c) and x = i(c,y)
by excouple_def
lemma ec_ji: forall c:cpl, x:el: excouple(c) and x in D(c) -> j(c,i(c,x)) = 0
by excouple_def
lemma ec_kerk: forall c:cpl, e:el: excouple(c) and e in E(c) and k(c,e) = 0 -> exists x:el: x in D(c) and e = j(c,x)
by excouple_def
lemma ec_kj: forall c:cpl, x:el: excouple(c) and x in D(c) -> k(c,j(c,x)) = 0
by excouple_def
lemma ec_keri: forall c:cpl, x:el: excouple(c) and x in D(c) and i(c,x) = 0 -> exists e:el: e in E(c) and x = k(c,e)
by excouple_def
lemma ec_ik: forall c:cpl, e:el: excouple(c) and e in E(c) -> i(c,k(c,e)) = 0
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

# the closed forms one at a time
lemma cf_D: forall c:cpl, r:int, y:el: excouple(c) and 1 <= r -> (Dr(c,r,y) <-> (exists x:el: x in D(c) and y = ipow(c,r-1,x)))
by closed_forms
lemma cf_Z: forall c:cpl, r:int, e:el: excouple(c) and 1 <= r -> (Zr(c,r,e) <-> (e in E(c) and exists x:el: x in D(c) and k(c,e) = ipow(c,r-1,x)))
by closed_forms
lemma cf_B: forall c:cpl, r:int, e:el: excouple(c) and 1 <= r -> (Br(c,r,e) <-> (exists x:el: x in D(c) and ipow(c,r-1,x) = 0 and e = j(c,x)))
by closed_forms
lemma cf_J: forall c:cpl, r:int, y e:el: excouple(c) and 1 <= r -> (Jr(c,r,y,e) <-> (exists x:el: x in D(c) and y = ipow(c,r-1,x) and Br(c,r,e - j(c,x))))
by closed_forms

# ---- the r-th derived couple, as the paper uses it ----
lemma dc_B0: forall c:cpl, r:int: excouple(c) and 1 <= r -> Br(c,r,0)
proof
  fix c r
  intro ec hr
  have z: 0 in D(c) by sub_zero ec_subD ec
  show by cf_B ec hr z with 0
qed

lemma dc_Bsub: forall c:cpl, r:int, u v:el: excouple(c) and 1 <= r and Br(c,r,u) and Br(c,r,v) -> Br(c,r,u - v)
proof
  fix c r u v
  intro ec hr hu hv
  have sD: subgroup(D(c)) by ec_subD ec
  obtain x1:el st x1D i1 e1: x1 in D(c) and ipow(c,r-1,x1) = 0 and u = j(c,x1) by cf_B ec hr hu
  obtain x2:el st x2D i2 e2: x2 in D(c) and ipow(c,r-1,x2) = 0 and v = j(c,x2) by cf_B ec hr hv
  have d: x1 - x2 in D(c) by sD x1D x2D
  show by cf_B ec hr d i1 i2 e1 e2 with x1 - x2
qed

# D_r = i^(r-1) D (= p^(r-1) H for the Bockstein couple)
lemma dc_Dr: forall c:cpl, r:int, y:el: excouple(c) and 1 <= r -> (Dr(c,r,y) <-> (exists x:el: x in D(c) and y = ipow(c,r-1,x)))
by cf_D

lemma dc_DD: forall c:cpl, r:int, y:el: excouple(c) and 1 <= r and Dr(c,r,y) -> y in D(c)
proof
  fix c r y
  intro ec hr hd
  obtain x:el st xD yx: x in D(c) and y = ipow(c,r-1,x) by cf_D ec hr hd
  have a: ipow(c,r-1,x) in D(c) by ipow_D ec xD
  show by a yx
qed

# i_r maps D_r into D_r (i_r is i restricted)
lemma dc_iD: forall c:cpl, r:int, y:el: excouple(c) and 1 <= r and Dr(c,r,y) -> Dr(c,r,i(c,y))
proof
  fix c r y
  intro ec hr hd
  have r0: 0 <= r - 1 by hr
  obtain x:el st xD yx: x in D(c) and y = ipow(c,r-1,x) by cf_D ec hr hd
  have ixD: i(c,x) in D(c) by ec_iD ec xD
  have s1: ipow(c,r,x) = i(c,ipow(c,r-1,x)) by ipow_succ[c, r-1, x] r0
  have s2: ipow(c,r,x) = ipow(c,r-1,i(c,x)) by ipow_shift[c, r-1] r0
  show by cf_D ec hr ixD yx s1 s2 with i(c,x)
qed

# j_r(i^(r-1) x) = [j x]  (j_r(p^(r-1) a) = a-bar)
lemma dc_jr: forall c:cpl, r:int, x:el: excouple(c) and 1 <= r and x in D(c) -> Jr(c,r,ipow(c,r-1,x),j(c,x))
proof
  fix c r x
  intro ec hr hx
  have b: Br(c,r,j(c,x) - j(c,x)) by dc_B0 ec hr
  show by cf_J ec hr hx b with x
qed

# j_r is additive and well defined on classes, with values in Z_r
lemma dc_Jsub: forall c:cpl, r:int, y1 y2 e1 e2:el: excouple(c) and 1 <= r and Jr(c,r,y1,e1) and Jr(c,r,y2,e2) -> Jr(c,r,y1 - y2,e1 - e2)
proof
  fix c r y1 y2 e1 e2
  intro ec hr h1 h2
  have sD: subgroup(D(c)) by ec_subD ec
  obtain x1:el st x1D y1e b1: x1 in D(c) and y1 = ipow(c,r-1,x1) and Br(c,r,e1 - j(c,x1)) by cf_J ec hr h1
  obtain x2:el st x2D y2e b2: x2 in D(c) and y2 = ipow(c,r-1,x2) and Br(c,r,e2 - j(c,x2)) by cf_J ec hr h2
  have d: x1 - x2 in D(c) by sD x1D x2D
  have b: Br(c,r,(e1 - e2) - j(c,x1 - x2)) by dc_Bsub[c, r, e1 - j(c,x1), e2 - j(c,x2)] ec hr b1 b2
  show by cf_J ec hr d y1e y2e b with x1 - x2
qed

lemma dc_Jwd: forall c:cpl, r:int, y e1 e2:el: excouple(c) and 1 <= r and Jr(c,r,y,e1) and Jr(c,r,y,e2) -> Br(c,r,e1 - e2)
proof
  fix c r y e1 e2
  intro ec hr h1 h2
  have j0: Jr(c,r,y - y,e1 - e2) by dc_Jsub[c, r, y, y, e1, e2] ec hr h1 h2
  obtain x:el st xD x0 b: x in D(c) and y - y = ipow(c,r-1,x) and Br(c,r,(e1 - e2) - j(c,x)) by cf_J ec hr j0
  have bx: Br(c,r,j(c,x)) by cf_B ec hr xD x0 with x
  have b0: Br(c,r,0) by dc_B0 ec hr
  have nx: Br(c,r,0 - j(c,x)) by dc_Bsub[c, r, 0, j(c,x)] ec hr b0 bx
  have s: Br(c,r,((e1 - e2) - j(c,x)) - (0 - j(c,x))) by dc_Bsub[c, r, (e1 - e2) - j(c,x), 0 - j(c,x)] ec hr b nx
  show by s
qed

# exactness of the r-th derived couple
# ker j_r = im i_r
lemma dc_kerj: forall c:cpl, r:int, y e:el: excouple(c) and 1 <= r and Jr(c,r,y,e) and Br(c,r,e) -> exists y2:el: Dr(c,r,y2) and y = i(c,y2)
proof
  fix c r y e
  intro ec hr hj hb
  have sD: subgroup(D(c)) by ec_subD ec
  have r0: 0 <= r - 1 by hr
  obtain x:el st xD yx b1: x in D(c) and y = ipow(c,r-1,x) and Br(c,r,e - j(c,x)) by cf_J ec hr hj
  obtain x2:el st x2D i2 e2: x2 in D(c) and ipow(c,r-1,x2) = 0 and e = j(c,x2) by cf_B ec hr hb
  obtain x3:el st x3D i3 e3: x3 in D(c) and ipow(c,r-1,x3) = 0 and e - j(c,x) = j(c,x3) by cf_B ec hr b1
  have zD: x - x2 + x3 in D(c) by sD xD x2D x3D
  have jz: j(c,x - x2 + x3) = 0 by e2 e3
  obtain w:el st wD wz: w in D(c) and x - x2 + x3 = i(c,w) by ec_kerj ec zD jz
  have d: Dr(c,r,ipow(c,r-1,w)) by cf_D ec hr wD with w
  have s1: ipow(c,r,w) = i(c,ipow(c,r-1,w)) by ipow_succ[c, r-1, w] r0
  have s2: ipow(c,r,w) = ipow(c,r-1,i(c,w)) by ipow_shift[c, r-1] r0
  have yy: y = i(c,ipow(c,r-1,w)) by s1 s2 wz yx i2 i3
  show by d yy with ipow(c,r-1,w)
qed

lemma dc_imi: forall c:cpl, r:int, y:el: excouple(c) and 1 <= r and Dr(c,r,y) -> Jr(c,r,i(c,y),0)
proof
  fix c r y
  intro ec hr hd
  have r0: 0 <= r - 1 by hr
  obtain x:el st xD yx: x in D(c) and y = ipow(c,r-1,x) by cf_D ec hr hd
  have ixD: i(c,x) in D(c) by ec_iD ec xD
  have jj: Jr(c,r,ipow(c,r-1,i(c,x)),j(c,i(c,x))) by dc_jr ec hr ixD
  have ji: j(c,i(c,x)) = 0 by ec_ji ec xD
  have s1: ipow(c,r,x) = i(c,ipow(c,r-1,x)) by ipow_succ[c, r-1, x] r0
  have s2: ipow(c,r,x) = ipow(c,r-1,i(c,x)) by ipow_shift[c, r-1] r0
  have iy: i(c,y) = ipow(c,r-1,i(c,x)) by yx s1 s2
  show by jj ji iy
qed

# ker k_r = im j_r
lemma dc_kerk: forall c:cpl, r:int, e:el: excouple(c) and 1 <= r and Zr(c,r,e) and k(c,e) = 0 -> exists y:el: Jr(c,r,y,e)
proof
  fix c r e
  intro ec hr hz hk
  have eE: e in E(c) by cf_Z ec hr hz
  obtain x:el st xD ex: x in D(c) and e = j(c,x) by ec_kerk ec eE hk
  have jj: Jr(c,r,ipow(c,r-1,x),j(c,x)) by dc_jr ec hr xD
  show by jj ex with ipow(c,r-1,x)
qed

lemma dc_imj: forall c:cpl, r:int, y e:el: excouple(c) and 1 <= r and Jr(c,r,y,e) -> k(c,e) = 0
proof
  fix c r y e
  intro ec hr hj
  have sD: subgroup(D(c)) by ec_subD ec
  obtain x:el st xD yx b: x in D(c) and y = ipow(c,r-1,x) and Br(c,r,e - j(c,x)) by cf_J ec hr hj
  obtain x3:el st x3D i3 e3: x3 in D(c) and ipow(c,r-1,x3) = 0 and e - j(c,x) = j(c,x3) by cf_B ec hr b
  have sD2: x + x3 in D(c) by sD xD x3D
  have kj: k(c,j(c,x + x3)) = 0 by ec_kj[c, x + x3] ec sD2
  show by kj e3
qed

lemma dc_JZ: forall c:cpl, r:int, y e:el: excouple(c) and 1 <= r and Jr(c,r,y,e) -> Zr(c,r,e)
proof
  fix c r y e
  intro ec hr hj
  have sD: subgroup(D(c)) by ec_subD ec
  have sE: subgroup(E(c)) by ec_subE ec
  have ke: k(c,e) = 0 by dc_imj ec hr hj
  obtain x:el st xD yx b: x in D(c) and y = ipow(c,r-1,x) and Br(c,r,e - j(c,x)) by cf_J ec hr hj
  obtain x3:el st x3D i3 e3: x3 in D(c) and ipow(c,r-1,x3) = 0 and e - j(c,x) = j(c,x3) by cf_B ec hr b
  have sD2: x + x3 in D(c) by sD xD x3D
  have jE: j(c,x + x3) in E(c) by ec_jE ec sD2
  have eE: e in E(c) by jE e3
  have z: 0 in D(c) by sub_zero sD
  show by cf_Z ec hr eE ke z with 0
qed

# ker i_r = im k_r
lemma dc_keri: forall c:cpl, r:int, y:el: excouple(c) and 1 <= r and Dr(c,r,y) and i(c,y) = 0 -> exists e:el: Zr(c,r,e) and y = k(c,e)
proof
  fix c r y
  intro ec hr hd hi
  obtain x:el st xD yx: x in D(c) and y = ipow(c,r-1,x) by cf_D ec hr hd
  have yD: y in D(c) by dc_DD ec hr hd
  obtain e:el st eE ye: e in E(c) and y = k(c,e) by ec_keri ec yD hi
  have z: Zr(c,r,e) by cf_Z ec hr eE xD ye yx with x
  show by z ye with e
qed

lemma dc_imk: forall c:cpl, r:int, e:el: excouple(c) and 1 <= r and Zr(c,r,e) -> Dr(c,r,k(c,e)) and i(c,k(c,e)) = 0
proof
  fix c r e
  intro ec hr hz
  have eE: e in E(c) by cf_Z ec hr hz
  obtain x:el st xD kx: x in D(c) and k(c,e) = ipow(c,r-1,x) by cf_Z ec hr hz
  have d: Dr(c,r,k(c,e)) by cf_D ec hr xD kx with x
  have i0: i(c,k(c,e)) = 0 by ec_ik ec eE
  show by d i0
qed

# the derived couple of an exact couple is exact, at every page r
theorem derived_exact: forall c:cpl, r:int: excouple(c) and 1 <= r -> (forall y e:el: Jr(c,r,y,e) and Br(c,r,e) -> exists y2:el: Dr(c,r,y2) and y = i(c,y2)) and (forall y:el: Dr(c,r,y) -> Jr(c,r,i(c,y),0)) and (forall e:el: Zr(c,r,e) and k(c,e) = 0 -> exists y:el: Jr(c,r,y,e)) and (forall y e:el: Jr(c,r,y,e) -> k(c,e) = 0) and (forall y:el: Dr(c,r,y) and i(c,y) = 0 -> exists e:el: Zr(c,r,e) and y = k(c,e)) and (forall e:el: Zr(c,r,e) -> Dr(c,r,k(c,e)) and i(c,k(c,e)) = 0)
proof
  fix c r
  intro ec hr
  have a1: forall y e:el: Jr(c,r,y,e) and Br(c,r,e) -> exists y2:el: Dr(c,r,y2) and y = i(c,y2) by dc_kerj ec hr
  have a2: forall y:el: Dr(c,r,y) -> Jr(c,r,i(c,y),0) by dc_imi ec hr
  have a3: forall e:el: Zr(c,r,e) and k(c,e) = 0 -> exists y:el: Jr(c,r,y,e) by dc_kerk ec hr
  have a4: forall y e:el: Jr(c,r,y,e) -> k(c,e) = 0 by dc_imj ec hr
  have a5: forall y:el: Dr(c,r,y) and i(c,y) = 0 -> exists e:el: Zr(c,r,e) and y = k(c,e) by dc_keri ec hr
  have a6: forall e:el: Zr(c,r,e) -> Dr(c,r,k(c,e)) and i(c,k(c,e)) = 0 by dc_imk ec hr
  show by a1 a2 a3 a4 a5 a6
qed

# d_r d_r = 0
theorem dd_zero: forall c:cpl, r:int, e f:el: excouple(c) and 1 <= r and Zr(c,r,e) and Jr(c,r,k(c,e),f) -> dzero(c,r,f)
proof
  fix c r e f
  intro ec hr hz hj
  have kf: k(c,f) = 0 by dc_imj ec hr hj
  have z: 0 in D(c) by sub_zero ec_subD ec
  have j0: Jr(c,r,0,0) by dc_jr[c, r, 0] ec hr z
  have b0: Br(c,r,0) by dc_B0 ec hr
  have j1: Jr(c,r,k(c,f),0) by j0 kf
  show by dzero_def j1 b0 with 0
qed

# ---- cochain complexes and cohomology ----
# A complex A has cochains Ch(A) (every degree; the grading below) and a
# differential dif.  H(A) = Z/B is given with its projection as the
# relation Cls(A,z,h), "h is the class of the cocycle z": the five cls_
# lines are the definition of a quotient group.
sort hom
func ap(hom, el): el hom
func Ch(cx): set
func dif(cx, el): el hom
pred iscx(cx)
def iscx_def: forall A:cx: iscx(A) <-> (subgroup(Ch(A)) and (forall x:el: x in Ch(A) -> dif(A,x) in Ch(A)) and (forall x:el: x in Ch(A) -> dif(A,dif(A,x)) = 0))
func Hc(cx): set
pred Cls(cx, el, el)
def cls_dom: forall A:cx, z h:el: Cls(A,z,h) -> z in Ch(A) and dif(A,z) = 0 and h in Hc(A)
def cls_tot: forall A:cx, z:el: z in Ch(A) and dif(A,z) = 0 -> exists h:el: Cls(A,z,h)
def cls_sur: forall A:cx, h:el: h in Hc(A) -> exists z:el: Cls(A,z,h)
def cls_sub: forall A:cx, z1 z2 h1 h2:el: Cls(A,z1,h1) and Cls(A,z2,h2) -> Cls(A,z1 - z2,h1 - h2)
def cls_zero: forall A:cx, z h:el: Cls(A,z,h) -> (h = 0 <-> (exists y:el: y in Ch(A) and z = dif(A,y)))

lemma cx_sub: forall A:cx: iscx(A) -> subgroup(Ch(A))
by iscx_def
lemma cx_d: forall A:cx, x:el: iscx(A) and x in Ch(A) -> dif(A,x) in Ch(A)
by iscx_def
lemma cx_dd: forall A:cx, x:el: iscx(A) and x in Ch(A) -> dif(A,dif(A,x)) = 0
by iscx_def

lemma cls_zero0: forall A:cx, h:el: iscx(A) and Cls(A,0,h) -> h = 0
proof
  fix A h
  intro cx hc
  have s: subgroup(Ch(A)) by cx_sub cx
  have z: 0 in Ch(A) by sub_zero s
  show h = 0 by cls_zero[A, 0, h] hc z with 0
qed

lemma cls_00: forall A:cx: iscx(A) -> Cls(A,0,0)
proof
  fix A
  intro cx
  have s: subgroup(Ch(A)) by cx_sub cx
  have z: 0 in Ch(A) by sub_zero s
  obtain h:el st hc: Cls(A,0,h) by cls_tot[A, 0] z
  have h0: h = 0 by cls_zero0 cx hc
  show Cls(A,0,0) by hc h0
qed

lemma cls_neg: forall A:cx, z h:el: iscx(A) and Cls(A,z,h) -> Cls(A,-z,-h)
proof
  fix A z h
  intro cx hc
  have c0: Cls(A,0,0) by cls_00 cx
  show Cls(A,-z,-h) by cls_sub[A, 0, z, 0, h] c0 hc
qed

lemma cls_add: forall A:cx, z1 z2 h1 h2:el: iscx(A) and Cls(A,z1,h1) and Cls(A,z2,h2) -> Cls(A,z1 + z2,h1 + h2)
proof
  fix A z1 z2 h1 h2
  intro cx c1 c2
  have n2: Cls(A,-z2,-h2) by cls_neg[A, z2, h2] cx c2
  show by cls_sub[A, z1, -z2, h1, -h2] c1 n2
qed

lemma cls_eq: forall A:cx, z1 z2 h1 h2 y:el: iscx(A) and Cls(A,z1,h1) and Cls(A,z2,h2) and y in Ch(A) and z1 - z2 = dif(A,y) -> h1 = h2
proof
  fix A z1 z2 h1 h2 y
  intro cx c1 c2 hy e
  have c: Cls(A,z1 - z2,h1 - h2) by cls_sub[A, z1, z2, h1, h2] c1 c2
  have z: h1 - h2 = 0 by cls_zero[A, z1 - z2, h1 - h2] c hy e with y
  show by z
qed

lemma cls_fun: forall A:cx, z h1 h2:el: iscx(A) and Cls(A,z,h1) and Cls(A,z,h2) -> h1 = h2
proof
  fix A z h1 h2
  intro cx c1 c2
  have s: subgroup(Ch(A)) by cx_sub cx
  have z0: 0 in Ch(A) by sub_zero s
  show by cls_eq[A, z, z, h1, h2, 0] cx c1 c2 z0
qed

lemma cls_rep: forall A:cx, z1 z2 h:el: Cls(A,z1,h) and Cls(A,z2,h) -> exists y:el: y in Ch(A) and z1 - z2 = dif(A,y)
proof
  fix A z1 z2 h
  intro c1 c2
  have c: Cls(A,z1 - z2,h - h) by cls_sub[A, z1, z2, h, h] c1 c2
  show by cls_zero[A, z1 - z2, h - h] c
qed

lemma cls_natmul: forall A:cx, z h:el: iscx(A) and Cls(A,z,h) -> forall n:int: 0 <= n -> Cls(A,n*z,n*h)
proof
  fix A z h
  intro cx hc
  have c0: Cls(A,0,0) by cls_00 cx
  induct n from 0
  base
  proof
    show Cls(A,0*z,0*h) by c0
  qed
  step
  proof
    fix m
    intro hm ih
    show Cls(A,(m+1)*z,(m+1)*h) by cls_add[A, m*z, z, m*h, h] cx ih hc
  qed
qed

lemma cls_smul: forall A:cx, n:int, z h:el: iscx(A) and Cls(A,z,h) -> Cls(A,n*z,n*h)
proof
  fix A n z h
  intro cx hc
  have nat: forall m:int: 0 <= m -> Cls(A,m*z,m*h) by cls_natmul cx hc
  cases hc2: n <= 0 or 1 <= n by int_discrete[n]
  case h1
  proof
    have a: Cls(A,(-n)*z,(-n)*h) by nat h1
    show by cls_neg[A, (-n)*z, (-n)*h] cx a
  qed
  case h2
  proof
    show by nat h2
  qed
qed

lemma Hc_sub: forall A:cx: iscx(A) -> subgroup(Hc(A))
proof
  fix A
  intro cx
  have c0: Cls(A,0,0) by cls_00 cx
  have z: 0 in Hc(A) by cls_dom[A, 0, 0] c0
  have ad: forall x y:el: x in Hc(A) and y in Hc(A) -> x + y in Hc(A)
  proof
    fix x y
    intro hx hy
    obtain u:el st cu: Cls(A,u,x) by cls_sur hx
    obtain v:el st cv: Cls(A,v,y) by cls_sur hy
    have c: Cls(A,u + v,x + y) by cls_add[A, u, v, x, y] cx cu cv
    show by cls_dom[A, u + v, x + y] c
  qed
  have ng: forall x:el: x in Hc(A) -> -x in Hc(A)
  proof
    fix x
    intro hx
    obtain u:el st cu: Cls(A,u,x) by cls_sur hx
    have c: Cls(A,-u,-x) by cls_neg[A, u, x] cx cu
    show by cls_dom[A, -u, -x] c
  qed
  show subgroup(Hc(A)) by subgroup_def z ad ng
qed

# the grading: C = (+)_n C^n with projections proj(A,n,-), d of degree +1
func Cn(cx, int): set
func proj(cx, int, el): el hom
def proj_in: forall A:cx, n:int, x:el: x in Ch(A) -> proj(A,n,x) in Cn(A,n)
def Cn_ch: forall A:cx, n:int, x:el: x in Cn(A,n) -> x in Ch(A)
def proj_id: forall A:cx, n:int, x:el: x in Cn(A,n) -> proj(A,n,x) = x
def proj_dif: forall A:cx, n:int, x:el: x in Ch(A) -> proj(A,n+1,dif(A,x)) = dif(A,proj(A,n,x))
# H^n(A): classes of cocycles of degree n; Hco(A,n): classes of cocycles
# without a degree n part (sets by comprehension)
func H(cx, int): set
def H_def: forall A:cx, n:int, h:el: h in H(A,n) <-> (exists z:el: z in Cn(A,n) and Cls(A,z,h))
func Hco(cx, int): set
def Hco_def: forall A:cx, n:int, h:el: h in Hco(A,n) <-> (exists z:el: z in Ch(A) and proj(A,n,z) = 0 and Cls(A,z,h))

lemma Cn_sub: forall A:cx, n:int: iscx(A) -> subgroup(Cn(A,n))
proof
  fix A n
  intro cx
  have s: subgroup(Ch(A)) by cx_sub cx
  have z0: 0 in Ch(A) by sub_zero s
  have z: 0 in Cn(A,n) by proj_in[A, n, 0] z0
  have ad: forall x y:el: x in Cn(A,n) and y in Cn(A,n) -> x + y in Cn(A,n)
  proof
    fix x y
    intro hx hy
    have xc: x in Ch(A) by Cn_ch hx
    have yc: y in Ch(A) by Cn_ch hy
    have xy: x + y in Ch(A) by s xc yc
    have m: proj(A,n,x + y) in Cn(A,n) by proj_in[A, n, x + y] xy
    have e: proj(A,n,x + y) = x + y by proj_id[A, n, x] proj_id[A, n, y] hx hy
    show by m e
  qed
  have ng: forall x:el: x in Cn(A,n) -> -x in Cn(A,n)
  proof
    fix x
    intro hx
    have xc: x in Ch(A) by Cn_ch hx
    have nx: -x in Ch(A) by s xc
    have m: proj(A,n,-x) in Cn(A,n) by proj_in[A, n, -x] nx
    have e: proj(A,n,-x) = -x by proj_id[A, n, x] hx
    show by m e
  qed
  show subgroup(Cn(A,n)) by subgroup_def z ad ng
qed

lemma H_in: forall A:cx, n:int, h:el: h in H(A,n) -> h in Hc(A)
proof
  fix A n h
  intro hh
  obtain z:el st zn cz: z in Cn(A,n) and Cls(A,z,h) by H_def hh
  show by cls_dom cz
qed

lemma Hco_in: forall A:cx, n:int, h:el: h in Hco(A,n) -> h in Hc(A)
proof
  fix A n h
  intro hh
  obtain z:el st zc pz cz: z in Ch(A) and proj(A,n,z) = 0 and Cls(A,z,h) by Hco_def hh
  show by cls_dom cz
qed

lemma H_sub: forall A:cx, n:int: iscx(A) -> subgroup(H(A,n))
proof
  fix A n
  intro cx
  have sn: subgroup(Cn(A,n)) by Cn_sub cx
  have c0: Cls(A,0,0) by cls_00 cx
  have z0: 0 in Cn(A,n) by sub_zero sn
  have z: 0 in H(A,n) by H_def z0 c0 with 0
  have ad: forall x y:el: x in H(A,n) and y in H(A,n) -> x + y in H(A,n)
  proof
    fix x y
    intro hx hy
    obtain u:el st un cu: u in Cn(A,n) and Cls(A,u,x) by H_def hx
    obtain v:el st vn cv: v in Cn(A,n) and Cls(A,v,y) by H_def hy
    have uv: u + v in Cn(A,n) by sn un vn
    have c: Cls(A,u + v,x + y) by cls_add[A, u, v, x, y] cx cu cv
    show by H_def uv c with u + v
  qed
  have ng: forall x:el: x in H(A,n) -> -x in H(A,n)
  proof
    fix x
    intro hx
    obtain u:el st un cu: u in Cn(A,n) and Cls(A,u,x) by H_def hx
    have nu: -u in Cn(A,n) by sn un
    have c: Cls(A,-u,-x) by cls_neg[A, u, x] cx cu
    show by H_def nu c with -u
  qed
  show subgroup(H(A,n)) by subgroup_def z ad ng
qed

lemma Hco_sub: forall A:cx, n:int: iscx(A) -> subgroup(Hco(A,n))
proof
  fix A n
  intro cx
  have s: subgroup(Ch(A)) by cx_sub cx
  have c0: Cls(A,0,0) by cls_00 cx
  have z0: 0 in Ch(A) by sub_zero s
  have z: 0 in Hco(A,n) by Hco_def z0 c0 with 0
  have ad: forall x y:el: x in Hco(A,n) and y in Hco(A,n) -> x + y in Hco(A,n)
  proof
    fix x y
    intro hx hy
    obtain u:el st uc pu cu: u in Ch(A) and proj(A,n,u) = 0 and Cls(A,u,x) by Hco_def hx
    obtain v:el st vc pv cv: v in Ch(A) and proj(A,n,v) = 0 and Cls(A,v,y) by Hco_def hy
    have uv: u + v in Ch(A) by s uc vc
    have puv: proj(A,n,u + v) = 0 by pu pv
    have c: Cls(A,u + v,x + y) by cls_add[A, u, v, x, y] cx cu cv
    show by Hco_def uv puv c with u + v
  qed
  have ng: forall x:el: x in Hco(A,n) -> -x in Hco(A,n)
  proof
    fix x
    intro hx
    obtain u:el st uc pu cu: u in Ch(A) and proj(A,n,u) = 0 and Cls(A,u,x) by Hco_def hx
    have nu: -u in Ch(A) by s uc
    have pnu: proj(A,n,-u) = 0 by pu
    have c: Cls(A,-u,-x) by cls_neg[A, u, x] cx cu
    show by Hco_def nu pnu c with -u
  qed
  show subgroup(Hco(A,n)) by subgroup_def z ad ng
qed

# H*(A) = H^n(A) (+) (the rest): from the grading alone
lemma H_summand: forall A:cx, n:int: iscx(A) -> summand(H(A,n),Hc(A))
proof
  fix A n
  intro cx
  have s1: subgroup(H(A,n)) by H_sub cx
  have s2: subgroup(Hc(A)) by Hc_sub cx
  have s3: subgroup(Hco(A,n)) by Hco_sub cx
  have sCh: subgroup(Ch(A)) by cx_sub cx
  have i1: forall x:el: x in H(A,n) -> x in Hc(A) by H_in
  have i2: forall x:el: x in Hco(A,n) -> x in Hc(A) by Hco_in
  have dec: forall x:el: x in Hc(A) -> exists s m:el: s in H(A,n) and m in Hco(A,n) and x = s + m
  proof
    fix x
    intro hx
    obtain z:el st cz: Cls(A,z,x) by cls_sur hx
    have zc: z in Ch(A) and dif(A,z) = 0 by cls_dom cz
    have z1n: proj(A,n,z) in Cn(A,n) by proj_in[A, n, z] zc
    have z1c: proj(A,n,z) in Ch(A) by Cn_ch z1n
    have pd: proj(A,n+1,dif(A,z)) = dif(A,proj(A,n,z)) by proj_dif[A, n, z] zc
    have d1: dif(A,proj(A,n,z)) = 0 by pd zc
    obtain s:el st cs: Cls(A,proj(A,n,z),s) by cls_tot[A, proj(A,n,z)] z1c d1
    have z2c: z - proj(A,n,z) in Ch(A) by sCh zc z1c
    have pp: proj(A,n,proj(A,n,z)) = proj(A,n,z) by proj_id[A, n, proj(A,n,z)] z1n
    have p2: proj(A,n,z - proj(A,n,z)) = 0 by pp
    have d2: dif(A,z - proj(A,n,z)) = 0 by zc d1
    obtain m:el st cm: Cls(A,z - proj(A,n,z),m) by cls_tot[A, z - proj(A,n,z)] z2c d2
    have c3: Cls(A,proj(A,n,z) + (z - proj(A,n,z)),s + m) by cls_add[A, proj(A,n,z), z - proj(A,n,z), s, m] cx cs cm
    have xe: x = s + m by cls_fun[A, z, x, s + m] cx cz c3
    have sH: s in H(A,n) by H_def z1n cs with proj(A,n,z)
    have mH: m in Hco(A,n) by Hco_def z2c p2 cm with z - proj(A,n,z)
    show by sH mH xe with s, m
  qed
  have ind: forall s m:el: s in H(A,n) and m in Hco(A,n) and s + m = 0 -> s = 0
  proof
    fix s m
    intro hs hm h0
    obtain u:el st un cu: u in Cn(A,n) and Cls(A,u,s) by H_def hs
    obtain v:el st vc pv cv: v in Ch(A) and proj(A,n,v) = 0 and Cls(A,v,m) by Hco_def hm
    have c: Cls(A,u + v,s + m) by cls_add[A, u, v, s, m] cx cu cv
    have c0: Cls(A,u + v,0) by c h0
    obtain y:el st yc ye: y in Ch(A) and u + v = dif(A,y) by cls_zero[A, u + v, 0] c0
    have pd: proj(A,n,dif(A,y)) = dif(A,proj(A,n-1,y)) by proj_dif[A, n-1, y] yc
    have pu: proj(A,n,u) = u by proj_id un
    have ue: u = dif(A,proj(A,n-1,y)) by ye pd pu pv
    have pn: proj(A,n-1,y) in Cn(A,n-1) by proj_in[A, n-1, y] yc
    have pc: proj(A,n-1,y) in Ch(A) by Cn_ch pn
    show s = 0 by cls_zero[A, u, s] cu pc ue with proj(A,n-1,y)
  qed
  show summand(H(A,n),Hc(A)) by summand_def s1 s2 s3 i1 i2 dec ind with Hco(A,n)
qed

# ---- chain maps, short exact sequences, the long exact sequence ----
pred chmap(cx, cx, hom)
def chmap_def: forall A B:cx, f:hom: chmap(A,B,f) <-> (iscx(A) and iscx(B) and (forall x:el: x in Ch(A) -> ap(f,x) in Ch(B)) and (forall x:el: x in Ch(A) -> ap(f,dif(A,x)) = dif(B,ap(f,x))))
# 0 -> A -f-> B -g-> C -> 0
pred isses(cx, cx, cx, hom, hom)
def isses_def: forall A B C:cx, f g:hom: isses(A,B,C,f,g) <-> (chmap(A,B,f) and chmap(B,C,g) and (forall x:el: x in Ch(A) and ap(f,x) = 0 -> x = 0) and (forall w:el: w in Ch(C) -> exists b:el: b in Ch(B) and w = ap(g,b)) and (forall b:el: b in Ch(B) and ap(g,b) = 0 -> exists x:el: x in Ch(A) and b = ap(f,x)) and (forall x:el: x in Ch(A) -> ap(g,ap(f,x)) = 0))
# the induced map H(f), as a relation on classes; the connecting map:
# lift a cocycle w of C to b in B, apply d, pull back along f, take the class
pred HM(cx, cx, hom, el, el)
def HM_def: forall A B:cx, f:hom, h h2:el: HM(A,B,f,h,h2) <-> (exists z:el: Cls(A,z,h) and Cls(B,ap(f,z),h2))
pred Del(cx, cx, cx, hom, hom, el, el)
def Del_def: forall A B C:cx, f g:hom, e h:el: Del(A,B,C,f,g,e,h) <-> (exists w b z:el: Cls(C,w,e) and b in Ch(B) and w = ap(g,b) and z in Ch(A) and ap(f,z) = dif(B,b) and Cls(A,z,h))

lemma cm_A: forall A B:cx, f:hom: chmap(A,B,f) -> iscx(A)
by chmap_def
lemma cm_B: forall A B:cx, f:hom: chmap(A,B,f) -> iscx(B)
by chmap_def
lemma cm_in: forall A B:cx, f:hom, x:el: chmap(A,B,f) and x in Ch(A) -> ap(f,x) in Ch(B)
by chmap_def
lemma cm_d: forall A B:cx, f:hom, x:el: chmap(A,B,f) and x in Ch(A) -> ap(f,dif(A,x)) = dif(B,ap(f,x))
by chmap_def
lemma ss_f: forall A B C:cx, f g:hom: isses(A,B,C,f,g) -> chmap(A,B,f)
by isses_def
lemma ss_g: forall A B C:cx, f g:hom: isses(A,B,C,f,g) -> chmap(B,C,g)
by isses_def
lemma ss_inj: forall A B C:cx, f g:hom, x:el: isses(A,B,C,f,g) and x in Ch(A) and ap(f,x) = 0 -> x = 0
by isses_def
lemma ss_sur: forall A B C:cx, f g:hom, w:el: isses(A,B,C,f,g) and w in Ch(C) -> exists b:el: b in Ch(B) and w = ap(g,b)
by isses_def
lemma ss_ker: forall A B C:cx, f g:hom, b:el: isses(A,B,C,f,g) and b in Ch(B) and ap(g,b) = 0 -> exists x:el: x in Ch(A) and b = ap(f,x)
by isses_def
lemma ss_gf: forall A B C:cx, f g:hom, x:el: isses(A,B,C,f,g) and x in Ch(A) -> ap(g,ap(f,x)) = 0
by isses_def

lemma HM_dom: forall A B:cx, f:hom, h h2:el: HM(A,B,f,h,h2) -> h in Hc(A) and h2 in Hc(B)
proof
  fix A B f h h2
  intro m
  obtain z:el st cz cf: Cls(A,z,h) and Cls(B,ap(f,z),h2) by HM_def m
  have a: h in Hc(A) by cls_dom cz
  have b: h2 in Hc(B) by cls_dom cf
  show by a b
qed

lemma Del_dom: forall A B C:cx, f g:hom, e h:el: Del(A,B,C,f,g,e,h) -> e in Hc(C) and h in Hc(A)
proof
  fix A B C f g e h
  intro dl
  obtain w b z:el st cw bc bw zc zf cz: Cls(C,w,e) and b in Ch(B) and w = ap(g,b) and z in Ch(A) and ap(f,z) = dif(B,b) and Cls(A,z,h) by Del_def dl
  have a: e in Hc(C) by cls_dom cw
  have c: h in Hc(A) by cls_dom cz
  show by a c
qed

# H(f) is well defined: total and single valued
lemma HM_tot: forall A B:cx, f:hom, h:el: chmap(A,B,f) and h in Hc(A) -> exists h2:el: HM(A,B,f,h,h2)
proof
  fix A B f h
  intro cm hh
  obtain z:el st cz: Cls(A,z,h) by cls_sur hh
  have zc: z in Ch(A) and dif(A,z) = 0 by cls_dom cz
  have fz: ap(f,z) in Ch(B) by cm_in[A, B, f, z] cm zc
  have fd: ap(f,dif(A,z)) = dif(B,ap(f,z)) by cm_d[A, B, f, z] cm zc
  have d0: dif(B,ap(f,z)) = 0 by fd zc
  obtain h2:el st c2: Cls(B,ap(f,z),h2) by cls_tot[B, ap(f,z)] fz d0
  have m: HM(A,B,f,h,h2) by HM_def cz c2 with z
  show by m with h2
qed

lemma HM_fun: forall A B:cx, f:hom, h h1 h2:el: chmap(A,B,f) and HM(A,B,f,h,h1) and HM(A,B,f,h,h2) -> h1 = h2
proof
  fix A B f h h1 h2
  intro cm m1 m2
  have cxB: iscx(B) by cm_B cm
  obtain z1:el st c1 d1: Cls(A,z1,h) and Cls(B,ap(f,z1),h1) by HM_def m1
  obtain z2:el st c2 d2: Cls(A,z2,h) and Cls(B,ap(f,z2),h2) by HM_def m2
  obtain y:el st yc ye: y in Ch(A) and z1 - z2 = dif(A,y) by cls_rep[A, z1, z2, h] c1 c2
  have fy: ap(f,y) in Ch(B) by cm_in[A, B, f, y] cm yc
  have fd: ap(f,dif(A,y)) = dif(B,ap(f,y)) by cm_d[A, B, f, y] cm yc
  have e: ap(f,z1) - ap(f,z2) = dif(B,ap(f,y)) by ye fd
  show by cls_eq[B, ap(f,z1), ap(f,z2), h1, h2, ap(f,y)] cxB d1 d2 fy e
qed

# the connecting map is well defined: total and single valued
lemma Del_tot: forall A B C:cx, f g:hom, e:el: isses(A,B,C,f,g) and e in Hc(C) -> exists h:el: Del(A,B,C,f,g,e,h)
proof
  fix A B C f g e
  intro ss he
  have cf: chmap(A,B,f) by ss_f ss
  have cg: chmap(B,C,g) by ss_g ss
  have cxA: iscx(A) by cm_A cf
  have cxB: iscx(B) by cm_B cf
  obtain w:el st cw: Cls(C,w,e) by cls_sur he
  have wc: w in Ch(C) and dif(C,w) = 0 by cls_dom cw
  obtain b:el st bc bw: b in Ch(B) and w = ap(g,b) by ss_sur[A, B, C, f, g, w] ss wc
  have db: dif(B,b) in Ch(B) by cx_d cxB bc
  have gd: ap(g,dif(B,b)) = dif(C,ap(g,b)) by cm_d[B, C, g, b] cg bc
  have gdb: ap(g,dif(B,b)) = 0 by gd bw wc
  obtain z:el st zc zf: z in Ch(A) and dif(B,b) = ap(f,z) by ss_ker[A, B, C, f, g, dif(B,b)] ss db gdb
  have dz: dif(A,z) in Ch(A) by cx_d cxA zc
  have fdz: ap(f,dif(A,z)) = dif(B,ap(f,z)) by cm_d[A, B, f, z] cf zc
  have ddb: dif(B,dif(B,b)) = 0 by cx_dd cxB bc
  have f0: ap(f,dif(A,z)) = 0 by fdz zf ddb
  have z0: dif(A,z) = 0 by ss_inj[A, B, C, f, g, dif(A,z)] ss dz f0
  obtain h:el st ch: Cls(A,z,h) by cls_tot[A, z] zc z0
  have zf2: ap(f,z) = dif(B,b) by zf
  have dl: Del(A,B,C,f,g,e,h) by Del_def cw bc bw zc zf2 ch with w, b, z
  show by dl with h
qed

lemma Del_fun: forall A B C:cx, f g:hom, e h1 h2:el: isses(A,B,C,f,g) and Del(A,B,C,f,g,e,h1) and Del(A,B,C,f,g,e,h2) -> h1 = h2
proof
  fix A B C f g e h1 h2
  intro ss d1 d2
  have cf: chmap(A,B,f) by ss_f ss
  have cg: chmap(B,C,g) by ss_g ss
  have cxA: iscx(A) by cm_A cf
  have cxB: iscx(B) by cm_B cf
  have sA: subgroup(Ch(A)) by cx_sub cxA
  have sB: subgroup(Ch(B)) by cx_sub cxB
  obtain w1 b1 z1:el st cw1 b1c bw1 z1c zf1 ch1: Cls(C,w1,e) and b1 in Ch(B) and w1 = ap(g,b1) and z1 in Ch(A) and ap(f,z1) = dif(B,b1) and Cls(A,z1,h1) by Del_def d1
  obtain w2 b2 z2:el st cw2 b2c bw2 z2c zf2 ch2: Cls(C,w2,e) and b2 in Ch(B) and w2 = ap(g,b2) and z2 in Ch(A) and ap(f,z2) = dif(B,b2) and Cls(A,z2,h2) by Del_def d2
  obtain u:el st uc ue: u in Ch(C) and w1 - w2 = dif(C,u) by cls_rep[C, w1, w2, e] cw1 cw2
  obtain v:el st vc vu: v in Ch(B) and u = ap(g,v) by ss_sur[A, B, C, f, g, u] ss uc
  have dv: dif(B,v) in Ch(B) by cx_d cxB vc
  have gdv: ap(g,dif(B,v)) = dif(C,ap(g,v)) by cm_d[B, C, g, v] cg vc
  have t1: b1 - b2 - dif(B,v) in Ch(B) by sB b1c b2c dv
  have t0: ap(g,b1 - b2 - dif(B,v)) = 0 by bw1 bw2 ue vu gdv
  obtain t:el st tc tf: t in Ch(A) and b1 - b2 - dif(B,v) = ap(f,t) by ss_ker[A, B, C, f, g, b1 - b2 - dif(B,v)] ss t1 t0
  have fdt: ap(f,dif(A,t)) = dif(B,ap(f,t)) by cm_d[A, B, f, t] cf tc
  have ddv: dif(B,dif(B,v)) = 0 by cx_dd cxB vc
  have dt: dif(A,t) in Ch(A) by cx_d cxA tc
  have zz: z1 - z2 - dif(A,t) in Ch(A) by sA z1c z2c dt
  have f0: ap(f,z1 - z2 - dif(A,t)) = 0 by zf1 zf2 tf fdt ddv
  have z0: z1 - z2 - dif(A,t) = 0 by ss_inj[A, B, C, f, g, z1 - z2 - dif(A,t)] ss zz f0
  have e2: z1 - z2 = dif(A,t) by z0
  show by cls_eq[A, z1, z2, h1, h2, t] cxA ch1 ch2 tc e2
qed

# exactness of H(A) -> H(B) -> H(C) -> H(A) at all three places
lemma les_fg: forall A B C:cx, f g:hom, h h2 h3:el: isses(A,B,C,f,g) and HM(A,B,f,h,h2) and HM(B,C,g,h2,h3) -> h3 = 0
proof
  fix A B C f g h h2 h3
  intro ss m1 m2
  have cg: chmap(B,C,g) by ss_g ss
  have cxC: iscx(C) by cm_B cg
  obtain z:el st cz cfz: Cls(A,z,h) and Cls(B,ap(f,z),h2) by HM_def m1
  have zc: z in Ch(A) by cls_dom cz
  have gf: ap(g,ap(f,z)) = 0 by ss_gf[A, B, C, f, g, z] ss zc
  have c0: Cls(C,0,0) by cls_00 cxC
  have c1: Cls(C,ap(g,ap(f,z)),0) by c0 gf
  have m3: HM(B,C,g,h2,0) by HM_def cfz c1 with ap(f,z)
  show by HM_fun[B, C, g, h2, h3, 0] cg m2 m3
qed

lemma les_kerg: forall A B C:cx, f g:hom, h2:el: isses(A,B,C,f,g) and h2 in Hc(B) and HM(B,C,g,h2,0) -> exists h:el: HM(A,B,f,h,h2)
proof
  fix A B C f g h2
  intro ss hh m
  have cf: chmap(A,B,f) by ss_f ss
  have cg: chmap(B,C,g) by ss_g ss
  have cxA: iscx(A) by cm_A cf
  have cxB: iscx(B) by cm_B cf
  have sB: subgroup(Ch(B)) by cx_sub cxB
  obtain z:el st cz cgz: Cls(B,z,h2) and Cls(C,ap(g,z),0) by HM_def m
  have zc: z in Ch(B) and dif(B,z) = 0 by cls_dom cz
  obtain u:el st uc ue: u in Ch(C) and ap(g,z) = dif(C,u) by cls_zero[C, ap(g,z), 0] cgz
  obtain v:el st vc vu: v in Ch(B) and u = ap(g,v) by ss_sur[A, B, C, f, g, u] ss uc
  have gdv: ap(g,dif(B,v)) = dif(C,ap(g,v)) by cm_d[B, C, g, v] cg vc
  have dv: dif(B,v) in Ch(B) by cx_d cxB vc
  have t1: z - dif(B,v) in Ch(B) by sB zc dv
  have t0: ap(g,z - dif(B,v)) = 0 by ue vu gdv
  obtain x:el st xc xf: x in Ch(A) and z - dif(B,v) = ap(f,x) by ss_ker[A, B, C, f, g, z - dif(B,v)] ss t1 t0
  have fdx: ap(f,dif(A,x)) = dif(B,ap(f,x)) by cm_d[A, B, f, x] cf xc
  have ddv: dif(B,dif(B,v)) = 0 by cx_dd cxB vc
  have dx: dif(A,x) in Ch(A) by cx_d cxA xc
  have f0: ap(f,dif(A,x)) = 0 by fdx xf zc ddv
  have x0: dif(A,x) = 0 by ss_inj[A, B, C, f, g, dif(A,x)] ss dx f0
  obtain h:el st ch: Cls(A,x,h) by cls_tot[A, x] xc x0
  have fx: ap(f,x) in Ch(B) by cm_in[A, B, f, x] cf xc
  have dfx: dif(B,ap(f,x)) = 0 by fdx x0
  obtain h3:el st c3: Cls(B,ap(f,x),h3) by cls_tot[B, ap(f,x)] fx dfx
  have e: z - ap(f,x) = dif(B,v) by xf
  have hh3: h2 = h3 by cls_eq[B, z, ap(f,x), h2, h3, v] cxB cz c3 vc e
  have m2: HM(A,B,f,h,h3) by HM_def ch c3 with x
  have m3: HM(A,B,f,h,h2) by m2 hh3
  show by m3 with h
qed

lemma les_gdel: forall A B C:cx, f g:hom, h2 e h:el: isses(A,B,C,f,g) and HM(B,C,g,h2,e) and Del(A,B,C,f,g,e,h) -> h = 0
proof
  fix A B C f g h2 e h
  intro ss m dl
  have cf: chmap(A,B,f) by ss_f ss
  have cxA: iscx(A) by cm_A cf
  obtain z:el st cz cgz: Cls(B,z,h2) and Cls(C,ap(g,z),e) by HM_def m
  have zc: z in Ch(B) and dif(B,z) = 0 by cls_dom cz
  have sA: subgroup(Ch(A)) by cx_sub cxA
  have z0A: 0 in Ch(A) by sub_zero sA
  have c0: Cls(A,0,0) by cls_00 cxA
  have f0: ap(f,0) = dif(B,z) by zc
  have d0: Del(A,B,C,f,g,e,0) by Del_def cgz zc z0A f0 c0 with ap(g,z), z, 0
  show by Del_fun[A, B, C, f, g, e, h, 0] ss dl d0
qed

lemma les_kerdel: forall A B C:cx, f g:hom, e:el: isses(A,B,C,f,g) and Del(A,B,C,f,g,e,0) -> exists h2:el: HM(B,C,g,h2,e)
proof
  fix A B C f g e
  intro ss dl
  have cf: chmap(A,B,f) by ss_f ss
  have cg: chmap(B,C,g) by ss_g ss
  have cxB: iscx(B) by cm_B cf
  have sB: subgroup(Ch(B)) by cx_sub cxB
  obtain w b z:el st cw bc bw zc zf cz: Cls(C,w,e) and b in Ch(B) and w = ap(g,b) and z in Ch(A) and ap(f,z) = dif(B,b) and Cls(A,z,0) by Del_def dl
  obtain t:el st tc te: t in Ch(A) and z = dif(A,t) by cls_zero[A, z, 0] cz
  have ft: ap(f,t) in Ch(B) by cm_in[A, B, f, t] cf tc
  have fdt: ap(f,dif(A,t)) = dif(B,ap(f,t)) by cm_d[A, B, f, t] cf tc
  have c1: b - ap(f,t) in Ch(B) by sB bc ft
  have d1: dif(B,b - ap(f,t)) = 0 by zf te fdt
  obtain h2:el st c2: Cls(B,b - ap(f,t),h2) by cls_tot[B, b - ap(f,t)] c1 d1
  have gt: ap(g,ap(f,t)) = 0 by ss_gf[A, B, C, f, g, t] ss tc
  have c3: Cls(C,ap(g,b - ap(f,t)),e) by cw bw gt
  have m: HM(B,C,g,h2,e) by HM_def c2 c3 with b - ap(f,t)
  show by m with h2
qed

lemma les_delf: forall A B C:cx, f g:hom, e h h2:el: isses(A,B,C,f,g) and Del(A,B,C,f,g,e,h) and HM(A,B,f,h,h2) -> h2 = 0
proof
  fix A B C f g e h h2
  intro ss dl m
  have cf: chmap(A,B,f) by ss_f ss
  have cxB: iscx(B) by cm_B cf
  obtain w b z:el st cw bc bw zc zf cz: Cls(C,w,e) and b in Ch(B) and w = ap(g,b) and z in Ch(A) and ap(f,z) = dif(B,b) and Cls(A,z,h) by Del_def dl
  have fz: ap(f,z) in Ch(B) by cm_in[A, B, f, z] cf zc
  have ddb: dif(B,dif(B,b)) = 0 by cx_dd cxB bc
  have dfz: dif(B,ap(f,z)) = 0 by zf ddb
  obtain h3:el st c3: Cls(B,ap(f,z),h3) by cls_tot[B, ap(f,z)] fz dfz
  have h30: h3 = 0 by cls_zero[B, ap(f,z), h3] c3 bc zf with b
  have c4: Cls(B,ap(f,z),0) by c3 h30
  have m2: HM(A,B,f,h,0) by HM_def cz c4 with z
  show by HM_fun[A, B, f, h, h2, 0] cf m m2
qed

lemma les_kerf: forall A B C:cx, f g:hom, h:el: isses(A,B,C,f,g) and h in Hc(A) and HM(A,B,f,h,0) -> exists e:el: Del(A,B,C,f,g,e,h)
proof
  fix A B C f g h
  intro ss hh m
  have cg: chmap(B,C,g) by ss_g ss
  obtain z:el st cz c0: Cls(A,z,h) and Cls(B,ap(f,z),0) by HM_def m
  have zc: z in Ch(A) by cls_dom cz
  obtain b:el st bc bf: b in Ch(B) and ap(f,z) = dif(B,b) by cls_zero[B, ap(f,z), 0] c0
  have gb: ap(g,b) in Ch(C) by cm_in[B, C, g, b] cg bc
  have gdb: ap(g,dif(B,b)) = dif(C,ap(g,b)) by cm_d[B, C, g, b] cg bc
  have gf: ap(g,ap(f,z)) = 0 by ss_gf[A, B, C, f, g, z] ss zc
  have dgb: dif(C,ap(g,b)) = 0 by gdb bf gf
  obtain e:el st ce: Cls(C,ap(g,b),e) by cls_tot[C, ap(g,b)] gb dgb
  have dl: Del(A,B,C,f,g,e,h) by Del_def ce bc zc bf cz with ap(g,b), b, z
  show by dl with e
qed

# the long exact sequence of a short exact sequence of complexes
theorem les_exact: forall A B C:cx, f g:hom: isses(A,B,C,f,g) -> (forall h h2 h3:el: HM(A,B,f,h,h2) and HM(B,C,g,h2,h3) -> h3 = 0) and (forall h2:el: h2 in Hc(B) and HM(B,C,g,h2,0) -> exists h:el: HM(A,B,f,h,h2)) and (forall h2 e h:el: HM(B,C,g,h2,e) and Del(A,B,C,f,g,e,h) -> h = 0) and (forall e:el: Del(A,B,C,f,g,e,0) -> exists h2:el: HM(B,C,g,h2,e)) and (forall e h h2:el: Del(A,B,C,f,g,e,h) and HM(A,B,f,h,h2) -> h2 = 0) and (forall h:el: h in Hc(A) and HM(A,B,f,h,0) -> exists e:el: Del(A,B,C,f,g,e,h))
proof
  fix A B C f g
  intro ss
  have a1: forall h h2 h3:el: HM(A,B,f,h,h2) and HM(B,C,g,h2,h3) -> h3 = 0 by les_fg ss
  have a2: forall h2:el: h2 in Hc(B) and HM(B,C,g,h2,0) -> exists h:el: HM(A,B,f,h,h2) by les_kerg ss
  have a3: forall h2 e h:el: HM(B,C,g,h2,e) and Del(A,B,C,f,g,e,h) -> h = 0 by les_gdel ss
  have a4: forall e:el: Del(A,B,C,f,g,e,0) -> exists h2:el: HM(B,C,g,h2,e) by les_kerdel ss
  have a5: forall e h h2:el: Del(A,B,C,f,g,e,h) and HM(A,B,f,h,h2) -> h2 = 0 by les_delf ss
  have a6: forall h:el: h in Hc(A) and HM(A,B,f,h,0) -> exists e:el: Del(A,B,C,f,g,e,h) by les_kerf ss
  show by a1 a2 a3 a4 a5 a6
qed

# ---- the Bockstein spectral sequence ----
# For a cochain complex C and an integer p: red(C,p) = C/pC with the
# projection rh(C,p) (the three red_ lines define the quotient complex),
# the short exact sequence 0 -> C -p-> C -> C/pC -> 0 (exact because C is
# torsion-free: a lemma), and its long exact sequence as the exact couple
# bock(C,p): D = H(C), E = H(C/p), i = H(p), j = H(rho), k = delta.
func red(cx, int): cx
func rh(cx, int): hom
func mulp(int): hom
def mulp_def: forall p:int, x:el: ap(mulp(p),x) = p*x
def red_ch: forall C:cx, p:int, w:el: w in Ch(red(C,p)) <-> (exists x:el: x in Ch(C) and w = ap(rh(C,p),x))
def red_ker: forall C:cx, p:int, x:el: x in Ch(C) -> (ap(rh(C,p),x) = 0 <-> (exists y:el: y in Ch(C) and x = p*y))
def red_dif: forall C:cx, p:int, x:el: x in Ch(C) -> dif(red(C,p),ap(rh(C,p),x)) = ap(rh(C,p),dif(C,x))
def red_Cn: forall C:cx, p n:int, w:el: w in Cn(red(C,p),n) <-> (exists x:el: x in Cn(C,n) and w = ap(rh(C,p),x))
# of finite type: here, a complex whose cochains are torsion-free (what
# free of finite rank in each degree gives; finite generation is not used)
pred fintype(cx)
def fintype_def: forall C:cx: fintype(C) <-> (iscx(C) and (forall m:int, x:el: x in Ch(C) and 1 <= m and m*x = 0 -> x = 0))
pred prime(int)
def prime_def: forall p:int: prime(p) <-> (2 <= p and (forall d q:int: 1 <= d and p = d*q -> d = 1 or d = p))
func bock(cx, int): cpl
def bock_D: forall C:cx, p:int, x:el: x in D(bock(C,p)) <-> x in Hc(C)
def bock_E: forall C:cx, p:int, x:el: x in E(bock(C,p)) <-> x in Hc(red(C,p))
def bock_i: forall C:cx, p:int, h h2:el: HM(C,C,mulp(p),h,h2) -> i(bock(C,p),h) = h2
def bock_j: forall C:cx, p:int, h h2:el: HM(C,red(C,p),rh(C,p),h,h2) -> j(bock(C,p),h) = h2
def bock_k: forall C:cx, p:int, e h:el: Del(C,C,red(C,p),mulp(p),rh(C,p),e,h) -> k(bock(C,p),e) = h
def bock_Edeg: forall C:cx, p m:int, e:el: Edeg(bock(C,p),m,e) <-> e in H(red(C,p),m)

lemma prime_ge2: forall p:int: prime(p) -> 2 <= p
by prime_def

lemma red_cx: forall C:cx, p:int: iscx(C) -> iscx(red(C,p))
proof
  fix C p
  intro cx
  have sC: subgroup(Ch(C)) by cx_sub cx
  have z: 0 in Ch(C) by sub_zero sC
  have z0: 0 in Ch(red(C,p)) by red_ch z with 0
  have ad: forall u v:el: u in Ch(red(C,p)) and v in Ch(red(C,p)) -> u + v in Ch(red(C,p))
  proof
    fix u v
    intro hu hv
    obtain x:el st xc xu: x in Ch(C) and u = ap(rh(C,p),x) by red_ch hu
    obtain y:el st yc yv: y in Ch(C) and v = ap(rh(C,p),y) by red_ch hv
    have s: x + y in Ch(C) by sC xc yc
    show by red_ch s xu yv with x + y
  qed
  have ng: forall u:el: u in Ch(red(C,p)) -> -u in Ch(red(C,p))
  proof
    fix u
    intro hu
    obtain x:el st xc xu: x in Ch(C) and u = ap(rh(C,p),x) by red_ch hu
    have s: -x in Ch(C) by sC xc
    show by red_ch s xu with -x
  qed
  have sub: subgroup(Ch(red(C,p))) by subgroup_def z0 ad ng
  have dd1: forall u:el: u in Ch(red(C,p)) -> dif(red(C,p),u) in Ch(red(C,p))
  proof
    fix u
    intro hu
    obtain x:el st xc xu: x in Ch(C) and u = ap(rh(C,p),x) by red_ch hu
    have dx: dif(C,x) in Ch(C) by cx_d cx xc
    have e: dif(red(C,p),ap(rh(C,p),x)) = ap(rh(C,p),dif(C,x)) by red_dif xc
    have m: ap(rh(C,p),dif(C,x)) in Ch(red(C,p)) by red_ch dx with dif(C,x)
    have e2: dif(red(C,p),u) = ap(rh(C,p),dif(C,x)) by e xu
    show by m e2
  qed
  have dd2: forall u:el: u in Ch(red(C,p)) -> dif(red(C,p),dif(red(C,p),u)) = 0
  proof
    fix u
    intro hu
    obtain x:el st xc xu: x in Ch(C) and u = ap(rh(C,p),x) by red_ch hu
    have dx: dif(C,x) in Ch(C) by cx_d cx xc
    have e1: dif(red(C,p),ap(rh(C,p),x)) = ap(rh(C,p),dif(C,x)) by red_dif xc
    have e0: dif(red(C,p),u) = ap(rh(C,p),dif(C,x)) by e1 xu
    have e2: dif(red(C,p),ap(rh(C,p),dif(C,x))) = ap(rh(C,p),dif(C,dif(C,x))) by red_dif dx
    have e3: dif(C,dif(C,x)) = 0 by cx_dd cx xc
    show by e0 e2 e3
  qed
  show iscx(red(C,p)) by iscx_def sub dd1 dd2
qed

# 0 -> C -p-> C -> C/pC -> 0 is a short exact sequence of complexes
lemma isses_bock: forall C:cx, p:int: fintype(C) and prime(p) -> isses(C,C,red(C,p),mulp(p),rh(C,p))
proof
  fix C p
  intro hf hp
  have cx: iscx(C) by fintype_def hf
  have tf: forall m:int, x:el: x in Ch(C) and 1 <= m and m*x = 0 -> x = 0 by fintype_def hf
  have p2: 2 <= p by prime_ge2 hp
  have cr: iscx(red(C,p)) by red_cx cx
  have sC: subgroup(Ch(C)) by cx_sub cx
  have cf: chmap(C,C,mulp(p))
  proof
    have a: forall x:el: x in Ch(C) -> ap(mulp(p),x) in Ch(C)
    proof
      fix x
      intro hx
      have m: p*x in Ch(C) by sC hx
      show by m mulp_def[p, x]
    qed
    have b: forall x:el: x in Ch(C) -> ap(mulp(p),dif(C,x)) = dif(C,ap(mulp(p),x))
    proof
      fix x
      intro hx
      show by mulp_def[p, dif(C,x)] mulp_def[p, x]
    qed
    show by chmap_def cx a b
  qed
  have cg: chmap(C,red(C,p),rh(C,p))
  proof
    have a: forall x:el: x in Ch(C) -> ap(rh(C,p),x) in Ch(red(C,p))
    proof
      fix x
      intro hx
      show by red_ch hx with x
    qed
    have b: forall x:el: x in Ch(C) -> ap(rh(C,p),dif(C,x)) = dif(red(C,p),ap(rh(C,p),x))
    proof
      fix x
      intro hx
      show by red_dif hx
    qed
    show by chmap_def cx cr a b
  qed
  have inj: forall x:el: x in Ch(C) and ap(mulp(p),x) = 0 -> x = 0
  proof
    fix x
    intro hx h0
    have px: p*x = 0 by h0 mulp_def[p, x]
    have p1: 1 <= p by p2
    show by tf[p, x] hx p1 px
  qed
  have sur: forall w:el: w in Ch(red(C,p)) -> exists b:el: b in Ch(C) and w = ap(rh(C,p),b)
  proof
    fix w
    intro hw
    show by red_ch hw
  qed
  have ker: forall b:el: b in Ch(C) and ap(rh(C,p),b) = 0 -> exists x:el: x in Ch(C) and b = ap(mulp(p),x)
  proof
    fix b
    intro hb h0
    obtain y:el st yc yb: y in Ch(C) and b = p*y by red_ker[C, p, b] hb h0
    show by yc yb mulp_def[p, y] with y
  qed
  have gf: forall x:el: x in Ch(C) -> ap(rh(C,p),ap(mulp(p),x)) = 0
  proof
    fix x
    intro hx
    have m: p*x in Ch(C) by sC hx
    have r0: ap(rh(C,p),p*x) = 0 by red_ker[C, p, p*x] m hx with x
    show by r0 mulp_def[p, x]
  qed
  show by isses_def cf cg inj sur ker gf
qed

lemma sub_eqv: forall S T:set: subgroup(T) and (forall x:el: x in S <-> x in T) -> subgroup(S)
proof
  fix S T
  intro hT eq
  have z0: 0 in T by sub_zero hT
  have z: 0 in S by eq z0
  have ad: forall x y:el: x in S and y in S -> x + y in S
  proof
    fix x y
    intro hx hy
    have xT: x in T by eq hx
    have yT: y in T by eq hy
    have s: x + y in T by hT xT yT
    show by eq s
  qed
  have ng: forall x:el: x in S -> -x in S
  proof
    fix x
    intro hx
    have xT: x in T by eq hx
    have s: -x in T by hT xT
    show by eq s
  qed
  show subgroup(S) by subgroup_def z ad ng
qed

# the Bockstein couple is an exact couple: the long exact sequence
theorem bock_couple: forall C:cx, p:int: fintype(C) and prime(p) -> excouple(bock(C,p))
proof
  fix C p
  intro hf hp
  have ss: isses(C,C,red(C,p),mulp(p),rh(C,p)) by isses_bock hf hp
  have cf: chmap(C,C,mulp(p)) by ss_f ss
  have cg: chmap(C,red(C,p),rh(C,p)) by ss_g ss
  have cx: iscx(C) by cm_A cf
  have cr: iscx(red(C,p)) by cm_B cg
  have sH: subgroup(Hc(C)) by Hc_sub cx
  have sR: subgroup(Hc(red(C,p))) by Hc_sub cr
  have sD: subgroup(D(bock(C,p))) by sub_eqv[D(bock(C,p)), Hc(C)] sH bock_D[C, p]
  have sE: subgroup(E(bock(C,p))) by sub_eqv[E(bock(C,p)), Hc(red(C,p))] sR bock_E[C, p]
  have iD: forall x:el: x in D(bock(C,p)) -> i(bock(C,p),x) in D(bock(C,p))
  proof
    fix x
    intro hx
    have xH: x in Hc(C) by bock_D hx
    obtain h:el st m: HM(C,C,mulp(p),x,h) by HM_tot[C, C, mulp(p), x] cf xH
    have e: i(bock(C,p),x) = h by bock_i m
    have hH: h in Hc(C) by HM_dom m
    have hD: h in D(bock(C,p)) by bock_D hH
    show by hD e
  qed
  have jE: forall x:el: x in D(bock(C,p)) -> j(bock(C,p),x) in E(bock(C,p))
  proof
    fix x
    intro hx
    have xH: x in Hc(C) by bock_D hx
    obtain h:el st m: HM(C,red(C,p),rh(C,p),x,h) by HM_tot[C, red(C,p), rh(C,p), x] cg xH
    have e: j(bock(C,p),x) = h by bock_j m
    have hH: h in Hc(red(C,p)) by HM_dom m
    have hE: h in E(bock(C,p)) by bock_E hH
    show by hE e
  qed
  have kD: forall e:el: e in E(bock(C,p)) -> k(bock(C,p),e) in D(bock(C,p))
  proof
    fix e
    intro he
    have eH: e in Hc(red(C,p)) by bock_E he
    obtain h:el st dl: Del(C,C,red(C,p),mulp(p),rh(C,p),e,h) by Del_tot[C, C, red(C,p), mulp(p), rh(C,p), e] ss eH
    have ke: k(bock(C,p),e) = h by bock_k dl
    have hH: h in Hc(C) by Del_dom dl
    have hD: h in D(bock(C,p)) by bock_D hH
    show by hD ke
  qed
  # ker j = im i
  have kerj: forall x:el: x in D(bock(C,p)) and j(bock(C,p),x) = 0 -> exists y:el: y in D(bock(C,p)) and x = i(bock(C,p),y)
  proof
    fix x
    intro hx h0
    have xH: x in Hc(C) by bock_D hx
    obtain h:el st m: HM(C,red(C,p),rh(C,p),x,h) by HM_tot[C, red(C,p), rh(C,p), x] cg xH
    have e: j(bock(C,p),x) = h by bock_j m
    have m0: HM(C,red(C,p),rh(C,p),x,0) by m e h0
    obtain y:el st my: HM(C,C,mulp(p),y,x) by les_kerg[C, C, red(C,p), mulp(p), rh(C,p), x] ss xH m0
    have yH: y in Hc(C) by HM_dom my
    have yD: y in D(bock(C,p)) by bock_D yH
    have iy: i(bock(C,p),y) = x by bock_i my
    show by yD iy with y
  qed
  have ji: forall x:el: x in D(bock(C,p)) -> j(bock(C,p),i(bock(C,p),x)) = 0
  proof
    fix x
    intro hx
    have xH: x in Hc(C) by bock_D hx
    obtain h:el st m: HM(C,C,mulp(p),x,h) by HM_tot[C, C, mulp(p), x] cf xH
    have e: i(bock(C,p),x) = h by bock_i m
    have hH: h in Hc(C) by HM_dom m
    obtain h3:el st m3: HM(C,red(C,p),rh(C,p),h,h3) by HM_tot[C, red(C,p), rh(C,p), h] cg hH
    have e3: j(bock(C,p),h) = h3 by bock_j m3
    have z: h3 = 0 by les_fg[C, C, red(C,p), mulp(p), rh(C,p), x, h, h3] ss m m3
    show by e e3 z
  qed
  # ker k = im j
  have kerk: forall e:el: e in E(bock(C,p)) and k(bock(C,p),e) = 0 -> exists x:el: x in D(bock(C,p)) and e = j(bock(C,p),x)
  proof
    fix e
    intro he h0
    have eH: e in Hc(red(C,p)) by bock_E he
    obtain h:el st dl: Del(C,C,red(C,p),mulp(p),rh(C,p),e,h) by Del_tot[C, C, red(C,p), mulp(p), rh(C,p), e] ss eH
    have ke: k(bock(C,p),e) = h by bock_k dl
    have dl0: Del(C,C,red(C,p),mulp(p),rh(C,p),e,0) by dl ke h0
    obtain x:el st m: HM(C,red(C,p),rh(C,p),x,e) by les_kerdel[C, C, red(C,p), mulp(p), rh(C,p), e] ss dl0
    have xH: x in Hc(C) by HM_dom m
    have xD: x in D(bock(C,p)) by bock_D xH
    have jx: j(bock(C,p),x) = e by bock_j m
    show by xD jx with x
  qed
  have kj: forall x:el: x in D(bock(C,p)) -> k(bock(C,p),j(bock(C,p),x)) = 0
  proof
    fix x
    intro hx
    have xH: x in Hc(C) by bock_D hx
    obtain h:el st m: HM(C,red(C,p),rh(C,p),x,h) by HM_tot[C, red(C,p), rh(C,p), x] cg xH
    have e: j(bock(C,p),x) = h by bock_j m
    have hH: h in Hc(red(C,p)) by HM_dom m
    obtain h4:el st dl: Del(C,C,red(C,p),mulp(p),rh(C,p),h,h4) by Del_tot[C, C, red(C,p), mulp(p), rh(C,p), h] ss hH
    have kh: k(bock(C,p),h) = h4 by bock_k dl
    have z: h4 = 0 by les_gdel[C, C, red(C,p), mulp(p), rh(C,p), x, h, h4] ss m dl
    show by e kh z
  qed
  # ker i = im k
  have keri: forall x:el: x in D(bock(C,p)) and i(bock(C,p),x) = 0 -> exists e:el: e in E(bock(C,p)) and x = k(bock(C,p),e)
  proof
    fix x
    intro hx h0
    have xH: x in Hc(C) by bock_D hx
    obtain h:el st m: HM(C,C,mulp(p),x,h) by HM_tot[C, C, mulp(p), x] cf xH
    have e: i(bock(C,p),x) = h by bock_i m
    have m0: HM(C,C,mulp(p),x,0) by m e h0
    obtain d:el st dl: Del(C,C,red(C,p),mulp(p),rh(C,p),d,x) by les_kerf[C, C, red(C,p), mulp(p), rh(C,p), x] ss xH m0
    have dH: d in Hc(red(C,p)) by Del_dom dl
    have dE: d in E(bock(C,p)) by bock_E dH
    have kd: k(bock(C,p),d) = x by bock_k dl
    show by dE kd with d
  qed
  have ik: forall e:el: e in E(bock(C,p)) -> i(bock(C,p),k(bock(C,p),e)) = 0
  proof
    fix e
    intro he
    have eH: e in Hc(red(C,p)) by bock_E he
    obtain h:el st dl: Del(C,C,red(C,p),mulp(p),rh(C,p),e,h) by Del_tot[C, C, red(C,p), mulp(p), rh(C,p), e] ss eH
    have ke: k(bock(C,p),e) = h by bock_k dl
    have hH: h in Hc(C) by Del_dom dl
    obtain h2:el st m: HM(C,C,mulp(p),h,h2) by HM_tot[C, C, mulp(p), h] cf hH
    have ih: i(bock(C,p),h) = h2 by bock_i m
    have z: h2 = 0 by les_delf[C, C, red(C,p), mulp(p), rh(C,p), e, h, h2] ss dl m
    show by ke ih z
  qed
  show excouple(bock(C,p)) by excouple_def sD sE iD jE kD kerj ji kerk kj keri ik
qed

# i = H(p) is multiplication by p on classes
lemma bock_ip: forall C:cx, p:int, x:el: fintype(C) and x in D(bock(C,p)) -> i(bock(C,p),x) = p*x
proof
  fix C p x
  intro hf hx
  have cx: iscx(C) by fintype_def hf
  have xH: x in Hc(C) by bock_D hx
  obtain z:el st cz: Cls(C,z,x) by cls_sur xH
  have cp: Cls(C,p*z,p*x) by cls_smul[C, p, z, x] cx cz
  have mz: Cls(C,ap(mulp(p),z),p*x) by cp mulp_def[p, z]
  have m: HM(C,C,mulp(p),x,p*x) by HM_def cz mz with z
  show by bock_i m
qed

lemma zs_mem: forall a:el, S:set: zsummand(a,S) -> a in S and subgroup(S)
proof
  fix a S
  intro hz
  obtain M:set st aS sS hM hMS hdec hind: a in S and subgroup(S) and subgroup(M) and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> n = 0) by zsummand_def hz
  show by aS sS
qed

# i^m = p^m on the Bockstein couple
lemma ipow_bock: forall C:cx, p:int: fintype(C) and prime(p) -> forall m:int: 0 <= m -> forall x:el: x in D(bock(C,p)) -> ipow(bock(C,p),m,x) = p^m*x
proof
  fix C p
  intro hf hp
  have ec: excouple(bock(C,p)) by bock_couple hf hp
  induct m from 0
  base
  proof
    fix x
    intro hx
    show by ipow_zero[bock(C,p), x]
  qed
  step
  proof
    fix m
    intro hm ih
    fix x
    intro hx
    have s: ipow(bock(C,p),m+1,x) = i(bock(C,p),ipow(bock(C,p),m,x)) by ipow_succ[bock(C,p), m, x] hm
    have a: ipow(bock(C,p),m,x) = p^m*x by ih hx
    have yD: ipow(bock(C,p),m,x) in D(bock(C,p)) by ipow_D ec hx
    have b: i(bock(C,p),ipow(bock(C,p),m,x)) = p*ipow(bock(C,p),m,x) by bock_ip hf yD
    have e: p^(m+1) = p^m*p by pow_succ[p, m] hm
    show by s a b e
  qed
qed

# the quotient differential is well defined (red_dif names one value)
lemma red_dif_wd: forall C:cx, p:int, x y:el: iscx(C) and x in Ch(C) and y in Ch(C) and ap(rh(C,p),x) = ap(rh(C,p),y) -> ap(rh(C,p),dif(C,x)) = ap(rh(C,p),dif(C,y))
proof
  fix C p x y
  intro cx hx hy he
  have s: subgroup(Ch(C)) by cx_sub cx
  have d: x - y in Ch(C) by s hx hy
  have r0: ap(rh(C,p),x - y) = 0 by he
  obtain u:el st uc ue: u in Ch(C) and x - y = p*u by red_ker[C, p, x - y] d r0
  have du: dif(C,u) in Ch(C) by cx_d cx uc
  have pd: p*dif(C,u) in Ch(C) by s du
  have r1: ap(rh(C,p),p*dif(C,u)) = 0 by red_ker[C, p, p*dif(C,u)] pd du with dif(C,u)
  have dd: dif(C,x) - dif(C,y) = p*dif(C,u) by ue
  show by r1 dd
qed

# the degrees of the Bockstein couple: i and j of degree 0, k of degree +1
lemma deg_i: forall C:cx, p n:int, h:el: fintype(C) and h in H(C,n) -> i(bock(C,p),h) in H(C,n)
proof
  fix C p n h
  intro hf hh
  have cx: iscx(C) by fintype_def hf
  have hc: h in Hc(C) by H_in hh
  have hD: h in D(bock(C,p)) by bock_D hc
  have e: i(bock(C,p),h) = p*h by bock_ip hf hD
  obtain z:el st zn cz: z in Cn(C,n) and Cls(C,z,h) by H_def hh
  have sn: subgroup(Cn(C,n)) by Cn_sub cx
  have pz: p*z in Cn(C,n) by sn zn
  have cp: Cls(C,p*z,p*h) by cls_smul[C, p, z, h] cx cz
  have m: p*h in H(C,n) by H_def pz cp with p*z
  show by m e
qed

lemma deg_j: forall C:cx, p n:int, h:el: fintype(C) and h in H(C,n) -> j(bock(C,p),h) in H(red(C,p),n)
proof
  fix C p n h
  intro hf hh
  have cx: iscx(C) by fintype_def hf
  obtain z:el st zn cz: z in Cn(C,n) and Cls(C,z,h) by H_def hh
  have zc: z in Ch(C) and dif(C,z) = 0 by cls_dom cz
  have wn: ap(rh(C,p),z) in Cn(red(C,p),n) by red_Cn zn with z
  have wc: ap(rh(C,p),z) in Ch(red(C,p)) by red_ch zc with z
  have dw: dif(red(C,p),ap(rh(C,p),z)) = ap(rh(C,p),dif(C,z)) by red_dif zc
  have dw0: dif(red(C,p),ap(rh(C,p),z)) = 0 by dw zc
  obtain e:el st ce: Cls(red(C,p),ap(rh(C,p),z),e) by cls_tot[red(C,p), ap(rh(C,p),z)] wc dw0
  have m: HM(C,red(C,p),rh(C,p),h,e) by HM_def cz ce with z
  have je: j(bock(C,p),h) = e by bock_j m
  have eH: e in H(red(C,p),n) by H_def wn ce with ap(rh(C,p),z)
  show by eH je
qed

lemma deg_k: forall C:cx, p n:int, e:el: fintype(C) and prime(p) and e in H(red(C,p),n) -> k(bock(C,p),e) in H(C,n+1)
proof
  fix C p n e
  intro hf hp he
  have cx: iscx(C) by fintype_def hf
  have tf: forall m:int, x:el: x in Ch(C) and 1 <= m and m*x = 0 -> x = 0 by fintype_def hf
  have p2: 2 <= p by prime_ge2 hp
  have p1: 1 <= p by p2
  have s: subgroup(Ch(C)) by cx_sub cx
  obtain w:el st wn cw: w in Cn(red(C,p),n) and Cls(red(C,p),w,e) by H_def he
  have wc: w in Ch(red(C,p)) and dif(red(C,p),w) = 0 by cls_dom cw
  obtain x:el st xn xw: x in Cn(C,n) and w = ap(rh(C,p),x) by red_Cn wn
  have xc: x in Ch(C) by Cn_ch xn
  have dx: dif(C,x) in Ch(C) by cx_d cx xc
  have rd: dif(red(C,p),ap(rh(C,p),x)) = ap(rh(C,p),dif(C,x)) by red_dif xc
  have r0: ap(rh(C,p),dif(C,x)) = 0 by rd xw wc
  obtain z:el st zc zx: z in Ch(C) and dif(C,x) = p*z by red_ker[C, p, dif(C,x)] dx r0
  # d x has degree n+1, so z does (C is torsion-free)
  have pdx: proj(C,n+1,dif(C,x)) = dif(C,proj(C,n,x)) by proj_dif[C, n, x] xc
  have px: proj(C,n,x) = x by proj_id xn
  have pz: p*(proj(C,n+1,z) - z) = 0 by pdx px zx
  have pzc: proj(C,n+1,z) in Cn(C,n+1) by proj_in[C, n+1, z] zc
  have pzc2: proj(C,n+1,z) in Ch(C) by Cn_ch pzc
  have dz: proj(C,n+1,z) - z in Ch(C) by s pzc2 zc
  have z1: proj(C,n+1,z) - z = 0 by tf[p, proj(C,n+1,z) - z] dz p1 pz
  have zn: z in Cn(C,n+1) by pzc z1
  # z is a cocycle: p d z = d d x = 0
  have ddx: dif(C,dif(C,x)) = 0 by cx_dd cx xc
  have dzc: dif(C,z) in Ch(C) by cx_d cx zc
  have pdz: p*dif(C,z) = 0 by ddx zx
  have dz0: dif(C,z) = 0 by tf[p, dif(C,z)] dzc p1 pdz
  obtain h:el st ch: Cls(C,z,h) by cls_tot[C, z] zc dz0
  have fz: ap(mulp(p),z) = dif(C,x) by mulp_def[p, z] zx
  have dl: Del(C,C,red(C,p),mulp(p),rh(C,p),e,h) by Del_def cw xc xw zc fz ch with w, x, z
  have ke: k(bock(C,p),e) = h by bock_k dl
  have hn: h in H(C,n+1) by H_def zn ch with z
  show by hn ke
qed

# Proposition 0.1, following the paper: a-bar = j_1(a) lies in im j_r,
# so in ker k_r, so in ker d_r; D_r = p^(r-1) H, j_r(p^(r-1) a) = a-bar,
# ker j_r = im i_r, i_r = p on D_r, im k_r = ker i_r (p-torsion of
# D_r); a-bar in im d_r would put p^(r-1) a in (p-torsion) + p^r H, which
# the valuation argument rules out for a generator of a Z summand.
theorem bockstein_paper: forall C:cx, p n r:int, a:el: fintype(C) and prime(p) and 0 <= n and 1 <= r and a in H(C,n) and zsummand(a,H(C,n)) -> Zr(bock(C,p),r,j(bock(C,p),a)) and dzero(bock(C,p),r,j(bock(C,p),a)) and not dimg(bock(C,p),r,n-1,j(bock(C,p),a))
proof
  fix C p n r a
  intro hC hp hn hr ha hs
  have ec: excouple(bock(C,p)) by bock_couple hC hp
  have p2: 2 <= p by prime_ge2 hp
  have cxC: iscx(C) by fintype_def hC
  have hsum: summand(H(C,n),Hc(C)) by H_summand cxC
  have hsT: zsummand(a,Hc(C)) by zsummand_trans[a, H(C,n), Hc(C)] hs hsum
  have aT: a in Hc(C) by zs_mem hsT
  have aD: a in D(bock(C,p)) by bock_D aT
  have r0: 0 <= r - 1 by hr
  # j_r(p^(r-1) a) = a-bar
  have jr: Jr(bock(C,p),r,ipow(bock(C,p),r-1,a),j(bock(C,p),a)) by dc_jr ec hr aD
  # a-bar in im j_r = ker k_r, and it is a class of E_r
  have kj: k(bock(C,p),j(bock(C,p),a)) = 0 by dc_imj ec hr jr
  have s1: Zr(bock(C,p),r,j(bock(C,p),a)) by dc_JZ ec hr jr
  # d_r a-bar = j_r k_r a-bar = j_r 0 = 0
  have s2: dzero(bock(C,p),r,j(bock(C,p),a))
  proof
    have z: 0 in D(bock(C,p)) by sub_zero ec_subD ec
    have j0: Jr(bock(C,p),r,0,0) by dc_jr[bock(C,p), r, 0] ec hr z
    have b0: Br(bock(C,p),r,0) by dc_B0 ec hr
    have j1: Jr(bock(C,p),r,k(bock(C,p),j(bock(C,p),a)),0) by j0 kj
    show by dzero_def j1 b0 with 0
  qed
  # a-bar is not in im d_r^(n-1)
  have s3: not dimg(bock(C,p),r,n-1,j(bock(C,p),a))
  proof
    intro h
    obtain e0 f:el st z0 dg jf bf: Zr(bock(C,p),r,e0) and Edeg(bock(C,p),n-1,e0) and Jr(bock(C,p),r,k(bock(C,p),e0),f) and Br(bock(C,p),r,j(bock(C,p),a) - f) by dimg_def h
    # j_r(p^(r-1) a - k_r e0) = a-bar - d_r e0 = 0 in E_r
    have j2: Jr(bock(C,p),r,ipow(bock(C,p),r-1,a) - k(bock(C,p),e0),j(bock(C,p),a) - f) by dc_Jsub[bock(C,p), r, ipow(bock(C,p),r-1,a), k(bock(C,p),e0), j(bock(C,p),a), f] ec hr jr jf
    # ker j_r = im i_r
    obtain y:el st yd yi: Dr(bock(C,p),r,y) and ipow(bock(C,p),r-1,a) - k(bock(C,p),e0) = i(bock(C,p),y) by dc_kerj ec hr j2 bf
    # D_r = p^(r-1) H
    obtain x:el st xD yx: x in D(bock(C,p)) and y = ipow(bock(C,p),r-1,x) by dc_Dr ec hr yd
    # im k_r = ker i_r: k_r e0 is p-torsion in D_r
    have tk: Dr(bock(C,p),r,k(bock(C,p),e0)) and i(bock(C,p),k(bock(C,p),e0)) = 0 by dc_imk ec hr z0
    have tD: k(bock(C,p),e0) in D(bock(C,p)) by dc_DD ec hr tk
    # i = p, i^(r-1) = p^(r-1)
    have e1: ipow(bock(C,p),r-1,a) = p^(r-1)*a by ipow_bock[C, p] hC hp r0 aD
    have e2: ipow(bock(C,p),r-1,x) = p^(r-1)*x by ipow_bock[C, p] hC hp r0 xD
    have yD: y in D(bock(C,p)) by dc_DD ec hr yd
    have e3: i(bock(C,p),y) = p*y by bock_ip[C, p, y] hC yD
    have tD0: k(bock(C,p),e0) in D(bock(C,p)) by dc_DD ec hr tk
    have e4: i(bock(C,p),k(bock(C,p),e0)) = p*k(bock(C,p),e0) by bock_ip[C, p, k(bock(C,p),e0)] hC tD0
    have pt: p*k(bock(C,p),e0) = 0 by tk e4
    # p^(r-1) a = t + p^r x with p t = 0
    have key: p^(r-1)*a = k(bock(C,p),e0) + (p*p^(r-1))*x by yi yx e1 e2 e3
    have p1: 1 <= p by p2
    have q1: 1 <= p^(r-1) by pow_pos p1 r0
    have tH: k(bock(C,p),e0) in Hc(C) by bock_D tD
    have xH: x in Hc(C) by bock_D xD
    show false by zsum_val[a, Hc(C), p, p^(r-1), k(bock(C,p),e0), x] hsT p2 q1 tH xH pt key
  qed
  show by s1 s2 s3
qed

# Proposition 0.1 by another route: the closed forms Z_(r+1) =
# k^-1(p^r H) and B_(r+1) = j(H[p^r]) and the coordinates of a direct sum
# H = <a> (+) M; a-bar in im d_r would mean a in H[p^r] + pH, and the
# a-coordinate gives 1 in pZ.
theorem bockstein_coords: forall C:cx, p n r:int, a:el: fintype(C) and prime(p) and 0 <= n and 1 <= r and a in H(C,n) and zsummand(a,H(C,n)) -> Zr(bock(C,p),r,j(bock(C,p),a)) and dzero(bock(C,p),r,j(bock(C,p),a)) and not dimg(bock(C,p),r,n-1,j(bock(C,p),a))
proof
  fix C p n r a
  intro hC hp hn hr ha hs
  have ec: excouple(bock(C,p)) by bock_couple hC hp
  have p2: 2 <= p by prime_ge2 hp
  have cxC: iscx(C) by fintype_def hC
  have hsum: summand(H(C,n),Hc(C)) by H_summand cxC
  have hsT: zsummand(a,Hc(C)) by zsummand_trans[a, H(C,n), Hc(C)] hs hsum
  have aT: a in Hc(C) by zs_mem hsT
  have aD: a in D(bock(C,p)) by bock_D aT
  have sD: subgroup(D(bock(C,p))) by ec_subD ec
  have rr: 0 <= r by hr
  have r1: 1 <= r + 1 by hr
  have kj: k(bock(C,p),j(bock(C,p),a)) = 0 by ec_kj ec aD
  have jaE: j(bock(C,p),a) in E(bock(C,p)) by ec_jE ec aD
  have z0: 0 in D(bock(C,p)) by sub_zero sD
  # a-bar in Z_(r+1) = k^-1(p^r H): it survives, and d_r a-bar = 0
  have zr1: Zr(bock(C,p),r+1,j(bock(C,p),a)) by cf_Z[bock(C,p), r+1, j(bock(C,p),a)] ec r1 jaE kj z0 with 0
  have s1: Zr(bock(C,p),r,j(bock(C,p),a)) by Zsucc_def zr1 hr
  have s2: dzero(bock(C,p),r,j(bock(C,p),a))
  proof
    obtain f:el st jf bf: Jr(bock(C,p),r,k(bock(C,p),j(bock(C,p),a)),f) and Br(bock(C,p),r,f) by Zsucc_def zr1 hr
    show by dzero_def jf bf with f
  qed
  have s3: not dimg(bock(C,p),r,n-1,j(bock(C,p),a))
  proof
    intro h
    obtain e0 f:el st z1 dg jf bf: Zr(bock(C,p),r,e0) and Edeg(bock(C,p),n-1,e0) and Jr(bock(C,p),r,k(bock(C,p),e0),f) and Br(bock(C,p),r,j(bock(C,p),a) - f) by dimg_def h
    # a-bar in im d_r: a-bar = 0 in E_(r+1)
    have b: Br(bock(C,p),r+1,j(bock(C,p),a)) by Bsucc_def hr z1 jf bf with e0, f
    # B_(r+1) = j(ker i^r) = j(H[p^r])
    obtain x:el st xD xi xj: x in D(bock(C,p)) and ipow(bock(C,p),r,x) = 0 and j(bock(C,p),a) = j(bock(C,p),x) by cf_B[bock(C,p), r+1, j(bock(C,p),a)] ec r1 b
    have dD: a - x in D(bock(C,p)) by sD aD xD
    have jd: j(bock(C,p),a - x) = 0 by xj
    # ker j = im i = pH
    obtain y:el st yD yi: y in D(bock(C,p)) and a - x = i(bock(C,p),y) by ec_kerj ec dD jd
    have e1: ipow(bock(C,p),r,x) = p^r*x by ipow_bock[C, p] hC hp rr xD
    have e2: i(bock(C,p),y) = p*y by bock_ip[C, p, y] hC yD
    have p1: 1 <= p by p2
    have w1: 1 <= p^r by pow_pos p1 rr
    have wx: p^r*x = 0 by xi e1
    have ax: a = x + p*y by yi e2
    have xH: x in Hc(C) by bock_D xD
    have yH: y in Hc(C) by bock_D yD
    show false by zsum_unit[a, Hc(C), p, p^r, x, y] hsT p2 w1 xH yH wx ax
  qed
  show by s1 s2 s3
qed

# ---- exponents: symbolic p^k ----
lemma pow_add: forall p a:int: 0 <= a -> forall b:int: 0 <= b -> p^(a+b) = p^a*p^b
proof
  fix p a
  intro ha
  induct b from 0
  base
  proof
    show by pow_zero[p]
  qed
  step
  proof
    fix b
    intro hb ih
    have e1: p^(a+b+1) = p^(a+b)*p by pow_succ[p, a+b] ha hb
    have e2: p^(b+1) = p^b*p by pow_succ[p, b] hb
    show by e1 e2 ih
  qed
qed

# no multiple of p is 1
lemma unit_p: forall p z:int: 2 <= p and 1 = p*z -> false
proof
  fix p z
  intro hp hz
  show false by hp hz
qed

lemma zpk_mem: forall a:el, S:set, p k:int: zpksummand(a,S,p,k) -> a in S and subgroup(S) and p^k*a = 0
proof
  fix a S p k
  intro hz
  obtain M:set st aS sS hM pka hMS hdec hind: a in S and subgroup(S) and subgroup(M) and p^k*a = 0 and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> exists q:int: n = p^k*q) by zpksummand_def hz
  show by aS sS pka
qed

lemma zpksummand_trans: forall a:el, S T:set, p k:int: zpksummand(a,S,p,k) and summand(S,T) -> zpksummand(a,T,p,k)
proof
  fix a S T p k
  intro hz hs
  obtain M:set st aS sS hM pka hMS hdec hind: a in S and subgroup(S) and subgroup(M) and p^k*a = 0 and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> exists q:int: n = p^k*q) by zpksummand_def hz
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
  have ind: forall q:int, w:el: w in ssum(M,N) and q*a + w = 0 -> exists t:int: q = p^k*t
  proof
    fix q w
    intro hw h0
    obtain m n:el st mM nN e: m in M and n in N and w = m + n by ssum_def hw
    have mS: m in S by hMS mM
    have qa: q*a in S by sub_smul[S, q, a] sS aS
    have s1: q*a + m in S by sS qa mS
    have s2: (q*a + m) + n = 0 by h0 e
    have s3: q*a + m = 0 by hind2[q*a + m, n] s1 nN s2
    show by hind[q, m] mM s3
  qed
  show zpksummand(a,T,p,k) by zpksummand_def aT sT hMN pka sub dec ind with ssum(M,N)
qed

# in S = <a> (+) M with a of order p^k: a is not in S[p^(r-1)] + pS for
# 1 <= r <= k (symbolic exponents: p^k = p^(r-1) p^(k-r+1))
lemma zpk_unit: forall a:el, S:set, p k r:int, x y:el: zpksummand(a,S,p,k) and 2 <= p and 1 <= r and r <= k and x in S and y in S and p^(r-1)*x = 0 and a = x + p*y -> false
proof
  fix a S p k r x y
  intro hz hp hr hrk hx hy hpx ha
  obtain M:set st aS sS hM pka hMS hdec hind: a in S and subgroup(S) and subgroup(M) and p^k*a = 0 and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> exists q:int: n = p^k*q) by zpksummand_def hz
  obtain s:int, m1:el st m1M xe: m1 in M and x = s*a + m1 by hdec hx
  obtain u:int, m2:el st m2M ye: m2 in M and y = u*a + m2 by hdec hy
  have r0: 0 <= r - 1 by hr
  have pm: p^(r-1)*m1 in M by hM m1M
  have e0: (p^(r-1)*s)*a + p^(r-1)*m1 = 0 by hpx xe
  obtain q:int st hq: p^(r-1)*s = p^k*q by hind[p^(r-1)*s, p^(r-1)*m1] pm e0
  have pk: p^k = p^(r-1)*p^(k-r+1) by pow_add[p, r-1] r0 hrk
  have pq: p^(r-1)*(s - p^(k-r+1)*q) = 0 by hq pk
  have p1: 1 <= p by hp
  have P1: 1 <= p^(r-1) by pow_pos p1 r0
  have s1: s = p^(k-r+1)*q
  proof
    cases hc: p^(r-1) = 0 or s - p^(k-r+1)*q = 0 by int_domain[p^(r-1), s - p^(k-r+1)*q] pq
    case h1
    proof
      show by h1 P1
    qed
    case h2
    proof
      show by h2
    qed
  qed
  have kr: 0 <= k - r by hrk
  have b1: p^(k-r+1) = p^(k-r)*p by pow_succ[p, k-r] kr
  have mm: -m1 - p*m2 in M by hM m1M m2M
  have co: (1 - s - p*u)*a + (-m1 - p*m2) = 0 by ha xe ye
  obtain q2:int st hq2: 1 - s - p*u = p^k*q2 by hind[1 - s - p*u, -m1 - p*m2] mm co
  have k0: 0 <= k - 1 by hr hrk
  have b2: p^k = p^(k-1)*p by pow_succ[p, k-1] k0
  have z: 1 = p*(p^(k-r)*q + u + p^(k-1)*q2) by hq2 s1 b1 b2
  show false by unit_p[p, p^(k-r)*q + u + p^(k-1)*q2] hp z
qed

# the order of a generator of a Z/p^e summand is exactly p^e
theorem cyclic_order: forall a:el, S:set, p e k:int: zpksummand(a,S,p,e) and 2 <= p and 0 <= k and k < e -> p^k*a != 0
proof
  fix a S p e k
  intro hz hp hk hke
  intro h0
  obtain M:set st aS sS hM pka hMS hdec hind: a in S and subgroup(S) and subgroup(M) and p^e*a = 0 and (forall x:el: x in M -> x in S) and (forall x:el: x in S -> exists n:int, m:el: m in M and x = n*a + m) and (forall n:int, m:el: m in M and n*a + m = 0 -> exists q:int: n = p^e*q) by zpksummand_def hz
  have z0: 0 in M by sub_zero hM
  have c0: p^k*a + 0 = 0 by h0
  obtain q:int st hq: p^k = p^e*q by hind[p^k, 0] z0 c0
  have pk: p^e = p^k*p^(e-k) by pow_add[p, k] hk hke
  have ek: 0 <= e - k - 1 by hke
  have b: p^(e-k) = p^(e-k-1)*p by pow_succ[p, e-k-1] ek
  have pq: p^k*(1 - p*(p^(e-k-1)*q)) = 0 by hq pk b
  have p1: 1 <= p by hp
  have P1: 1 <= p^k by pow_pos p1 hk
  cases hc: p^k = 0 or 1 - p*(p^(e-k-1)*q) = 0 by int_domain[p^k, 1 - p*(p^(e-k-1)*q)] pq
  case h1
  proof
    show by h1 P1
  qed
  case h2
  proof
    have z: 1 = p*(p^(e-k-1)*q) by h2
    show false by unit_p[p, p^(e-k-1)*q] hp z
  qed
qed

# The Bockstein of a Z/p^k summand dies exactly at d_k: a-bar is nonzero
# on the pages E_1 .. E_k and zero on E_(k+1) (a-bar is in im d_k)
theorem bockstein_torsion: forall C:cx, p n k:int, a:el: fintype(C) and prime(p) and 1 <= k and a in H(C,n) and zpksummand(a,H(C,n),p,k) -> (forall r:int: 1 <= r and r <= k -> not Br(bock(C,p),r,j(bock(C,p),a))) and Br(bock(C,p),k+1,j(bock(C,p),a))
proof
  fix C p n k a
  intro hC hp hk ha hz
  have ec: excouple(bock(C,p)) by bock_couple hC hp
  have p2: 2 <= p by prime_ge2 hp
  have cxC: iscx(C) by fintype_def hC
  have hsum: summand(H(C,n),Hc(C)) by H_summand cxC
  have hzT: zpksummand(a,Hc(C),p,k) by zpksummand_trans[a, H(C,n), Hc(C), p, k] hz hsum
  have am: a in Hc(C) and subgroup(Hc(C)) and p^k*a = 0 by zpk_mem hzT
  have aD: a in D(bock(C,p)) by bock_D am
  have sD: subgroup(D(bock(C,p))) by ec_subD ec
  have part1: forall r:int: 1 <= r and r <= k -> not Br(bock(C,p),r,j(bock(C,p),a))
  proof
    fix r
    intro hr hrk
    intro hb
    have r0: 0 <= r - 1 by hr
    obtain x:el st xD xi xj: x in D(bock(C,p)) and ipow(bock(C,p),r-1,x) = 0 and j(bock(C,p),a) = j(bock(C,p),x) by cf_B[bock(C,p), r, j(bock(C,p),a)] ec hr hb
    have dD: a - x in D(bock(C,p)) by sD aD xD
    have jd: j(bock(C,p),a - x) = 0 by xj
    obtain y:el st yD yi: y in D(bock(C,p)) and a - x = i(bock(C,p),y) by ec_kerj ec dD jd
    have e1: ipow(bock(C,p),r-1,x) = p^(r-1)*x by ipow_bock[C, p] hC hp r0 xD
    have e2: i(bock(C,p),y) = p*y by bock_ip[C, p, y] hC yD
    have wx: p^(r-1)*x = 0 by xi e1
    have ax: a = x + p*y by yi e2
    have xH: x in Hc(C) by bock_D xD
    have yH: y in Hc(C) by bock_D yD
    show false by zpk_unit[a, Hc(C), p, k, r, x, y] hzT p2 hr hrk xH yH wx ax
  qed
  have k0: 0 <= k by hk
  have k1: 1 <= k + 1 by hk
  have ek: ipow(bock(C,p),k,a) = p^k*a by ipow_bock[C, p] hC hp k0 aD
  have part2: Br(bock(C,p),k+1,j(bock(C,p),a)) by cf_B[bock(C,p), k+1, j(bock(C,p),a)] ec k1 aD ek am with a
  show by part1 part2
qed

# ---- subgroup lattices ----
func inter(set, set): set
def inter_def: forall A B:set, x:el: x in inter(A,B) <-> (x in A and x in B)

# Dedekind's modular law: A <= C implies A + (B n C) = (A + B) n C
theorem modular_law: forall A B C:set: subgroup(A) and subgroup(B) and subgroup(C) and (forall x:el: x in A -> x in C) -> (forall x:el: x in ssum(A,inter(B,C)) <-> (x in ssum(A,B) and x in C))
proof
  fix A B C
  intro ha hb hc hac
  fix x
  have l: x in ssum(A,inter(B,C)) -> x in ssum(A,B) and x in C
  proof
    intro h
    obtain a b:el st aA bBC e: a in A and b in inter(B,C) and x = a + b by ssum_def h
    have bB: b in B by inter_def bBC
    have bC: b in C by inter_def bBC
    have aC: a in C by hac aA
    have s1: x in ssum(A,B) by ssum_def aA bB e with a, b
    have ab: a + b in C by hc aC bC
    have s2: x in C by e ab
    show by s1 s2
  qed
  have rr: x in ssum(A,B) and x in C -> x in ssum(A,inter(B,C))
  proof
    intro h xC
    obtain a b:el st aA bB e: a in A and b in B and x = a + b by ssum_def h
    have aC: a in C by hac aA
    have xa: x - a in C by hc xC aC
    have bC: b in C by xa e
    have bBC: b in inter(B,C) by inter_def bB bC
    show by ssum_def aA bBC e with a, b
  qed
  show by l rr
qed

# ---- diagram chases with arbitrary homomorphisms ----
# the short five lemma, injective half: rows A -f-> B -g-> C exact at B
# and A2 -f2-> B2 with f2 injective, squares commuting, alpha and gamma
# injective: then beta is injective
theorem short_five_mono: forall f g f2 g2 al be ga:hom, A B C A2:set: (forall x:el: x in A -> ap(f,x) in B) and (forall y:el: y in B -> ap(g,y) in C) and (forall x:el: x in A -> ap(al,x) in A2) and (forall y:el: y in B and ap(g,y) = 0 -> exists x:el: x in A and y = ap(f,x)) and (forall x:el: x in A2 and ap(f2,x) = 0 -> x = 0) and (forall x:el: x in A -> ap(be,ap(f,x)) = ap(f2,ap(al,x))) and (forall y:el: y in B -> ap(ga,ap(g,y)) = ap(g2,ap(be,y))) and (forall x:el: x in A and ap(al,x) = 0 -> x = 0) and (forall z:el: z in C and ap(ga,z) = 0 -> z = 0) -> (forall y:el: y in B and ap(be,y) = 0 -> y = 0)
proof
  fix f g f2 g2 al be ga A B C A2
  intro hf hg hal ex mf2 sq1 sq2 ial iga
  fix y
  intro yB by0
  have gC: ap(g,y) in C by hg yB
  have s: ap(ga,ap(g,y)) = 0 by sq2[y] yB by0
  have gy: ap(g,y) = 0 by iga gC s
  obtain x:el st xA yx: x in A and y = ap(f,x) by ex yB gy
  have aA2: ap(al,x) in A2 by hal xA
  have t: ap(f2,ap(al,x)) = 0 by sq1[x] xA yx by0
  have ax: ap(al,x) = 0 by mf2 aA2 t
  have x0: x = 0 by ial xA ax
  show by yx x0
qed

# the snake lemma's first exactness: in a commutative diagram with exact
# top row A -f-> B -g-> C and an injective f2: A2 -> B2, the induced
# ker alpha -> ker beta -> ker gamma is exact at ker beta
theorem snake_ker_exact: forall f g f2 g2 al be ga:hom, A B A2:set: (forall y:el: y in B and ap(g,y) = 0 -> exists x:el: x in A and y = ap(f,x)) and (forall x:el: x in A -> ap(al,x) in A2) and (forall x:el: x in A2 and ap(f2,x) = 0 -> x = 0) and (forall x:el: x in A -> ap(be,ap(f,x)) = ap(f2,ap(al,x))) -> (forall y:el: y in B and ap(be,y) = 0 and ap(g,y) = 0 -> exists x:el: x in A and ap(al,x) = 0 and y = ap(f,x))
proof
  fix f g f2 g2 al be ga A B A2
  intro ex hal mf2 sq
  fix y
  intro yB by0 gy
  obtain x:el st xA yx: x in A and y = ap(f,x) by ex yB gy
  have aA2: ap(al,x) in A2 by hal xA
  have t: ap(f2,ap(al,x)) = 0 by sq[x] xA yx by0
  have ax: ap(al,x) = 0 by mf2 aA2 t
  show by xA ax yx with x
qed
