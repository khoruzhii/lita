#------------------------------------------------------------------------------
# -*-mapleV-*-
#------------------------------------------------------------------------------
# Even-dimensional Local Improvements to Trilinear Aggregation (LITA).
#
# The formula uses the Schwartz--Zwecher transformed TA skeleton, removes the
# eight-term diagonal correction motif in each diagonal block, and inserts the
# rational rank-7 local identity.
#
# The procedure supports even square dimensions N > 18.
#------------------------------------------------------------------------------

LITACOMPLEXITY:=proc(z)
    if type(z,numeric)
    then
        if not type(z,integer)
        then error "dimension should be an even integer at least 18"
        end if:

        if z < 18 or irem(z,2)<>0
        then error "dimension should be an even integer at least 18"
        end if:
    end if:

    return z^3/3+15*z^2/4+29*z/3+7:
end proc:

# Return the seven local rank-one triples.
# Coordinates are ordered as [00,10,01,11].  The sign pullback on the second
# and third factors is included in the local parametrisation.
LITA_LOCAL7:=proc(d)
local e,u,v,w,A,B,C,Ap,Bp,Cp,lam,a,b,c,ap,bp,cp,D0,D2,raw,t,r:

    e := [-1,0,0,1]:

    u := [d*(8-d)/(d-6), -2*d/(d-6), d*(d-2)/(d-6), 0]:
    v := [0,-1,1,0]:
    w := [1/2,0,1/2,0]:

    A  := d/(d-6):
    B  := -(d-6)/d:
    C  := (d-3)/(2*d):
    Ap := d*(d-7)/(d-6):
    Bp := 3/(2*d):
    Cp := (d-6)/d:
    lam := (d^2-11*d+27)/d:

    a  := [seq(u[t]+A*e[t],t=1..4)]:
    b  := [seq(v[t]+B*e[t],t=1..4)]:
    c  := [seq(w[t]+C*e[t],t=1..4)]:
    ap := [seq(-u[t]+Ap*e[t],t=1..4)]:
    bp := [seq(w[t]+Bp*e[t],t=1..4)]:
    cp := [seq(v[t]+Cp*e[t],t=1..4)]:

    D0 := [1,-1,-1,1]:
    D2 := [1,1,-1,-1]:

    raw := [
        [[seq(lam*e[t],t=1..4)], e, e],
        [a,b,c], [b,c,a], [c,a,b],
        [ap,bp,cp], [bp,cp,ap], [cp,ap,bp]
    ]:

    return [seq([
        raw[r][1],
        [seq(D0[t]*raw[r][2][t],t=1..4)],
        [seq(D2[t]*raw[r][3][t],t=1..4)]
    ],r=1..7)]:
end proc:

LITA_FORM4:=proc(M,v,i,foo,bar)
    return
        v[1]*M[foo(i),foo(i)]
        + v[2]*M[bar(i),foo(i)]
        + v[3]*M[foo(i),bar(i)]
        + v[4]*M[bar(i),bar(i)]:
end proc:

LITA:=proc(triad :: TRIAD)

description
    "Given a triad, return the trilinear form of the even-dimensional LITA",
    "local improvement to Schwartz--Zwecher trilinear aggregation",
    COPYLEFT :

local
    i,j,k,r,
    dims,n,d,coeffs,foo,bar,u,phi,L,R,
    Astar,Bstar,Cstar
:

    dims := TRIADDIM(triad) :

    if nops(convert(dims,`set`))<>1
    then error "triad should be square"
    end if:

    if map(irem,dims,2)<>[0,0,0]
    then error "triad's dimension should be a multiple of 2"
    end if:

    if dims[1] < 18
    then error "triad's dimension should be at least 18"
    end if:

    n:=dims[1]/2:
    d:=1+n:

    # In the source formulas, indices are in [0,..,d-1].
    # Maple matrices are 1-based, so foo/bar add one.
    foo:= unapply(1+i,i):
    bar:= unapply(1+`mod`(i+d,dims[1]+2),i):

    u:=Vector([1$n]):
    L:=<LINALG:-IdentityMatrix(n),-LINALG:-Transpose(u)>:
    R:=<LINALG:-IdentityMatrix(n)-u.LINALG:-Transpose(u)/d | -u/d>:

    # R.L is the identity, hence Trace(Astar.Bstar.Cstar)=Trace(A.B.C).
    phi:=unapply(LINALG:-KroneckerProduct(
            LINALG:-IdentityMatrix(2),L).
            i.
            LINALG:-KroneckerProduct(LINALG:-IdentityMatrix(2),R),i):

    Astar:=phi(EXTRACTMAT(1,triad)):
    Bstar:=phi(EXTRACTMAT(2,triad)):
    Cstar:=phi(EXTRACTMAT(3,triad)):

    coeffs:=LITA_LOCAL7(d):

    return

    # First aggregation family: unbarred part.
    add(add(add(
        (Astar[foo(i),foo(j)]+Astar[foo(j),foo(k)]+Astar[foo(k),foo(i)])
        *(Bstar[foo(j),foo(k)]+Bstar[foo(k),foo(i)]+Bstar[foo(i),foo(j)])
        *(Cstar[foo(k),foo(i)]+Cstar[foo(i),foo(j)]+Cstar[foo(j),foo(k)])
    ,i=0..j),j=0..k-1),k=0..n)

    # First aggregation family: barred part.
    +add(add(add(
        (Astar[bar(i),bar(j)]+Astar[bar(j),bar(k)]+Astar[bar(k),bar(i)])
        *(Bstar[bar(j),bar(k)]+Bstar[bar(k),bar(i)]+Bstar[bar(i),bar(j)])
        *(Cstar[bar(k),bar(i)]+Cstar[bar(i),bar(j)]+Cstar[bar(j),bar(k)])
    ,i=0..j),j=0..k-1),k=0..n)

    # Symmetric half of the first aggregation family: unbarred part.
    +add(add(add(
        (Astar[foo(i),foo(j)]+Astar[foo(j),foo(k)]+Astar[foo(k),foo(i)])
        *(Bstar[foo(j),foo(k)]+Bstar[foo(k),foo(i)]+Bstar[foo(i),foo(j)])
        *(Cstar[foo(k),foo(i)]+Cstar[foo(i),foo(j)]+Cstar[foo(j),foo(k)])
    ,k=0..j-1),j=0..i),i=0..n)

    # Symmetric half of the first aggregation family: barred part.
    +add(add(add(
        (Astar[bar(i),bar(j)]+Astar[bar(j),bar(k)]+Astar[bar(k),bar(i)])
        *(Bstar[bar(j),bar(k)]+Bstar[bar(k),bar(i)]+Bstar[bar(i),bar(j)])
        *(Cstar[bar(k),bar(i)]+Cstar[bar(i),bar(j)]+Cstar[bar(j),bar(k)])
    ,k=0..j-1),j=0..i),i=0..n)

    # Second aggregation family.  The unbarred diagonal is already absent in
    # Schwartz--Zwecher.  For LITA, the mirrored diagonal is also removed
    # and replaced jointly with the old diagonal correction by the rank-7 block.
    +add(add(add(`if`(i=j and j=k and k=i,0,
        (Astar[bar(j),foo(k)]+Astar[foo(k),bar(i)]-Astar[foo(i),foo(j)])
        *(Bstar[foo(j),bar(k)]+Bstar[foo(k),foo(i)]+Bstar[bar(i),foo(j)])
        *(Cstar[foo(i),bar(j)]+Cstar[foo(j),foo(k)]-Cstar[bar(k),foo(i)])
    ),i=0..n),j=0..n),k=0..n)

    +add(add(add(`if`(i=j and j=k and k=i,0,
        (Astar[foo(j),bar(k)]+Astar[bar(k),foo(i)]-Astar[bar(i),bar(j)])
        *(Bstar[bar(j),foo(k)]+Bstar[bar(k),bar(i)]+Bstar[foo(i),bar(j)])
        *(Cstar[bar(i),foo(j)]+Cstar[bar(j),bar(k)]-Cstar[foo(k),bar(i)])
    ),i=0..n),j=0..n),k=0..n)

    # New diagonal local rank-7 replacement, one block for every i in [0,..,d-1].
    +add(add(
        LITA_FORM4(Astar,coeffs[r][1],i,foo,bar)
        *LITA_FORM4(Bstar,coeffs[r][2],i,foo,bar)
        *LITA_FORM4(Cstar,coeffs[r][3],i,foo,bar)
    ,r=1..7),i=0..n)

    # Off-diagonal correction blocks.  This is the same Winograd variant of
    # Strassen's algorithm used in the Schwartz--Zwecher skeleton.
    - d*(add(add(`if`(i=j,0,
        (Astar[foo(i),foo(j)]-Astar[bar(i),foo(j)])
            *(Bstar[foo(i),foo(j)]-Bstar[bar(i),foo(j)])
            *(Cstar[foo(i),foo(j)]+Cstar[bar(i),foo(j)])
        +(Astar[foo(i),foo(j)]+Astar[foo(i),bar(j)])
            *(Bstar[foo(i),foo(j)]+Bstar[foo(i),bar(j)])
            *(Cstar[foo(i),foo(j)]-Cstar[foo(i),bar(j)])
        +Astar[bar(i),bar(j)]*Bstar[bar(i),bar(j)]*Cstar[bar(i),bar(j)]
        -Astar[bar(i),foo(j)]
            *(Bstar[bar(i),foo(j)]+Bstar[bar(i),bar(j)]
                -Bstar[foo(i),foo(j)]-Bstar[foo(i),bar(j)])
            *Cstar[foo(i),bar(j)]
        -(Astar[bar(i),foo(j)]+Astar[bar(i),bar(j)]
            -Astar[foo(i),foo(j)]-Astar[foo(i),bar(j)])
            *Bstar[foo(i),bar(j)]*Cstar[bar(i),foo(j)]
        +Astar[foo(i),bar(j)]*Bstar[bar(i),foo(j)]
            *(Cstar[foo(i),bar(j)]+Cstar[bar(i),bar(j)]
                -Cstar[foo(i),foo(j)]-Cstar[bar(i),foo(j)])
        +(Astar[bar(i),foo(j)]-Astar[foo(i),foo(j)]-Astar[foo(i),bar(j)])
            *(Bstar[foo(i),foo(j)]+Bstar[foo(i),bar(j)]-Bstar[bar(i),foo(j)])
            *(Cstar[foo(i),foo(j)]-Cstar[foo(i),bar(j)]+Cstar[bar(i),foo(j)])
    ),i=0..n),j=0..n))
    :
end proc:
