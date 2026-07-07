#------------------------------------------------------------------------------
# -*-mapleV-*-
#------------------------------------------------------------------------------
# Even-dimensional Local Improvements to Trilinear Aggregation (LITA).
#
# This direct formula suppresses the certificate-position off(i,d-1,7) strip
# and absorbs it into the third factors using an explicit UV certificate.
# Rank: N^3/3 + 15*N^2/4 + 55*N/6 + 7.
#------------------------------------------------------------------------------

LITACOMPLEXITY:=proc(z)
    if type(z,numeric) then
        if not type(z,integer) or z < 18 or irem(z,2)<>0 then
            error "dimension should be an even integer at least 18"
        end if:
    end if:
    return z^3/3+15*z^2/4+55*z/6+7:
end proc:

LITA_LOCAL7:=proc(d)
local e,u,v,w,A,B,C,Ap,Bp,Cp,lam,a,b,c,ap,bp,cp,D0,D2,raw,t,r:
    e := [-1,0,0,1]:
    u := [d*(8-d)/(d-6), -2*d/(d-6), d*(d-2)/(d-6), 0]:
    v := [0,-1,1,0]:
    w := [1/2,0,1/2,0]:
    A  := d/(d-6):       B  := -(d-6)/d:       C  := (d-3)/(2*d):
    Ap := d*(d-7)/(d-6): Bp := 3/(2*d):        Cp := (d-6)/d:
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
    return v[1]*M[foo(i),foo(i)]
        + v[2]*M[bar(i),foo(i)]
        + v[3]*M[foo(i),bar(i)]
        + v[4]*M[bar(i),bar(i)]:
end proc:

LITA_REMOVED_CFACTORS:=proc(C,d,foo,bar,coeffs)
local D,m,R,T:
    D := d-1:
    R := Array(0..D-1):
    T := 0:
    for m from 0 to D-1 do
        R[m] := LITA_FORM4(C,coeffs[7][3],m+1,foo,bar):
        T := T+R[m]:
    end do:
    return [R,T]:
end proc:

LITA_REM:=proc(RT,a,d)
    if a=d-1 then return -RT[2] else return RT[1][a] end if:
end proc:

LITA_KEY:=proc(f,i,j,k) return cat(f,"|",i,"|",j,"|",k) end proc:

LITA_VERTEX:=proc(s,d)
local t,D:
    D := d-1:
    t := irem(s,d):
    if t=0 then return D else return t-1 end if:
end proc:

LITA_EDGE:=proc(r,d,offset)
local s:
    s := irem(r+offset,d*d):
    return [LITA_VERTEX(iquo(s,d),d),LITA_VERTEX(irem(s,d),d)]:
end proc:

LITA_AGG2_ENDPOINTS:=proc(d)
option remember:
local E,fams,ff,f,offset,h,r,used,cur,i,j,k,zero,consume:
    E := table():
    fams := ["agg2a","agg2b"]:
    for ff from 1 to 2 do
        f := fams[ff]:
        if f="agg2a" then
            offset := 1:
        elif irem(d,2)=0 then
            offset := 0:
        else
            h := (d-1)/2:
            offset := h*(d+1)+1:
        end if:

        r := 0: used := 0: cur := LITA_EDGE(r,d,offset):
        for k from 0 to d-1 do
            for j from 0 to d-1 do
                for i from 0 to d-1 do
                    zero := evalb(cur[1]=cur[2]):
                    if i=j and j=k then
                        if f="agg2a" then
                            consume := zero:
                        elif irem(d,2)=0 then
                            consume := evalb(zero and irem(i,2)=1):
                        else
                            consume := evalb(zero and i<>0):
                        end if:
                        if consume then
                            r := r+1: used := 0: cur := LITA_EDGE(r,d,offset):
                        end if:
                    else
                        E[LITA_KEY(f,i,j,k)] := cur:
                        used := used+1:
                        if used=2 then
                            r := r+1: used := 0: cur := LITA_EDGE(r,d,offset):
                        end if:
                    end if:
                end do:
            end do:
        end do:
    end do:
    return E:
end proc:

LITA_CERT_LABELS:=proc(d)
local D,R,L,q,i,j,k,r:
    D := d-1:
    R := 4*binomial(d+1,3)+2*(d^3-d)+7*d^2:
    L := Array(1..R):
    q := 0:

    for k from 0 to D do
        for j from 0 to k-1 do
            for i from 0 to j do
                q := q+1: L[q] := ["agg1F",i,j,k]:
            end do:
        end do:
    end do:
    for k from 0 to D do
        for j from 0 to k-1 do
            for i from 0 to j do
                q := q+1: L[q] := ["agg1B",i,j,k]:
            end do:
        end do:
    end do:
    for i from 0 to D do
        for j from 0 to i do
            for k from 0 to j-1 do
                q := q+1: L[q] := ["agg1Fsym",i,j,k]:
            end do:
        end do:
    end do:
    for i from 0 to D do
        for j from 0 to i do
            for k from 0 to j-1 do
                q := q+1: L[q] := ["agg1Bsym",i,j,k]:
            end do:
        end do:
    end do:
    for k from 0 to D do
        for j from 0 to D do
            for i from 0 to D do
                if not (i=j and j=k) then
                    q := q+1: L[q] := ["agg2a",i,j,k]:
                end if:
            end do:
        end do:
    end do:
    for k from 0 to D do
        for j from 0 to D do
            for i from 0 to D do
                if not (i=j and j=k) then
                    q := q+1: L[q] := ["agg2b",i,j,k]:
                end if:
            end do:
        end do:
    end do:
    for i from 0 to D do
        for r from 1 to 7 do
            q := q+1: L[q] := ["l7",i,r,0]:
        end do:
    end do:
    for j from 0 to D do
        for i from 0 to D do
            if i<>j then
                for r from 1 to 7 do
                    q := q+1: L[q] := ["off",i,j,r]:
                end do:
            end if:
        end do:
    end do:
    if q<>R then error "internal certificate label count mismatch" end if:
    return L:
end proc:

LITA_ROW_CERT_TABLE:=proc(d)
option remember:
local D,labels,P,q,i,j,k,r,forward,backward,shift,fam:
    D := d-1:
    labels := LITA_CERT_LABELS(d):
    P := table():
    q := 0:

    for i from 0 to D do
        for j from 0 to D do
            for k from 0 to D do
                forward := evalb(i<=j and j<k):
                backward := evalb(k<j and j<=i):
                if forward or backward then
                    for shift from 0 to 1 do
                        if forward then
                            if shift=0 then fam := "agg1F" else fam := "agg1B" end if:
                        else
                            if shift=0 then fam := "agg1Fsym" else fam := "agg1Bsym" end if:
                        end if:
                        q := q+1: P[LITA_KEY(fam,i,j,k)] := labels[q]:
                    end do:
                end if:
            end do:
        end do:
    end do:

    for i from 0 to D do
        for j from 0 to D do
            for k from 0 to D do
                if not (i=j and j=k) then
                    q := q+1: P[LITA_KEY("agg2a",i,j,k)] := labels[q]:
                    q := q+1: P[LITA_KEY("agg2b",i,j,k)] := labels[q]:
                end if:
            end do:
        end do:
    end do:

    for i from 0 to D do
        for j from 0 to D do
            if i<>j then
                for r from 1 to 7 do
                    q := q+1: P[LITA_KEY("off",i,j,r)] := labels[q]:
                end do:
            end if:
        end do:
    end do:

    for i from 0 to D do
        for r from 1 to 7 do
            q := q+1: P[LITA_KEY("l7",i,r,0)] := labels[q]:
        end do:
    end do:
    return P:
end proc:

LITA_CERT_CFACTOR:=proc(kind,c0,a,b,t,RT,E,d)
local D,L,S,T,ep,v,corr:
    D := d-1: L := 2*d: T := RT[2]:

    if kind="agg1F" or kind="agg1B" or kind="agg1Fsym" or kind="agg1Bsym" then
        return c0:
    end if:

    if kind="agg2a" or kind="agg2b" then
        ep := E[LITA_KEY(kind,a,b,t)]:
        return c0 + 2*(LITA_REM(RT,ep[1],d)-LITA_REM(RT,ep[2],d)):
    end if:

    if kind="l7" then
        if a=D then
            v := [[-2,1,2,1,2,-1,-1],[-1,0,1,1,1,0,-1]]:
            return c0 + L*(v[1][b]*RT[1][0]+v[2][b]*(T-RT[1][0])):
        else
            v := [[2,-1,-2,-1,-2,1,1],[1,-1,-1,0,-1,1,0]]:
            return c0 + L*(v[1][b]*RT[1][a]+v[2][b]*(T-RT[1][a])):
        end if:
    end if:

    if kind<>"off" then return c0 end if:

    S := 2*d^2/(d-6):
    if a=D and b=D-1 then
        v := [0,1,S,4,4,-S,1]:
        return c0 + v[t]*T:
    elif a<=D-1 and b=D then
        v := [0,-1,-S,-4,-4,S,0]:
        return c0 + v[t]*RT[1][a]:
    elif a=D and 0<=b and b<=D-2 then
        v := [-2,1,2,1,2,-1,-1]:
        corr := L*v[t]*RT[1][b+1]:
        v := [-1,0,1,1,1,0,-1]:
        return c0 + corr + L*v[t]*(T-RT[1][b+1]):
    end if:

    corr := 0:
    if 0<=b and b<=D-1 and 0<=a and a<=D-1 and a<>b then
        v := [-1,1,1,0,1,-1,0]:
        corr := corr + L*v[t]*RT[1][b]:
        v := [1,0,-1,-1,-1,0,1]:
        corr := corr + L*v[t]*RT[1][a]:
    end if:
    return c0 + corr:
end proc:

LITA_ROW_CFACTOR:=proc(kind,c0,a,b,t,P,RT,E,d)
local lab:
    lab := P[LITA_KEY(kind,a,b,t)]:
    if lab[1]="off" and lab[3]=d-1 and lab[4]=7 then
        return 0:
    end if:
    return LITA_CERT_CFACTOR(lab[1],c0,lab[2],lab[3],lab[4],RT,E,d):
end proc:

LITA_OFF_FORM:=proc(M,v,i,j,foo,bar)
    return v[1]*M[foo(i),foo(j)]
        + v[2]*M[bar(i),foo(j)]
        + v[3]*M[foo(i),bar(j)]
        + v[4]*M[bar(i),bar(j)]:
end proc:

LITA_OFF:=proc(A,B,C,i,j,d,foo,bar,P,RT,E)
local USTR,VSTR,WSTR,csign,r,z,cw,out,u,v,c0:
    if i=j then return 0 end if:

    USTR := [
        [1,0,0,1], [0,0,0,1], [0,1,0,1], [1,-1,0,0],
        [1,0,1,0], [1,0,0,0], [0,0,1,-1]
    ]:
    VSTR := [
        [1,0,0,1], [1,0,1,0], [0,0,-1,1], [0,0,0,1],
        [1,-1,0,0], [0,-1,0,-1], [1,0,0,0]
    ]:
    WSTR := [
        [1,0,0,1], [-1,1,0,0], [-1,0,0,0], [-1,0,-1,0],
        [0,0,0,-1], [0,0,-1,1], [0,1,0,1]
    ]:
    csign := [1,-1,-1,1]:

    out := 0:
    for r from 1 to 7 do
        cw := [seq(-d*WSTR[r][z]*csign[z],z=1..4)]:
        u := LITA_OFF_FORM(A,USTR[r],i,j,foo,bar):
        v := LITA_OFF_FORM(B,VSTR[r],i,j,foo,bar):
        c0 := LITA_OFF_FORM(C,cw,i,j,foo,bar):
        out := out + u*v*LITA_ROW_CFACTOR("off",c0,i,j,r,P,RT,E,d):
    end do:
    return out:
end proc:

LITA:=proc(triad :: TRIAD)

description
    "Given a triad, return the trilinear form of the even-dimensional LITA",
    "direct pair-reduced local improvement to Schwartz--Zwecher trilinear aggregation",
    COPYLEFT :

local i,j,k,r,dims,n,d,D,foo,bar,u,phi,L,R,A,B,C,coeffs,P,RT,E:

    dims := TRIADDIM(triad):
    if nops(convert(dims,`set`))<>1 then error "triad should be square" end if:
    if map(irem,dims,2)<>[0,0,0] then error "triad's dimension should be a multiple of 2" end if:
    if dims[1] < 18 then error "triad's dimension should be at least 18" end if:

    n:=dims[1]/2: d:=n+1: D:=d-1:
    foo:=unapply(1+i,i):
    bar:=unapply(1+`mod`(i+d,dims[1]+2),i):

    u:=Vector([1$n]):
    L:=<LINALG:-IdentityMatrix(n),-LINALG:-Transpose(u)>:
    R:=<LINALG:-IdentityMatrix(n)-u.LINALG:-Transpose(u)/d | -u/d>:
    phi:=unapply(LINALG:-KroneckerProduct(LINALG:-IdentityMatrix(2),L).
            i.LINALG:-KroneckerProduct(LINALG:-IdentityMatrix(2),R),i):

    A:=phi(EXTRACTMAT(1,triad)):
    B:=phi(EXTRACTMAT(2,triad)):
    C:=phi(EXTRACTMAT(3,triad)):
    coeffs:=LITA_LOCAL7(d):
    P:=LITA_ROW_CERT_TABLE(d):
    RT:=LITA_REMOVED_CFACTORS(C,d,foo,bar,coeffs):
    E:=LITA_AGG2_ENDPOINTS(d):

    return

    add(add(add(
        (A[foo(i),foo(j)]+A[foo(j),foo(k)]+A[foo(k),foo(i)])
        *(B[foo(j),foo(k)]+B[foo(k),foo(i)]+B[foo(i),foo(j)])
        *(C[foo(k),foo(i)]+C[foo(i),foo(j)]+C[foo(j),foo(k)])
    ,i=0..j),j=0..k-1),k=0..D)

    +add(add(add(
        (A[bar(i),bar(j)]+A[bar(j),bar(k)]+A[bar(k),bar(i)])
        *(B[bar(j),bar(k)]+B[bar(k),bar(i)]+B[bar(i),bar(j)])
        *(C[bar(k),bar(i)]+C[bar(i),bar(j)]+C[bar(j),bar(k)])
    ,i=0..j),j=0..k-1),k=0..D)

    +add(add(add(
        (A[foo(i),foo(j)]+A[foo(j),foo(k)]+A[foo(k),foo(i)])
        *(B[foo(j),foo(k)]+B[foo(k),foo(i)]+B[foo(i),foo(j)])
        *(C[foo(k),foo(i)]+C[foo(i),foo(j)]+C[foo(j),foo(k)])
    ,k=0..j-1),j=0..i),i=0..D)

    +add(add(add(
        (A[bar(i),bar(j)]+A[bar(j),bar(k)]+A[bar(k),bar(i)])
        *(B[bar(j),bar(k)]+B[bar(k),bar(i)]+B[bar(i),bar(j)])
        *(C[bar(k),bar(i)]+C[bar(i),bar(j)]+C[bar(j),bar(k)])
    ,k=0..j-1),j=0..i),i=0..D)

    +add(add(add(`if`(i=j and j=k,0,
        (A[bar(j),foo(k)]+A[foo(k),bar(i)]-A[foo(i),foo(j)])
        *(B[foo(j),bar(k)]+B[foo(k),foo(i)]+B[bar(i),foo(j)])
        *LITA_ROW_CFACTOR("agg2a",C[foo(i),bar(j)]+C[foo(j),foo(k)]-C[bar(k),foo(i)],i,j,k,P,RT,E,d)
    ),i=0..D),j=0..D),k=0..D)

    +add(add(add(`if`(i=j and j=k,0,
        (A[foo(j),bar(k)]+A[bar(k),foo(i)]-A[bar(i),bar(j)])
        *(B[bar(j),foo(k)]+B[bar(k),bar(i)]+B[foo(i),bar(j)])
        *LITA_ROW_CFACTOR("agg2b",C[bar(i),foo(j)]+C[bar(j),bar(k)]-C[foo(k),bar(i)],i,j,k,P,RT,E,d)
    ),i=0..D),j=0..D),k=0..D)

    +add(add(
        LITA_FORM4(A,coeffs[r][1],i,foo,bar)
        *LITA_FORM4(B,coeffs[r][2],i,foo,bar)
        *LITA_ROW_CFACTOR("l7",LITA_FORM4(C,coeffs[r][3],i,foo,bar),i,r,0,P,RT,E,d)
    ,r=1..7),i=0..D)

    +add(add(LITA_OFF(A,B,C,i,j,d,foo,bar,P,RT,E),i=0..D),j=0..D)
    :
end proc:
