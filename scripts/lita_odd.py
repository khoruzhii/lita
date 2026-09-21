"""
Rational LITA schemes for odd square matrix multiplication.

R(N) = N^3/3 + 15*N^2/4 + 14*N/3 + 13/4.

API:
    rank = lita_odd_rank(N)
    scheme = lita_odd(N)
    scheme.save("scheme.npz")

N must be odd with 7 <= N < 32.
"""

import argparse
from array import array
from itertools import combinations, permutations
import json
from math import comb, gcd
from pathlib import Path

import numpy as np


# ---------- Integer linear forms ----------


def _check_dimension(N):
    if type(N) is not int or N < 7 or N >= 32 or N % 2 != 1:
        raise ValueError("N must be an odd integer with 7 <= N < 32")


def lita_odd_rank(N):
    _check_dimension(N)
    return (4*N**3 + 45*N**2 + 56*N + 39)//12


def _add_scaled(row, other, scale):
    if not scale:
        return row
    for index, value in other.items():
        value = row.get(index, 0) + scale * value
        if value:
            row[index] = value
        elif index in row:
            del row[index]
    return row


def _sum(forms):
    result = {}
    for form in forms:
        _add_scaled(result, form, 1)
    return _Form(result)


class _Form(dict):
    """Integer linear forms on a fixed rational coefficient grid."""

    def __add__(self, other):
        return _add_scaled(_Form(self), other, 1)

    def __sub__(self, other):
        return _add_scaled(_Form(self), other, -1)

    def __neg__(self):
        return self * -1

    def __mul__(self, scalar):
        return _Form((i, c * scalar) for i, c in self.items() if c * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        result = _Form()
        for i, value in self.items():
            quotient, remainder = divmod(value, scalar)
            if remainder:
                raise ArithmeticError("nonintegral coefficient on the LITA grid")
            if quotient:
                result[i] = quotient
        return result


# ---------- Signed blocks and row-column parameters ----------


class _Coordinates:
    """Forms in original row-major coordinates, stored as grid*F.

    Upper vertices 0 and 1 duplicate one row and average two columns.
    The standard weighted block completion is shared by all three inputs.
    All divisions on the coefficient grid are exact.
    """

    def __init__(self, N):
        _check_dimension(N)
        self.N, self.D = N, (N+1) // 2
        self.w = 2-self.D
        self.grid = 384*(self.D-2)**2*(self.D-3)
        self._entries = {}

    def weight(self, i):
        return self.w if i == self.D else 1

    def entry(self, block, i, j):
        key = block, i, j
        if key not in self._entries:
            D, N = self.D, self.N
            br, bc = divmod(block, 2)
            if i < D and j < D:
                row = D-1+i if br else max(0,i-1)
                col = D-1+j if bc else max(0,j-1)
                value = self.grid * (-1 if bc else 1)
                if bc == 0 and j < 2:
                    value //= 2
                result = _Form({row*N+col: value})
            elif i < D:
                result = _sum(self.entry(2*br+b, i, k)
                              for b in range(2) for k in range(D))/(2*(D-2))
            elif j < D:
                result = _sum(self.entry(2*a+bc, k, j)
                              for a in range(2) for k in range(D))/(2*(D-2))
            else:
                result = _sum(self.entry(b, r, s) for b in range(4)
                              for r in range(D) for s in range(D))/(4*(D-2)**2)
            self._entries[key] = result
        return self._entries[key]


class _View:
    """Opposite pole normalizations, evaluated directly in the final blocks."""

    def __init__(self, coordinates, axis):
        self.coordinates = coordinates
        self.D = D = coordinates.D
        X = coordinates.entry
        if axis == 0:
            trace = _Form({i*coordinates.N+i: coordinates.grid for i in range(coordinates.N)})
            self.gamma = (X(1,0,D)+X(2,D,0)+X(0,0,0)+X(0,D,D)-trace/(D-2))/3
        else:
            self.gamma = (X(0,0,0)+X(1,0,2)+X(2,2,0)+X(3,2,2))/3
        self.r0 = [X(0,i,0)+X(2,i,0) for i in range(D)]
        self.r0.append(_sum(self.r0)/(D-2))
        self.c0 = [self.r0[i]-X(0,i,i)+self.gamma for i in range(D)]
        self.c1 = [X(3,i,i)-self.gamma for i in range(D)]
        closing_sum = _sum(self.c0+self.c1)/(D-2)
        if axis == 0:
            last = self.r0[D]-X(0,D,D)+self.gamma
        elif axis == 1:
            last = closing_sum+X(0,D,D)+self.r0[D]+self.gamma
        else:
            last = closing_sum-X(0,D,D)+self.gamma
        self.c0.append(last)
        self.c1.append(closing_sum-last)
        Y = self._base_entry
        self.upper = [_Form() for _ in range(D+1)]
        self.field = [_Form() for _ in range(D+1)]
        # Normalize pi(pole,j) to gamma, including the closing column.
        pole = 1 if axis == 0 else 0
        for j in range(2,D+1):
            self.field[j] = (2*pole-1)*(Y(1,pole,j)+Y(2,j,pole)-self.gamma)
        if axis == 0:
            self.upper[2] = Y(2,2,3)+Y(1,3,0)-Y(0,0,2)
            for j in range(3,D):
                self.upper[j] = Y(0,0,1)+Y(0,1,j)+Y(0,j,0)
            self.upper[D] = Y(0,1,D)+Y(1,1,D)
            self.field[1] = -Y(0,0,1)-Y(1,0,1)
        elif axis == 1:
            for j in range(3,D):
                self.upper[j] = -Y(0,0,1)-Y(0,1,j)-Y(0,j,0)
            self.upper[D] = -Y(0,D,1)-Y(0,1,0)-Y(0,0,D)
            self.field[0] = Y(0,1,0)+Y(1,1,0)
        # Weighted closure of the combined field determines its last free value.
        free = 1 if axis == 1 else 0
        self.field[free] = ((D-2)*(self.upper[D]+self.field[D])
                            -_sum(self.upper[:D]+self.field[:D]))
        self._entries = {}
        self.row_sums = {b: tuple(_sum(coordinates.weight(k)*self.entry(b,j,k)
                                     for k in range(D+1)) for j in range(D+1))
                         for b in (0,3)}
        self.column_sums = {b: tuple(_sum(coordinates.weight(k)*self.entry(b,k,i)
                                        for k in range(D+1)) for i in range(D+1))
                            for b in (0,3)}

    def _base_entry(self, block, i, j):
        X = self.coordinates.entry(block,i,j)
        if block == 0:
            return X-self.r0[i]+self.c0[j]
        if block == 1:
            return X+self.r0[i]+self.c1[j]
        return X-(self.c0[j] if block == 2 else self.c1[j])

    def entry(self, block, i, j):
        key = block, i, j
        if key not in self._entries:
            result = self._base_entry(block,i,j)
            if block < 2 and i < 2:
                correction = self.upper[j] if block == 0 else self.field[j]
                result = result+(1-2*i)*correction
            self._entries[key] = result
        return self._entries[key]

    def half_sum(self, block, i, j):
        return (self.row_sums[block][j]+self.column_sums[block][i])/2


def _merge(first, second, axis, description):
    """Combine two unweighted products along their only unshared axis."""
    if any(first[a] != second[a] for a in range(3) if a != axis):
        raise ArithmeticError(description+": shared factors differ")
    row = list(first)
    row[axis] = row[axis]+second[axis]
    return tuple(row)


def _cycle_midpoint(first, second):
    """Keep twice the midpoint product; its two-field error is pooled below."""
    if first[2] != second[2]:
        raise ArithmeticError("upper cycles on C differ")
    return first[0]+second[0], (first[1]+second[1])/2, first[2]


# ---------- One weighted aggregation identity ----------


def _cycle(view, i, j, k, barred):
    b = 3*barred
    return _sum(view.entry(b,u,v) for u,v in ((i,j),(j,k),(k,i)))


def _mixed(view, i, j, k, barred):
    X = view.entry
    return X(2-barred,j,k)+X(1+barred,k,i)-X(3*barred,i,j)


def _mixed_factors(views, i, j, k, barred):
    A,B,C = views
    return (_mixed(A,i,j,k,barred), _mixed(B,k,i,j,barred),
            (-1 if barred else 1)*_mixed(C,j,k,i,barred))


def _weighted(term, coordinates, ids):
    return tuple(f*coordinates.weight(i) for f,i in zip(term,ids))


def _cyclic(recipes, shifts=(0,1,2)):
    """Rotate recipes first, evaluating each in its destination input."""
    for shift in shifts:
        yield tuple(recipes[a][(a+shift)%3] for a in range(3))


def _triangle_terms(c, views):
    D = c.D
    for vertices in combinations(range(D+1),3):
        for i,j,k in permutations(vertices):
            for barred in (0,1):
                if i < j < k or k < j < i:
                    other_ids = None
                    retain = True
                    if not barred and 1 in vertices:
                        retain = 0 in vertices and i < j < k and k == 2
                        if retain:
                            other_ids = (k,j,i)
                    elif not barred and 0 in vertices:
                        other_ids = tuple(1 if v == 0 else v for v in (i,j,k))
                    if retain:
                        row = tuple(_cycle(v,i,j,k,barred) for v in views)
                        if other_ids is not None:
                            other = tuple(_cycle(v,*other_ids,0) for v in views)
                            row = _cycle_midpoint(row, other)
                        row = row[0],row[1],(-1 if barred else 1)*row[2]
                        yield _weighted(row,c,(i,j,k))
                if barred and ((i >= 2 and j == 1 and k == 0)
                               or (i == 0 and j >= 2 and k == 1)
                               or (i == 0 and j == 1 and k >= 2)):
                    continue  # Included in the pooled pole group.
                row = _mixed_factors(views,i,j,k,barred)
                if not barred and (i,j,k) == (0,1,D):
                    other = _mixed_factors(views,1,1,D,0)
                    row = _merge(row, other, 0, "pole-one boundary absorption")
                yield _weighted(row,c,(i,j,k))


def _seeds(view, i, j):
    x0,x1,x2,x3 = (view.entry(b,i,j) for b in range(4))
    return (x3-x0, -x2-x3, -x1-x3,
            x0+view.half_sum(3,i,j), -x3-view.half_sum(0,i,j))


def _heptad(seeds):
    yield tuple(h[0] for h in seeds)
    yield from _cyclic(tuple((h[1],h[2],h[3]) for h in seeds))
    yield from _cyclic(tuple((h[4],-h[0]-h[2],-h[0]-h[1]) for h in seeds))


# ---------- Ordinary-edge prototypes and pooled vertex terms ----------


def _edge_recipe(view, i, j):
    Y,g = view.entry,view.gamma
    a,b,c,d = (Y(k,i,j) for k in range(4))
    ap,bp,cp,dp = (Y(k,j,i) for k in range(4))
    ui,uj = Y(1,i,i)+g,Y(1,j,j)+g
    li,lj = Y(2,i,i)+g,Y(2,j,j)+g
    pi = b+cp-g
    return ((-c-d,b+d,2*(dp+g-view.half_sum(3,i,j))),
            (c+d+li,pi,bp+dp+a+c+lj+ui),
            (a+c,2*(ap+g-view.half_sum(0,i,j)),a+b),
            (-a-c-lj,ap+bp+c+d+li+uj,pi))


def _edge_terms(c, views):
    for i,j in combinations(range(c.D),2):
        for left,right in ((i,j),(j,i)):
            recipes = tuple(_edge_recipe(v,left,right) for v in views)
            for r in range(4):
                rows = tuple(recipe[r] for recipe in recipes)
                for shift,row in enumerate(_cyclic(rows)):
                    if r < 2 and (((left,right) == (1,0) and shift in (1,2))
                                  or ((left,right) == (0,1) and shift == 0)):
                        continue  # Use the orientation matching the common pole factor.
                    if (left,right,r) == (1,2,2) and shift in (0,1):
                        continue
                    if (left,right,r) == (0,2,2) and shift in (0,1):
                        other = tuple(_edge_recipe(v,1,2)[2] for v in views)
                        other = tuple(other[a][(a+shift)%3] for a in range(3))
                        row = _merge(row,other,(-shift)%3,"reference-edge identification")
                    yield row


def _pole_terms(coordinates, views):
    """Pool orientation 0->1 on A and 1->0 on B and C."""
    for p,q,shifts in ((0,1,(0,)),(1,0,(1,2))):
        for j in range(1,coordinates.D+1):
            recipes = []
            for view in views:
                Y,g = view.entry,view.gamma
                z = Y(2,p,q)+Y(3,p,q)
                if j == 1:
                    u = Y(2,q,p)-g-Y(3,p,q)
                    v = Y(1,q,p)+Y(3,q,p)+Y(1,p,p)+g
                else:
                    u = Y(2,q,j)-Y(3,j,p)-Y(3,p,q)
                    v = Y(1,j,p)+Y(2,p,q)-Y(3,q,j)
                recipes.append((z,u,v))
            for u,v,w in _cyclic(recipes,shifts):
                yield coordinates.weight(j)*u,v,w


def _upper_error_terms(c, views):
    """Sum the two-field cycle errors, including the closing boundary term."""
    A,B,C = views
    D,Y = c.D,C.entry
    for i in range(3,D+1):
        H = (C.row_sums[0][i]+C.column_sums[0][0]-2*Y(0,i,0)
             -2*C.gamma-c.weight(i)*_cycle(C,0,i,i,0))
        if i == D:
            H -= (D-3)*_boundary_cycle(C,0,0)
        yield 2*c.weight(i)*A.upper[i],B.upper[i],H


def _vertex_terms(coordinates, views):
    D,w = coordinates.D,coordinates.w
    for i in range(D):
        recipes = []
        for v in views:
            Y,g = v.entry,v.gamma
            u,l = Y(1,i,i)+g,Y(2,i,i)+g
            k0 = v.half_sum(0,i,i)+w*(Y(1,i,D)+Y(2,D,i))+(D-4)*g
            k3 = v.half_sum(3,i,i)+w*(Y(1,D,i)+Y(2,i,D))+(D-4)*g
            recipes.append(((2*u,l,k0),(-2*l,u,k3)))
        for r in range(2):
            yield from _cyclic(tuple(recipe[r] for recipe in recipes))


# ---------- Boundary edges and the closing vertex ----------


def _boundary_cycle(view, i, barred):
    D,b = view.D,3*barred
    return (view.entry(b,i,D)+view.entry(b,D,i)
            +((D-2)*view.entry(b,D,D)-view.gamma)/(D-3))


def _boundary_terms(c, views):
    D,w = c.D,c.w
    for i in range(D):
        for barred in (0,1):
            if not barred and i == 1:
                continue
            row = tuple(_boundary_cycle(v,i,barred) for v in views)
            if not barred and i == 0:
                other = tuple(_boundary_cycle(v,1,0) for v in views)
                row = _cycle_midpoint(row, other)
            yield (D-2)*(D-3)*row[0],row[1],(-1 if barred else 1)*row[2]
        for ids in ((i,i,D),(i,D,i),(i,D,D),(D,i,i),(D,i,D),(D,D,i)):
            for barred in (0,1):
                if not barred and ids == (1,1,D):
                    # Absorbed into the positive (0,1,D) corner product.
                    continue
                yield _weighted(_mixed_factors(views,*ids,barred),c,ids)
        for left,right in ((i,D),(D,i)):
            for u,v,z in _heptad(tuple(_seeds(view,left,right) for view in views)):
                yield c.weight(left)*u,c.weight(right)*v,2*z


def _center_terms(coordinates, views):
    D = coordinates.D
    t = D-2
    recipes,polar = [],[]
    for view in views:
        a,b,c,d = (view.entry(k,D,D) for k in range(4))
        g,q = view.gamma,coordinates.entry(0,D,D)
        h = d-a
        eta = 3*g-4*(5*D-12)*q-2*t*(b-c)
        L = (2*g-a-d)/(2*(D-3))-4*q
        U = a+view.half_sum(3,D,D)+t*(3*g-10*q+2*c+2*d-eta)
        V = -d-view.half_sum(0,D,D)-t*(3*g-10*q+2*a+2*c-eta)
        recipes.append(((c+d+h/(D-3),b+d,U),
                        (U+2*t*(a+b),c+d,b+d),
                        (b+d,U,c+d),
                        (V,a+b,a+c-h/(D-3)),
                        (a+c,V,a+b),
                        (a+b,a+c,V-2*t*(b+d))))
        polar.append((eta,h,L))
    for r in range(6):
        u,v,z = (recipes[axis][r][axis] for axis in range(3))
        yield 2*t*t*u,v,z
    for epsilon in (-1,1):
        for delta in (-1,1):
            u,v,z = (eta+epsilon*h+delta*L for eta,h,L in polar)
            yield (-t**3*epsilon*delta*u)/4,v,z


# ---------- Sparse serialization ----------


class SparseAxis:
    def __init__(self):
        self.indptr = array("q", (0,))
        self.indices = array("i")
        self.numerators = array("q")
        self.denominators = array("q")

    @property
    def rows(self):
        return len(self.indptr) - 1

    def append(self, row, denominator, transpose=0):
        if denominator <= 0:
            raise ValueError("the common denominator must be positive")
        if transpose:
            N = transpose
            items = (((index % N) * N + index // N, value) for index, value in row.items())
        else:
            items = row.items()
        start = len(self.indices)
        try:
            for index, numerator in sorted(items):
                if not numerator:
                    continue
                common = gcd(abs(numerator), denominator)
                self.indices.append(index)
                self.numerators.append(numerator // common)
                self.denominators.append(denominator // common)
        except OverflowError:
            del self.indices[start:]
            del self.numerators[start:]
            del self.denominators[start:]
            raise OverflowError("LITA CSR indices or reduced coefficients exceed their integer storage") from None
        self.indptr.append(len(self.indices))

    def arrays(self, name):
        return {
            f"{name}_indptr": np.frombuffer(self.indptr, dtype=np.int64),
            f"{name}_indices": np.frombuffer(self.indices, dtype=np.int32),
            f"{name}_numerators": np.frombuffer(self.numerators, dtype=np.int64),
            f"{name}_denominators": np.frombuffer(self.denominators, dtype=np.int64),
        }


class Scheme:
    def __init__(self, N):
        _check_dimension(N)
        self.N = N
        self.U = SparseAxis()
        self.V = SparseAxis()
        self.W = SparseAxis()
        self.statistics = {}

    @property
    def rank(self):
        return self.U.rows

    def append(self, u, v, w):
        self.U.append(*u)
        self.V.append(*v)
        self.W.append(*w, transpose=self.N)

    def save(self, path):
        if not self.U.rows == self.V.rows == self.W.rows:
            raise ValueError("the three factor axes must have equal row counts")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        arrays.update(self.U.arrays("u"))
        arrays.update(self.V.arrays("v"))
        arrays.update(self.W.arrays("w"))
        arrays["metadata_json"] = np.asarray(json.dumps({
            "coefficient_field": "Q",
            "format": "fmmp.qcsr.v1",
            "rank": self.rank,
            "tensor": [self.N, self.N, self.N],
            "is_complete_matrix_multiplication_scheme": True,
            "axis_convention": "U,V read row-major inputs; W writes row-major C=AB.",
            "construction": "LITA odd row-column aggregation",
        }, separators=(",", ":")))
        np.savez_compressed(path, **arrays)


def iter_terms(N, statistics=None):
    """Yield rational factors for tr(A B C) on a common positive grid."""
    c = _Coordinates(N)
    views = tuple(_View(c,a) for a in range(3))
    D = c.D
    families = (
        ("triangles", _triangle_terms, 16*comb(D+1,3)-(D-1)*(D+8)),
        ("edges", _edge_terms, 24*comb(D,2)-20*D+18),
        ("poles", _pole_terms, 3*D),
        ("upper_errors", _upper_error_terms, D-2),
        ("vertices", _vertex_terms, 6*(D-2)),
        ("boundary", _boundary_terms, 28*D-20),
        ("center", _center_terms, 10),
    )
    counts = {}
    for name,family,expected in families:
        count = 0
        for term in family(c,views):
            if not all(term):
                continue
            count += 1
            yield tuple((f,c.grid) for f in term)
        if count != expected:
            raise ArithmeticError(f"{name}: emitted {count}, expected {expected}")
        counts[name] = count
    if sum(counts.values()) != lita_odd_rank(N):
        raise ArithmeticError("odd-LITA total length mismatch")
    if statistics is not None:
        statistics.update(N=N,D=D,m=D-2,rank=sum(counts.values()),
                          family_counts=counts,numerator_grid=c.grid,
                          construction="odd LITA in the original even ansatz")


def lita_odd(N):
    scheme = Scheme(N)
    for term in iter_terms(N,scheme.statistics):
        scheme.append(*term)
    return scheme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int)
    parser.add_argument("output", nargs="?")
    parser.add_argument("--rank-only", action="store_true")
    args = parser.parse_args()
    if args.rank_only:
        print(f"N={args.N} rank={lita_odd_rank(args.N)}")
        return
    scheme = lita_odd(args.N)
    if args.output:
        scheme.save(args.output)
    print(f"N={args.N} rank={scheme.rank}")
    if args.output:
        print(Path(args.output))


if __name__ == "__main__":
    main()
