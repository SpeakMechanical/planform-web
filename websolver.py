"""3D stress analysis of an extruded outline, in pure NumPy/SciPy.

This is the solver the Planform website runs in the browser through Pyodide,
where Gmsh is not available. Planform 2D meshes the outline into triangles;
this module stacks them into layers, splits each prism into three tetrahedra,
adds mid-edge nodes (10-node tets) and solves linear elasticity.

Units: mm, N, MPa. Voigt order xx, yy, zz, yz, xz, xy (engineering shear).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# 10-node tet: corners 0-3, then mid-edge nodes on these corner pairs
EDGES = np.array([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)])
FACES = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])  # face opposite each corner
_A, _B = 0.5854101966249685, 0.1381966011250105
GAUSS = np.array([[_A, _B, _B, _B], [_B, _A, _B, _B], [_B, _B, _A, _B], [_B, _B, _B, _A]])  # volume coords
WEIGHT = 1 / 24
LOCAL = np.vstack([np.eye(4), (np.eye(4)[EDGES[:, 0]] + np.eye(4)[EDGES[:, 1]]) / 2])  # node volume coords
DL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)  # d(L1..L4)/d(xi, eta, zeta)


def grads(L):
    """Shape-function gradients in reference coordinates at volume coords L (4,) -> (10, 3)."""
    g = np.empty((10, 3))
    g[:4] = (4 * L - 1)[:, None] * DL
    i, j = EDGES[:, 0], EDGES[:, 1]
    g[4:] = 4 * (L[j][:, None] * DL[i] + L[i][:, None] * DL[j])
    return g


def extrude(x, y, tris, thickness, layers):
    """Tet10 mesh of the outline extruded in +z. Returns nodes, elems, base vertex pairs."""
    n2, x, y = len(x), np.asarray(x, float), np.asarray(y, float)
    tris = np.sort(np.asarray(tris, np.intp), axis=1)  # sorting makes neighbouring prisms split alike
    p, q, r = tris.T
    tets = []
    for k in range(layers):
        b, t = k * n2, (k + 1) * n2
        tets += [np.stack([p + b, q + b, r + b, r + t], 1),
                 np.stack([p + b, q + b, q + t, r + t], 1),
                 np.stack([p + b, p + t, q + t, r + t], 1)]
    tets = np.vstack(tets)
    z = np.repeat(np.linspace(0, thickness, layers + 1), n2)
    corners = np.column_stack([np.tile(x, layers + 1), np.tile(y, layers + 1), z])
    # positive volume for every tet
    a, b_, c, d = (corners[tets[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", b_ - a, np.cross(c - a, d - a)) < 0
    tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
    # mid-edge nodes
    pairs = np.sort(tets[:, EDGES].reshape(-1, 2), axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    nc = len(corners)
    nodes = np.vstack([corners, (corners[uniq[:, 0]] + corners[uniq[:, 1]]) / 2])
    elems = np.hstack([tets, nc + inv.reshape(-1, 6)])
    base = np.vstack([np.column_stack([np.arange(nc) % n2] * 2), uniq % n2])  # 2D vertices under each node
    return nodes, elems, base


def _B(nodes, elems, pts):
    """Strain-displacement (E, q, 6, 30) and det J (E, q) at reference points `pts` (q, 4)."""
    dN = np.stack([grads(L) for L in pts])
    J = np.einsum("eai,qak->eqik", nodes[elems], dN)
    detJ = np.linalg.det(J)
    g = np.einsum("qak,eqki->eqai", dN, np.linalg.inv(J))
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    B = np.zeros(g.shape[:2] + (6, 30))
    B[:, :, 0, 0::3] = gx
    B[:, :, 1, 1::3] = gy
    B[:, :, 2, 2::3] = gz
    B[:, :, 3, 1::3], B[:, :, 3, 2::3] = gz, gy
    B[:, :, 4, 0::3], B[:, :, 4, 2::3] = gz, gx
    B[:, :, 5, 0::3], B[:, :, 5, 1::3] = gy, gx
    return B, detJ


def elasticity(E, nu):
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] += 2 * mu
    D[[3, 4, 5], [3, 4, 5]] = mu
    return D


def solve(x, y, tris, thickness, layers, E, nu, fixed_edges, loads):
    """Solve the extruded part.

    x, y, tris: the 2D triangle mesh (corner nodes only, CCW or not).
    fixed_edges: 2D boundary edges (u, v) whose side faces are clamped.
    loads: [{"edges": [(u, v), ...], "f": (fx, fy, fz)}]: total force spread evenly over the
    side faces of those edges.
    Returns displacements (n, 3), nodal von Mises (n,), and a summary.
    """
    nodes, elems, base = extrude(x, y, tris, thickness, layers)
    n, D = len(nodes), elasticity(E, nu)
    dofs = (3 * elems[:, :, None] + np.arange(3)).reshape(len(elems), -1)

    B, detJ = _B(nodes, elems, GAUSS)
    if np.any(detJ <= 0):
        raise ValueError("The 3D mesh has inverted elements.")
    Ke = np.einsum("eqji,jk,eqkl,eq->eil", B, D, B, detJ * WEIGHT, optimize=True)
    m = dofs.shape[1]
    K = sp.csr_matrix((Ke.ravel(), (np.repeat(dofs, m, axis=1).ravel(), np.tile(dofs, (1, m)).ravel())),
                      shape=(3 * n, 3 * n))

    key = lambda u, v: (min(u, v), max(u, v))
    fixed_set = {key(*e) for e in fixed_edges}
    fixed_verts = {v for e in fixed_set for v in e}
    single = base[:, 0] == base[:, 1]
    on_fixed = np.array([(b0 in fixed_verts) if s else (key(b0, b1) in fixed_set)
                         for (b0, b1), s in zip(base.tolist(), single)])
    if not on_fixed.any():
        raise ValueError("No fixed edges on the outline.")

    # boundary tet faces, for the side-face loads
    fc = elems[:, FACES].reshape(-1, 3)                      # corner nodes of each face
    edge_of = {(i, j): 4 + k for k, (i, j) in enumerate(EDGES.tolist())}
    fm = np.array([[edge_of[tuple(sorted((f[0], f[1])))], edge_of[tuple(sorted((f[1], f[2])))],
                    edge_of[tuple(sorted((f[0], f[2])))]] for f in FACES.tolist()])
    fmid = elems[:, fm].reshape(-1, 3)                       # mid nodes of each face
    keys = np.sort(fc, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    bnd = first[counts == 1]
    f = np.zeros(3 * n)
    applied = np.zeros(3)
    for L in loads:
        edges = {key(*e) for e in L["edges"]}
        sel = [i for i in bnd if len(set(base[fc[i], 0])) == 2 and key(*sorted(set(base[fc[i], 0]))) in edges]
        if not sel:
            raise ValueError("A load's edges are not on the outline.")
        a, b_, c = (nodes[fc[sel, k]] for k in range(3))
        area = 0.5 * np.linalg.norm(np.cross(b_ - a, c - a), axis=1)
        t = np.asarray(L["f"], float) / area.sum()
        for k in range(3):  # flat 6-node face, uniform traction: each mid node carries a third
            np.add.at(f, (3 * fmid[sel, k][:, None] + np.arange(3)).ravel(), np.outer(area / 3, t).ravel())
        applied += L["f"]

    fixed = np.flatnonzero(on_fixed)
    held = (3 * fixed[:, None] + np.arange(3)).ravel()
    free = np.setdiff1d(np.arange(3 * n), held)
    u = np.zeros(3 * n)
    u[free] = spsolve(K[free][:, free].tocsc(), f[free])
    if not np.all(np.isfinite(u)):
        raise ValueError("The part is not held enough. Fix more edges.")
    reaction = (K @ u - f)[held].reshape(-1, 3).sum(axis=0)

    Bn, _ = _B(nodes, elems, LOCAL)
    sig = np.einsum("eqjl,el->eqj", Bn, u[dofs]) @ D.T
    total = np.zeros((n, 6))
    np.add.at(total, elems, sig)
    s = total / np.bincount(elems.ravel(), minlength=n)[:, None]
    xx, yy, zz, yz, xz, xy = s.T
    vm = np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (yz ** 2 + xz ** 2 + xy ** 2))
    u = u.reshape(-1, 3)
    i = int(vm.argmax())
    return u, vm, nodes, base, dict(
        nodes=n, tets=len(elems), dofs=3 * n, layers=layers, umax=float(np.linalg.norm(u, axis=1).max()),
        vm_max=float(vm[i]), peak_at=nodes[i].round(2).tolist(),
        reaction=reaction.round(4).tolist(), applied=applied.round(4).tolist())


def run_json(req):
    """Entry point for the browser: a dict in, a JSON-friendly dict out."""
    u, vm, nodes, base, s = solve(req["x"], req["y"], req["tris"], req["thickness"], req["layers"],
                                  req["E"], req["nu"], req["fixed"], req["loads"])
    n2 = len(req["x"])
    corner = base[:, 0] == base[:, 1]
    top = np.zeros(n2)  # worst von Mises through the thickness, for each 2D vertex
    np.maximum.at(top, base[corner, 0], vm[corner])
    s["vm_2d"] = top.tolist()
    s["uz_2d"] = u[(req["layers"]) * n2 + np.arange(n2), 2].tolist()  # top-face sag
    return s
