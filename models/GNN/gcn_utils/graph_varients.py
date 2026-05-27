import numpy as np
import re
import math
import cmath
import numpy as np

import re
import math
import cmath
import numpy as np


def matpower_case_to_ei_ew(
    case_path: str,
    directed: bool = True,
    weight_mode: str = "abs_Yij",   
    add_self_loops: bool = False,
    self_loop_weight: float = 1.0,
    eps: float = 1e-12,
):
    """
    Read MATPOWER caseXX.m and return:
      ei_u: np.ndarray shape (2, M), dtype int32, 0-based indices
      ew_u: np.ndarray shape (M,), dtype float32

    directed=True  -> include both i->j and j->i (and keep asymmetry if tap/shift exists)
    directed=False -> output undirected edges but still in (2,M); we add both directions with same weight
                      (common for GNNs expecting directed edge_index anyway)

    weight_mode:
      - "abs_Yij": build Ybus off-diagonal and use w_ij = |Y_ij|
      - "abs_yij": use w_ij = |y_ij / t*| (consistent with Y off-diagonal magnitude without building full Y)
    """

    # -------- 1) read file text --------
    text = open(case_path, "r", encoding="utf-8", errors="ignore").read()

    # baseMVA
    m = re.search(r"mpc\.baseMVA\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*;", text)
    if not m:
        raise ValueError("Cannot find mpc.baseMVA in the case file.")
    baseMVA = float(m.group(1))

    # bus matrix
    mb = re.search(r"mpc\.bus\s*=\s*\[(.*?)\]\s*;", text, flags=re.S)
    if not mb:
        raise ValueError("Cannot find mpc.bus matrix.")
    bus_lines = mb.group(1).splitlines()
    bus = []
    for line in bus_lines:
        line = line.split('%', 1)[0].strip()
        if not line:
            continue
        line = line.rstrip(';').strip()
        if not line:
            continue
        bus.append([float(x) for x in line.split()])
    bus = np.array(bus, dtype=float)

    # branch matrix
    mbr = re.search(r"mpc\.branch\s*=\s*\[(.*?)\]\s*;", text, flags=re.S)
    if not mbr:
        raise ValueError("Cannot find mpc.branch matrix.")
    br_lines = mbr.group(1).splitlines()
    branch = []
    for line in br_lines:
        line = line.split('%', 1)[0].strip()
        if not line:
            continue
        line = line.rstrip(';').strip()
        if not line:
            continue
        branch.append([float(x) for x in line.split()])
    branch = np.array(branch, dtype=float)

    n = bus.shape[0]

    # MATPOWER indices
    # branch: fbus tbus r x b ... ratio angle status ...
    FBUS, TBUS = 0, 1
    R, X, B = 2, 3, 4
    RATIO, ANGLE, STATUS = 8, 9, 10

    # bus: ... Gs Bs ...
    GS, BS = 4, 5

    # -------- 2) optionally build Ybus (only needed for abs_Yij) --------
    if weight_mode == "abs_Yij":
        Y = np.zeros((n, n), dtype=np.complex128)

        # bus shunt into diagonal (does not affect off-diagonal weights, but keeps Y correct)
        ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
        Y += np.diag(ysh)

        for row in branch:
            if int(row[STATUS]) == 0:
                continue

            i = int(row[FBUS]) - 1
            j = int(row[TBUS]) - 1
            r, x, b = float(row[R]), float(row[X]), float(row[B])

            z = complex(r, x)
            if abs(z) < eps:
                continue
            y = 1.0 / z

            ratio = float(row[RATIO])
            angle_deg = float(row[ANGLE])
            if ratio == 0.0:
                t = 1.0 + 0j
            else:
                t = ratio * cmath.exp(1j * math.radians(angle_deg))

            y_c = 1j * (b / 2.0)

            # assemble
            Yff = (y + y_c) / (abs(t) ** 2)
            Yft = -y / np.conj(t)
            Ytf = -y / t
            Ytt = (y + y_c)

            Y[i, i] += Yff
            Y[i, j] += Yft
            Y[j, i] += Ytf
            Y[j, j] += Ytt

    # -------- 3) collect edges into lists --------
    src = []
    dst = []
    wts = []

    def _add_edge(u, v, w):
        if w > eps:
            src.append(u)
            dst.append(v)
            wts.append(w)

    if weight_mode == "abs_Yij":
        # use off-diagonal of Y
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = abs(Y[i, j])
                if w > eps:
                    if directed:
                        _add_edge(i, j, float(w))
                    else:
                        # undirected: we'll still output both directions (common for message passing)
                        # only add once when i<j, then add both
                        pass
        if not directed:
            # add undirected edges as both directions with same weight
            for i in range(n):
                for j in range(i + 1, n):
                    w = abs(Y[i, j])
                    if w > eps:
                        _add_edge(i, j, float(w))
                        _add_edge(j, i, float(w))

    else:
        raise ValueError("weight_mode must be 'abs_Yij'.")

    # -------- 4) self-loops (optional) --------
    if add_self_loops:
        for i in range(n):
            _add_edge(i, i, float(self_loop_weight))

    # -------- 5) pack to arrays --------
    ei_u = np.array([src, dst], dtype=np.int32)      # (2, M)
    ew_u = np.array(wts, dtype=np.float32)           # (M,)

    return ei_u, ew_u

    

def _ensure_edge_index_2xE(edge_index: np.ndarray) -> np.ndarray:
    """Ensure edge_index is shape [2, E]."""
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {edge_index.shape}")
    if edge_index.shape[0] == 2:
        return edge_index
    if edge_index.shape[1] == 2:
        return edge_index.T
    raise ValueError(f"edge_index must be [2,E] or [E,2], got {edge_index.shape}")

def _coalesce_undirected(edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int):
    """
    Make graph undirected + remove self-loops + deduplicate.
    Returns edge_index [2,E'], edge_weight [E'].
    """
    ei = _ensure_edge_index_2xE(edge_index).astype(np.int64, copy=False)
    ew = np.asarray(edge_weight).astype(np.float32, copy=False)
    if ew.ndim != 1 or ew.shape[0] != ei.shape[1]:
        raise ValueError("edge_weight must be shape [E] matching edge_index")

    src, dst = ei[0], ei[1]

    # Remove self loops
    mask = src != dst
    src, dst, ew = src[mask], dst[mask], ew[mask]

    # Symmetrize: add reverse edges
    src2 = np.concatenate([src, dst])
    dst2 = np.concatenate([dst, src])
    ew2  = np.concatenate([ew,  ew])

    # Deduplicate by keeping max weight (or just 1.0)
    # Use a hash key: key = src * N + dst
    key = src2 * num_nodes + dst2
    order = np.argsort(key)
    key, src2, dst2, ew2 = key[order], src2[order], dst2[order], ew2[order]

    # Keep one per key: take max weight
    unique_keys, first_idx = np.unique(key, return_index=True)
    # max per group
    # compute max by scanning groups
    max_w = np.zeros_like(unique_keys, dtype=np.float32)
    # indices of group boundaries
    boundaries = np.append(first_idx, len(key))
    for i in range(len(unique_keys)):
        a, b = boundaries[i], boundaries[i+1]
        max_w[i] = np.max(ew2[a:b])

    src_u = (unique_keys // num_nodes).astype(np.int64)
    dst_u = (unique_keys %  num_nodes).astype(np.int64)

    ei_u = np.vstack([src_u, dst_u]).astype(np.int32)
    ew_u = max_w.astype(np.float32)
    return ei_u, ew_u

def make_graph_varient(A_grid, num_nodes: int, mode: str, seed: int = 0, undirected: bool = True):
    """
    Input / output: A_grid = (edge_index, edge_weight) with edge_index [2,E].
    mode: "real" | "identity" | "random" | "random_deg" (optional)
    Returns: (edge_index_var, edge_weight_var) in the same format.
    """
    edge_index, edge_weight = A_grid
    ei = _ensure_edge_index_2xE(edge_index)
    ew = np.asarray(edge_weight)

    # 1) REAL: just clean up / enforce undirected (recommended)
    if mode == "real":
        if undirected:
            return _coalesce_undirected(ei, ew, num_nodes)
        # remove self-loops only
        src, dst = ei[0], ei[1]
        mask = src != dst
        return np.vstack([src[mask], dst[mask]]).astype(np.int32), ew[mask].astype(np.float32)

    # 2) IDENTITY: only self-loops (i,i)
    if mode == "identity":
        nodes = np.arange(num_nodes, dtype=np.int32)
        ei_id = np.vstack([nodes, nodes])          # [2, N]
        ew_id = np.ones(num_nodes, dtype=np.float32)
        return ei_id, ew_id

    # 3) RANDOM: keep edge count (rough), random pairs
    if mode == "random":
        rng = np.random.default_rng(seed)

        # use undirected edge count if we coalesce; otherwise use raw E
        if undirected:
            ei_real, ew_real = _coalesce_undirected(ei, ew, num_nodes)
            # count undirected edges by upper-tri (src<dst)
            src, dst = ei_real[0], ei_real[1]
            und_mask = src < dst
            E_undir = int(np.sum(und_mask))
            # sample that many undirected pairs
            u = rng.integers(0, num_nodes, size=E_undir, dtype=np.int64)
            v = rng.integers(0, num_nodes, size=E_undir, dtype=np.int64)
            mask = u != v
            u, v = u[mask], v[mask]
            # build symmetric directed edges
            src2 = np.concatenate([u, v]).astype(np.int64)
            dst2 = np.concatenate([v, u]).astype(np.int64)
            ew2  = np.ones_like(src2, dtype=np.float32)
            ei2  = np.vstack([src2, dst2]).astype(np.int32)
            # coalesce/dedup to be safe
            return _coalesce_undirected(ei2, ew2, num_nodes)

        # directed random graph
        E = ei.shape[1]
        u = rng.integers(0, num_nodes, size=E, dtype=np.int64)
        v = rng.integers(0, num_nodes, size=E, dtype=np.int64)
        mask = u != v
        u, v = u[mask], v[mask]
        ei2 = np.vstack([u, v]).astype(np.int32)
        ew2 = np.ones(ei2.shape[1], dtype=np.float32)
        return ei2, ew2

    # 4) RANDOM_DEG (optional): preserve degree sequence approximately
    #    This is a bit more involved; simplest configuration-model style using stubs.
    if mode == "random_deg":
        if not undirected:
            raise ValueError("random_deg implemented for undirected only")
        rng = np.random.default_rng(seed)
        ei_real, _ = _coalesce_undirected(ei, ew, num_nodes)
        src, dst = ei_real[0], ei_real[1]
        # degree from directed undirected representation counts twice; still OK for stubs
        deg = np.bincount(src, minlength=num_nodes).astype(np.int64)

        stubs = np.repeat(np.arange(num_nodes, dtype=np.int64), deg)
        rng.shuffle(stubs)
        # pair stubs
        if len(stubs) % 2 == 1:
            stubs = stubs[:-1]
        u = stubs[0::2]
        v = stubs[1::2]
        mask = u != v
        u, v = u[mask], v[mask]
        src2 = np.concatenate([u, v])
        dst2 = np.concatenate([v, u])
        ew2  = np.ones_like(src2, dtype=np.float32)
        ei2  = np.vstack([src2, dst2]).astype(np.int32)
        return _coalesce_undirected(ei2, ew2, num_nodes)
    if mode == "nobalmatrix":
        ei_u, ew_u = matpower_case_to_ei_ew("/home/goatoine/Documents/Lanyue/data/raw_data/case2746wop.m",
                                   directed=True,
                                   weight_mode="abs_Yij",
                                   add_self_loops=False)
        return ei_u, ew_u
    raise ValueError(f"Unknown graph mode: {mode}")
