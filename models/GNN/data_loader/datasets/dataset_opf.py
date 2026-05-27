# dataset_opf.py
import torch
from torch.utils.data import Dataset
import numpy as np
from gcn_utils import normalize

def edge_to_dense_A(edge_index, edge_weight, N, dtype=torch.float32, device="cpu"):
    """
    将 numpy (edge_index, edge_weight) 转换为稠密 A:(N,N)
    """
    edge_index = torch.from_numpy(edge_index).long().to(device)
    edge_weight = torch.from_numpy(edge_weight).to(dtype=dtype, device=device)
    A_sp = torch.sparse_coo_tensor(edge_index, edge_weight, size=(N, N), dtype=dtype, device=device)
    return A_sp.to_dense()

def normalize_A_hat(A):
    """
    对称归一化 A_hat = D^{-1/2} (A + I) D^{-1/2}
    A: (N, N) torch.Tensor
    """
    N = A.size(0)
    I = torch.eye(N, dtype=A.dtype, device=A.device)
    A_tilde = A + I
    deg = A_tilde.sum(dim=1).clamp_min(1e-8)
    D_inv_sqrt = deg.pow(-0.5)
    return D_inv_sqrt.view(-1,1) * A_tilde * D_inv_sqrt.view(1,-1)

class OPFGraphDataset(Dataset):
    """
Optimized tool selection- The **Reader** decides the `samples` (how many there are / which keys they include).
- The **Dataset** does only one “special” thing: convert the edge list in `raw["A"]` into a dense adjacency matrix `A`, then normalize it into `A_hat`.
- All other keys (`X`, various `y_*`, `extra`, etc.) are passed through as-is, without checking or interpreting their names.
- Optional: generate `X` as needed using `build_graph` / `build_features` (keeping your original logic).
    """
    def __init__(self, samples, build_features=None,  device="cpu"):
        self.samples = samples
        self.build_features = build_features     
        self.device = torch.device(device)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        out = {}
        # 1) deal with the feature part
        if self.build_features is not None:
            X = self.build_features(raw)
            X = X.to(self.device).float()
            out["X"] = X
            N = X.shape[0]

        # 2) deal with the graph part
        if "A" in raw:
            edge_index, edge_weight = raw["A"]
        elif "A_grid" in raw:
            edge_index, edge_weight = raw["A_grid"]
        else:
            raise KeyError("样本缺少键 'A' 或 'A_grid'（边列表）。")
     # 允许 numpy 或 torch
        A = edge_to_dense_A(edge_index, edge_weight, N, dtype=torch.float32, device=self.device)
        A_hat = normalize_A_hat(A)
        out["A_hat"] = A_hat

        # 3) deal with the other part, y class
        for k, v in raw.items():
            if k == "A" or k == "A_grid":
                continue  
            if k in out:
                continue  
            # Keep tensor fields for training/collate.
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
                continue

            # Clique-pooling fields are python lists (not tensors) produced by the reader.
            # They must be kept so collate_fn can build clique_M/A_clique/sep_size.
            if k in {"clique_nodes", "A_clique_edges", "sep_edges"}:
                out[k] = v
        return out
    


def make_collate_fn(dataset=None):
    """Factory for DataLoader collate_fn.

    Two modes are supported:
    - default/stack: stack all tensor fields (expects every field in a sample is a tensor)
    - clique_pooling: build clique_M/A_clique/sep_size/clique_mask from python-list fields

    If dataset is provided, mode is auto-detected from dataset.samples[0] keys.
    """

    def _collate_stack_tensors(batch):
        # 所有值都是 tensor，直接 stack
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if not torch.is_tensor(vals[0]):
                raise TypeError(
                    f"collate(stack) expects tensors only, but key '{k}' has type {type(vals[0])}. "
                    "If you're using clique pooling reader, you need clique_pooling collate."
                )
            out[k] = torch.stack(vals, 0)
        return out

    def _collate_clique_pooling(batch):
        out = {}
        B = len(batch)

        # infer N & device
        if "X" in batch[0]:
            N = int(batch[0]["X"].shape[0])
            device = batch[0]["X"].device
        else:
            N = int(batch[0]["node_load"].shape[0])
            device = batch[0]["node_load"].device

        Cmax = max(len(b.get("clique_nodes", [])) for b in batch)
        if Cmax == 0:
            raise KeyError(
                "clique_pooling collate requires 'clique_nodes' in samples, but it's missing/empty. "
                "Check your reader (should be opf_reg_pointwise_clique_pooling) and dataset __getitem__."
            )

        clique_M = torch.zeros(B, Cmax, N, dtype=torch.float32, device=device)
        A_clique = torch.zeros(B, Cmax, Cmax, dtype=torch.float32, device=device)
        sep_size = torch.zeros(B, Cmax, Cmax, dtype=torch.float32, device=device)
        clique_mask = torch.zeros(B, Cmax, dtype=torch.bool, device=device)

        for bi, b in enumerate(batch):
            clique_nodes = b["clique_nodes"]
            C = len(clique_nodes)
            clique_mask[bi, :C] = True

            for ci, nodes in enumerate(clique_nodes):
                clique_M[bi, ci, nodes] = 1.0

            for i, j in b["A_clique_edges"]:
                A_clique[bi, i, j] = 1.0
                A_clique[bi, j, i] = 1.0

            for i, j, w in b["sep_edges"]:
                sep_size[bi, i, j] = float(w)
                sep_size[bi, j, i] = float(w)

        out["clique_M"] = clique_M
        out["A_clique"] = A_clique
        out["sep_size"] = sep_size
        out["clique_mask"] = clique_mask

        # Stack all tensor fields (X, A_hat, labels, ids, ...), skip the clique list fields.
        skip_keys = {"clique_nodes", "A_clique_edges", "sep_edges"}
        for k in batch[0].keys():
            if k in skip_keys:
                continue
            v0 = batch[0][k]
            if torch.is_tensor(v0):
                out[k] = torch.stack([b[k] for b in batch], dim=0)

        return out


    wants_clique = False
    if dataset is not None and hasattr(dataset, "samples") and len(getattr(dataset, "samples", [])) > 0:
        s0 = dataset.samples[0]
        if isinstance(s0, dict) and (
            "clique_nodes" in s0 or "A_clique_edges" in s0 or "sep_edges" in s0
        ):
            wants_clique = True

    return _collate_clique_pooling if wants_clique else _collate_stack_tensors

    
