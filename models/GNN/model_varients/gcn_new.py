import torch, torch.nn as nn
from registries import register

# Different GCN variants 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)
    def forward(self, A_hat, X):
        # A_hat: [Batch, N, N], X: [Batch, N, F]
        if A_hat.ndim == 3:  # Batched dense matrix
            B, N, F = X.shape
            H_list = []
            for i in range(B):
                A_sparse = A_hat[i].to_sparse()
                H_i = torch.sparse.mm(A_sparse, X[i])  # [N, F]
                H_list.append(H_i)
            H = torch.stack(H_list, dim=0)  # [B, N, F]
        else:
            H = torch.matmul(A_hat, X)
        
        H = self.lin(H)
        return self.drop(self.act(self.ln(H)))
    
class GraphConvOneTwoHop(nn.Module):
    """
    α1 A_hat + α2 A_hat^2 to approximate g_hat
    forward(A_hat, X)
        A_hat: [B, N, N] 
        X    : [B, N, F_in]
    """
    def __init__(self, in_dim, out_dim, dropout=0.0,
                 alpha1_init=1.0, alpha2_init=1.0, learnable_alpha=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)

        if learnable_alpha:
            self.alpha1 = nn.Parameter(torch.tensor(alpha1_init, dtype=torch.float32))
            self.alpha2 = nn.Parameter(torch.tensor(alpha2_init, dtype=torch.float32))
            self.alpha3 = nn.Parameter(torch.tensor(alpha2_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha1", torch.tensor(alpha1_init, dtype=torch.float32))
            self.register_buffer("alpha2", torch.tensor(alpha2_init, dtype=torch.float32))
            self.register_buffer("alpha3", torch.tensor(alpha2_init, dtype=torch.float32))
    def forward(self, A_hat, X):
        if A_hat.ndim == 3:  # batched dense -> per-batch sparse
            B, N, F = X.shape
            H1_list, H2_list, H3_list = [], [], []
            for i in range(B):
                A_sparse = A_hat[i].to_sparse()
                H1_i = torch.sparse.mm(A_sparse, X[i])   # 1-hop: A X
                H2_i = torch.sparse.mm(A_sparse, H1_i)
                H3_i = torch.sparse.mm(A_sparse, H2_i)
                H1_list.append(H1_i)
                H2_list.append(H2_i)
                H3_list.append(H3_i)
            H1 = torch.stack(H1_list, dim=0)             # [B, N, F]
            H2 = torch.stack(H2_list, dim=0)             # [B, N, F]
            H3 = torch.stack(H3_list, dim=0)             # [B, N, F]

        H_mix = self.alpha1 * H1 + self.alpha2 * H2 + self.alpha3 * H3     # (α1 A + α2 A^2 + α3 A^3) X
        H = self.lin(H_mix)
        H = self.ln(H)
        H = self.act(H)
        H = self.drop(H)
        return H

class GraphConvNHop(nn.Module):
    """
    Use α1 A_hat + α2 A_hat^2 + ... + αn A_hat^n to approximate g_hat
    forward(A_hat, X)
        A_hat: [B, N, N] 
        X    : [B, N, F_in]
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, n_hops=3,
                 alpha_init=1.0, learnable_alpha=True):
        """
        Args:
            in_dim: input feature dimension
            out_dim: output feature dimension
            dropout: dropout rate
            n_hops: number of hops (1, 2, 3, ..., n)
            alpha_init: initial value for all alphas
            learnable_alpha: whether to learn alpha parameters
        """
        super().__init__()
        self.n_hops = n_hops
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)

        # ✅ 创建 n 个 alpha 参数
        if learnable_alpha:
            self.alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
                for _ in range(n_hops)
            ])
        else:
            self.alphas = nn.ModuleList()
            for i in range(n_hops):
                self.register_buffer(f"alpha{i+1}", torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, A_hat, X):
        """
        A_hat: [B, N, N] 
        X    : [B, N, F] 
        """
        if A_hat.ndim == 3:  # batched dense -> per-batch sparse
            B, N, F = X.shape
            
            H_lists = [[] for _ in range(self.n_hops)]
            
            for i in range(B):
                A_sparse = A_hat[i].to_sparse()
                
                H_prev = X[i]  
                for hop in range(self.n_hops):
                    H_curr = torch.sparse.mm(A_sparse, H_prev)  # A^hop X
                    H_lists[hop].append(H_curr)
                    H_prev = H_curr  
            
            
            H_hops = [torch.stack(H_list, dim=0) for H_list in H_lists]
        
        else:
            
            H_hops = []
            H_prev = X
            for hop in range(self.n_hops):
                H_curr = torch.matmul(A_hat, H_prev)
                H_hops.append(H_curr)
                H_prev = H_curr

    
        if isinstance(self.alphas, nn.ParameterList):
            # learnable_alpha=True
            H_mix = sum(alpha * H for alpha, H in zip(self.alphas, H_hops))
        else:
            # learnable_alpha=False
            H_mix = sum(
                getattr(self, f"alpha{i+1}") * H 
                for i, H in enumerate(H_hops)
            )
        
        # 线性变换 + 归一化 + 激活 + dropout
        H = self.lin(H_mix)
        H = self.ln(H)
        H = self.act(H)
        H = self.drop(H)
        return H

class APPNPConv(nn.Module):
    """
    APPNP propagation layer:
        1) H0 = f(X) 
        2) K 次迭代: H^{k+1} = (1 - alpha) A_hat H^k + alpha H0
    forward(A_hat, X)
        A_hat: [B, N, N]  (normalized adjacency with self-loops)
        X    : [B, N, F_in]
    """
    def __init__(self, in_dim, out_dim, K, alpha, dropout=0.0,
                 use_ln=False):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.use_ln = use_ln
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.ln = nn.LayerNorm(out_dim) if use_ln else None
        self.drop = nn.Dropout(dropout)

    def forward(self, A_hat, X):
        """
        A_hat: [B, N, N] or [N, N]
        X    : [B, N, F_in] or [N, F_in]
        """
        if A_hat.dim() == 2:
            # [N,N] -> [1,N,N]
            A_hat = A_hat.unsqueeze(0)
        if X.dim() == 2:
            # [N,F] -> [1,N,F]
            X = X.unsqueeze(0)
        # H0 = f(X)
        H0 = self.lin(X)         
        H0 = self.act(H0)

        H = H0
        for _ in range(self.K):
            # [B,N,N] @ [B,N,F] -> [B,N,F]
            H = torch.matmul(A_hat, H)
            H = (1.0 - self.alpha) * H + self.alpha * H0

        if self.ln is not None:
            H = self.ln(H)
        H = self.drop(H)

        return H

# Different GCN-based model architectures


@register("model", "gcn_basicA_appnp")
class GCNBasicAPPNP(nn.Module):
    """
    APPNP-style GCN model
    input : batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
    output: pred_arr_reg (B,K), H_list (for checking)
    """
    def __init__(self, in_dim, hidden=64, layers=1, readout="max",
                 out_array_dim=None, dropout=0.1,
                 K=1, alpha=0.5, use_ln=False,num_nodes=1888, out_scalar=False, **kwargs):
        super().__init__()
        self.in_dim = int(in_dim)
        dims = [in_dim] + [hidden] * layers
        
        self.gcns = nn.ModuleList([
            APPNPConv(dims[i], dims[i+1],
                      K=K, alpha=alpha,
                      dropout=dropout, use_ln=use_ln)
            for i in range(layers)
        ])

        self.readout = readout
        self.num_nodes = int(num_nodes)
        self.hidden = hidden
        fuse_dim = self.num_nodes * self.in_dim
        self.head_array = nn.Linear(fuse_dim, out_array_dim) if out_array_dim else None


    def _readout(self, H):
        if self.readout == "sum":
            return H.sum(1)
        if self.readout == "max":
            return H.max(1).values
        return H.mean(1)

    def forward(self, batch, return_all_H=True):
        A_hat, X = batch["A_hat"], batch["X"]
        H0 = X
        H = X
        H_list = [H]

        for g in self.gcns:
            H = g(A_hat, H)
            H_list.append(H)

        g_raw = X.reshape(X.size(0), -1)   # 原始特征直连
        g_gnn = H.reshape(H.size(0), -1)   # 不用mean，直接flatten
        g = torch.cat([g_raw], dim=-1)

        out = {}
        out["pred_arr_reg"] = self.head_array(g)
        if return_all_H:
            out["H_list"] = H_list
        return out


@register("model", "mlp_x_only")
class MLPXOnly(nn.Module):
    """
    MLP baseline.
    input : batch["X"] (B,N,F)
    output: pred_arr_reg (B,K)
    output2 : pred_reg (B,1)
    """
    def __init__(self, in_dim, hidden=256, out_array_dim=None,
                 num_nodes=1888, dropout=0.1, **kwargs):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_nodes = int(num_nodes)
        self.out_array_dim = out_array_dim
        x_dim = self.num_nodes * self.in_dim

        self.mlp = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_array = nn.Linear(hidden, out_array_dim) if out_array_dim else None

    def forward(self, batch, return_all_H=True):
        X = batch["X"]                         # (B,N,F)
        g = X.reshape(X.size(0), -1)           # (B,N*F)
        h = self.mlp(g)
        out = {}
        if self.out_array_dim  == 1:
            out["pred_reg"] = self.head_array(h).squeeze(-1)  # (B,)
        else:
            out["pred_arr_reg"] = self.head_array(h)
        return out
    
    
@register("model", "gcn_only_appnp")
class GCNOnlyAPPNP(nn.Module):
    """
    APPNP-style GCN only model.
    input : batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
    output: pred_arr_reg (B,K)
    This model does NOT directly concatenate raw X.
    """
    def __init__(self, in_dim, hidden=64, layers=1, readout="mean",
                 out_array_dim=None, dropout=0.1,
                 K=1, alpha=0.5, use_ln=False,
                 num_nodes=1888,att_hidden=None, **kwargs):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_nodes = int(num_nodes)
        self.readout = readout
        self.att_hidden = int(att_hidden) if att_hidden is not None else int(hidden)
        dims = [self.in_dim] + [self.hidden] * layers

        self.gcns = nn.ModuleList([
            APPNPConv(
                dims[i], dims[i + 1],
                K=K,
                alpha=alpha,
                dropout=dropout,
                use_ln=use_ln
            )
            for i in range(layers)
        ])
        # attention scoring function
        self.att = nn.Sequential(
            nn.Linear(self.hidden, self.att_hidden),
            nn.Tanh(),
            nn.Linear(self.att_hidden, 1)
        )
        fuse_dim = self.num_nodes * self.hidden
        self.head_array = nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_array_dim)
        ) if out_array_dim else None

    def _readout(self, H):
        score = self.att(H)                  # [B,N,1]
        weight = torch.softmax(score, dim=1) # [B,N,1]
        g = (weight * H).sum(dim=1)          # [B,hidden]
        return g

    def forward(self, batch, return_all_H=True):
        A_hat, X = batch["A_hat"], batch["X"]

        H = X
        H_list = [H]

        for gcn in self.gcns:
            H = gcn(A_hat, H)
            H_list.append(H)

        #g_gnn = self._readout(H)
        g_gnn = H.reshape(H.size(0), -1)   # 不用mean，直接flatten
        g = torch.cat([g_gnn], dim=-1)
        
        out = {}
        if self.out_array_dim  == 1:
            out["pred_reg"] = self.head_array(g).squeeze(-1)  # (B,)
        else:
            out["pred_arr_reg"] = self.head_array(g)

        if return_all_H:
            out["H_list"] = H_list

        return out


@register("model", "gcn_only_flatten_appnp")
class GCNOnlyAPPNP(nn.Module):
    """
    APPNP-style GCN only model.
    input : batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
    output: pred_arr_reg (B,K)

    This model does NOT directly concatenate raw X.
    """
    def __init__(self, in_dim, hidden=64, layers=1, readout="mean",
                 out_array_dim=None, dropout=0.1,
                 K=1, alpha=0.5, use_ln=False,
                 num_nodes=1888,att_hidden=None, **kwargs):
        super().__init__()
        self.out_array_dim = out_array_dim
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_nodes = int(num_nodes)
        self.readout = readout
        self.att_hidden = int(att_hidden) if att_hidden is not None else int(hidden)
        dims = [self.in_dim] + [self.hidden] * layers

        self.gcns = nn.ModuleList([
            APPNPConv(
                dims[i], dims[i + 1],
                K=K,
                alpha=alpha,
                dropout=dropout,
                use_ln=use_ln
            )
            for i in range(layers)
        ])
        # attention scoring function
        self.att = nn.Sequential(
            nn.Linear(self.hidden, self.att_hidden),
            nn.Tanh(),
            nn.Linear(self.att_hidden, 1)
        )
        self.head_array = nn.Sequential(
            nn.Linear(self.num_nodes, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_array_dim)
        ) if out_array_dim else None

    def _readout(self, H):
        score = self.att(H)                  # [B,N,1]
        weight = torch.softmax(score, dim=1) # [B,N,1]
        g = torch.flatten(weight, start_dim=1)          # [B,hidden]
        return g

    def forward(self, batch, return_all_H=True):
        A_hat, X = batch["A_hat"], batch["X"]

        H = X
        H_list = [H]

        for gcn in self.gcns:
            H = gcn(A_hat, H)
            H_list.append(H)

        g_gnn = self._readout(H)

        out = {}
        if self.out_array_dim  == 1:
            out["pred_reg"] = self.head_array(g_gnn).squeeze(-1)  # (B,)
        else:
            out["pred_arr_reg"] = self.head_array(g_gnn)

        if return_all_H:
            out["H_list"] = H_list

        return out
    
@register("model", "gcn_x_appnp")
class GCNPlusXAPPNP(nn.Module):
    """
    APPNP-style GCN + raw X skip model.

    input :
        batch["A_hat"] : (B, N, N)
        batch["X"]     : (B, N, F)

    output:
        pred_arr_reg : (B, K)

    This model uses:
        graph branch   : APPNP-GCN(X, A_hat)
        feature branch : raw X flattened
        fusion         : [g_gnn, g_raw]
    """

    def __init__(
        self,
        in_dim,
        hidden=64,
        x_hidden=256,
        layers=1,
        readout="mean",
        out_array_dim=None,
        dropout=0.1,
        K=1,
        alpha=0.5,
        use_ln=False,
        num_nodes=1888,
        att_hidden=None,
        **kwargs
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.x_hidden = int(x_hidden)
        self.num_nodes = int(num_nodes)
        self.readout = readout

        self.att_hidden = (
            int(att_hidden)
            if att_hidden is not None
            else int(hidden)
        )

        dims = [self.in_dim] + [self.hidden] * layers

        self.gcns = nn.ModuleList([
            APPNPConv(
                dims[i],
                dims[i + 1],
                K=K,
                alpha=alpha,
                dropout=dropout,
                use_ln=use_ln
            )
            for i in range(layers)
        ])

        x_dim = self.num_nodes * self.in_dim


        self.att = nn.Sequential(
            nn.Linear(self.hidden, self.att_hidden),
            nn.Tanh(),
            nn.Linear(self.att_hidden, 1)
        )

        fuse_dim = self.num_nodes + x_dim

        self.head_array = (
            nn.Sequential(
                nn.Linear(fuse_dim, out_array_dim),
            )
            if out_array_dim
            else None
        )

    def _readout(self, H):
        """
        H : [B, N, hidden]
        """

        score = self.att(H)                  # [B, N, 1]
        weight = torch.softmax(score, dim=1) # [B, N, 1]

        g = torch.flatten(weight, start_dim=1)  # [B, N]

        return g

    def forward(self, batch, return_all_H=True):

        A_hat = batch["A_hat"]
        X = batch["X"]

        H = X
        H_list = [H]

        for gcn in self.gcns:
            H = gcn(A_hat, H)
            H_list.append(H)

        g_gnn = self._readout(H)

        g_raw = X.reshape(X.size(0), -1)

        g = torch.cat([g_gnn, g_raw], dim=-1)

        out = {}

        if self.out_array_dim  == 1:
            out["pred_reg"] = self.head_array(g).squeeze(-1)  # (B,)
        else:
            out["pred_arr_reg"] = self.head_array(g)

        if return_all_H:
            out["H_list"] = H_list

        return out
    
@register("model", "gcn_x_wx_appnp")
class GCNPlusXAPPNP(nn.Module):
    """
    APPNP-style GCN + raw X skip model.
    input : batch["A_hat"] (B,N,N), batch["X"] (B,N,F)
    output: pred_arr_reg (B,K)
    This model uses:
        graph branch: APPNP-GCN(X, A_hat)
        feature branch: raw X flattened and compressed
        fusion: [g_gnn, h_x]
    """
    def __init__(self, in_dim, hidden=64, x_hidden=256, layers=1,
                 readout="mean", out_array_dim=None, dropout=0.1,
                 K=1, alpha=0.5, use_ln=False,
                 num_nodes=1888, att_hidden=None, **kwargs):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.x_hidden = int(x_hidden)
        self.num_nodes = int(num_nodes)
        self.readout = readout
        self.att_hidden = int(att_hidden) if att_hidden is not None else int(hidden)
        dims = [self.in_dim] + [self.hidden] * layers
        self.gcns = nn.ModuleList([
            APPNPConv(
                dims[i], dims[i + 1],
                K=K,
                alpha=alpha,
                dropout=dropout,
                use_ln=use_ln
            )
            for i in range(layers)
        ])

        x_dim = self.num_nodes * self.in_dim

        
        self.att = nn.Sequential(
            nn.Linear(self.hidden, self.att_hidden),
            nn.Tanh(),
            nn.Linear(self.att_hidden, 1)
        )
        

        fuse_dim = self.num_nodes * self.in_dim
        self.head_array = nn.Sequential(
            nn.Linear(fuse_dim,  out_array_dim),
        ) if out_array_dim else None

    def _readout(self, H):
        score = self.att(H)                  # [B,N,1]
        weight = torch.softmax(score, dim=1) # [B,N,1]# [B,hidden]
        return weight

    def forward(self, batch, return_all_H=True):
        A_hat, X = batch["A_hat"], batch["X"]

        H = X
        H_list = [H]

        for gcn in self.gcns:
            H = gcn(A_hat, H)
            H_list.append(H)
        
        weight = self._readout(H)                # [B, N, 1]
        N = X.size(1)
        X_att = X * (1.0 + weight * N)           # [B, N, F]
        g = X_att.reshape(X.size(0), -1)         # [B, N*F]
        out = {}
        if self.out_array_dim == 1:
            out["pred_reg"] = self.head_array(g).squeeze(-1)  # (B,)
        else:
            out["pred_arr_reg"] = self.head_array(g)

        if return_all_H:
            out["H_list"] = H_list

        return out
    
    
    
    
## clique pooling

@register("model", "gcn_only_clique_pooling_appnp")
class GCNCliqueTreeAPPNP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden=64,
        layers=1,
        out_array_dim=None,
        dropout=0.1,
        K=1,
        alpha=0.5,
        use_ln=False,
        att_hidden=None,
        **kwargs
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.out_array_dim = out_array_dim
        self.att_hidden = int(att_hidden) if att_hidden is not None else int(hidden)

        dims = [self.in_dim] + [self.hidden] * layers

        self.node_gcns = nn.ModuleList([
            APPNPConv(
                dims[i],
                dims[i + 1],
                K=K,
                alpha=alpha,
                dropout=dropout,
                use_ln=use_ln,
            )
            for i in range(layers)
        ])

        # clique tree 上再跑一层 APPNP
        self.clique_gcn = APPNPConv(
            self.hidden,
            self.hidden,
            K=K,
            alpha=alpha,
            dropout=dropout,
            use_ln=use_ln,
        )

        self.att = nn.Sequential(
            nn.Linear(self.hidden, self.att_hidden),
            nn.Tanh(),
            nn.Linear(self.att_hidden, 1),
        )

        self.head_array = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, out_array_dim),
        )

    def normalize_adj(self, A):
        """
        A: [B,C,C]
        """
        B, C, _ = A.shape
        I = torch.eye(C, device=A.device).unsqueeze(0).expand(B, C, C)
        A = A + I

        deg = A.sum(dim=-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)

        A_hat = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        return A_hat

    def clique_pool(self, H, clique_M):
        """
        H:        [B,N,H]
        clique_M:[B,C,N]

        return:
            Hc: [B,C,H]
        """
        size = clique_M.sum(dim=-1, keepdim=True).clamp(min=1.0)
        M_norm = clique_M / size
        Hc = torch.bmm(M_norm, H)
        return Hc

    def readout_cliques(self, Hc, clique_mask):
        """
        Hc: [B,C,H]
        clique_mask: [B,C]
        """
        score = self.att(Hc).squeeze(-1)  # [B,C]
        score = score.masked_fill(~clique_mask, -1e9)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        g = (weight * Hc).sum(dim=1)  # [B,H]
        return g

    def forward(self, batch, return_all_H=True):
        A_hat = batch["A_hat"]
        X = batch["X"]

        clique_M = batch["clique_M"]
        A_clique = batch["A_clique"]
        sep_size = batch["sep_size"]
        clique_mask = batch["clique_mask"]

        H = X
        H_list = [H]

        # 1. 原图上 message passing
        for gcn in self.node_gcns:
            H = gcn(A_hat, H)
            H_list.append(H)

        # 2. maximal clique -> supernode
        Hc = self.clique_pool(H, clique_M)

        # 3. separator -> edge weight
        # 最简单：A_clique * sep_size
        A_c_weighted = A_clique * sep_size

        # 防止全 0
        A_c_hat = self.normalize_adj(A_c_weighted)

        # 4. clique tree 上 message passing
        Hc = self.clique_gcn(A_c_hat, Hc)

        # 5. clique-level readout
        g = self.readout_cliques(Hc, clique_mask)

        pred = self.head_array(g)

        out = {}

        if self.out_array_dim == 1:
            out["pred_reg"] = pred.squeeze(-1)
        else:
            out["pred_arr_reg"] = pred

        if return_all_H:
            out["H_list"] = H_list
            out["H_clique"] = Hc

        return out