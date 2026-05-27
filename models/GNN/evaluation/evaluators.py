import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from registries import register
from .eval_utils import (
    _save_json, _plot_save, _save_metrics, _save_preds_targets,
    _logsumexp, _conf_mat, _cls_report
)
from typing import List, Dict, Optional, Tuple
scenario_id_list_15 = [
            "0.0_Chordal_MD",
            "2.0_Chordal_MD", 
            "3.0_Chordal_MD",
            "4.0_Chordal_MD",
            "5.0_Chordal_MD",
            "0.0_Chordal_AMD",
            "2.0_Chordal_AMD",
            "3.0_Chordal_AMD",
            "4.0_Chordal_AMD",
            "5.0_Chordal_AMD",
            "0.0_Chordal_MFI",
            "2.0_Chordal_MFI",
            "3.0_Chordal_MFI",
            "4.0_Chordal_MFI",
            "5.0_Chordal_MFI",
        ]
scenario_id_list_6 = [
    "3.0_Chordal_MD",
    "4.0_Chordal_MD",
    "3.0_Chordal_AMD",
    "4.0_Chordal_AMD",
    "3.0_Chordal_MFI",
    "4.0_Chordal_MFI",
]

class BaseEvaluator(object):
    def __init__(self, output_dir="./outputs", device=None):
        self.output_dir = output_dir
        self.device = device
        os.makedirs(self.output_dir, exist_ok=True)
        self._reset()

    def _reset(self): ...
    def update(self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        self._accumulate(batch, model_out)
    def _accumulate(self, batch, model_out): ...
    def compute(self):
        return self._finalize(self.output_dir)
    @torch.no_grad()
    def run(self, model, loader, device=None, output_dir=None):
        """标准测试入口：清空缓存 → 遍历 loader 前向 → 累积 → 汇总"""
        self._reset()
        output_dir = output_dir or self.output_dir
        dev = device or self.device
        model.eval()
        for batch in loader:
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(dev)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            v[kk] = vv.to(dev)
            out = model(batch)
            self._accumulate(batch, out)
        return self._finalize(output_dir)


@register("evaluator", "scalar_reg")
class ScalarRegEvaluator(BaseEvaluator):
    """
    Scalar regression evaluator for pointwise setting:
      - model_out["pred_y_reg"] : [B]
      - batch["y_reg"]          : [B]
    """
    def __init__(self, output_dir, device=None, *, save_plots: bool = True, num_classes: int = None, **kwargs):
        super().__init__(output_dir, device)
        self.save_plots = bool(save_plots)

    def _reset(self):
        self._pred: List[torch.Tensor] = []
        self._true: List[torch.Tensor] = []

    def _accumulate(self, batch, model_out):
        self._pred.append(model_out["pred_y_reg"].detach().cpu().view(-1))
        self._true.append(batch["y_reg"].detach().cpu().view(-1))

    def _finalize(self, out_dir: str):
        _ensure_dir(out_dir)
        p = torch.cat(self._pred, dim=0).numpy()
        t = torch.cat(self._true, dim=0).numpy()
        diff = p - t
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "N": int(t.shape[0]),
        }

        if self.save_plots:
            fig = plt.figure(figsize=(6.2, 5.2))
            plt.scatter(t, p, s=8, alpha=0.25)
            mn = min(float(t.min()), float(p.min()))
            mx = max(float(t.max()), float(p.max()))
            plt.plot([mn, mx], [mn, mx], linewidth=1)
            plt.xlabel("True y_reg")
            plt.ylabel("Predicted y_reg")
            plt.title("Scalar regression: predicted vs true")
            _plot_save(fig, os.path.join(out_dir, "scatter_scalar.png"))

        _save_metrics(metrics, out_dir)
        _save_preds_targets(p, t, out_dir)
        return metrics


@register("evaluator", "arr_cls")
class ArrClsEvaluator(BaseEvaluator):
    """
    数组回归 + 分类：
      * 不使用 mask
      * 分类由回归数组派生：
          logits = -(y_arr_reg_pred - center) / tau
      * 计算 MAE / RMSE / 分类准确率 / 可选交叉熵
    """
    def __init__(self, output_dir, device=None, *,
                 tau: float = 1.0, center: bool = True,
                 report_ce: bool = True, save_plots: bool = True, num_classes: int = None):
        super().__init__(output_dir, device)
        self.tau = float(tau)
        self.center = bool(center)
        self.report_ce = bool(report_ce)
        self.save_plots = bool(save_plots)
        self.num_classes = num_classes
        if self.num_classes ==6:
            self.labels = scenario_id_list_6
        elif self.num_classes ==15:
            self.labels = scenario_id_list_15
        
    def _reset(self):
        self._arr_p: List[torch.Tensor] = []
        self._arr_t: List[torch.Tensor] = []
        self._ytrue: List[torch.Tensor] = []

    def _accumulate(self, batch, model_out):
        P = model_out["pred_arr_reg"].detach().cpu()
        T = batch["y_arr_reg"].detach().cpu()
        self._arr_p.append(P)
        self._arr_t.append(T)
        if "y_cls" in batch:
            self._ytrue.append(batch["y_cls"].detach().cpu())

    def _finalize(self, out_dir: str):
        P = torch.cat(self._arr_p, dim=0).numpy()  # [N,K]
        T = torch.cat(self._arr_t, dim=0).numpy()
        N, K = P.shape

        # --- 数组回归 ---
        diff = P - T
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        metrics = {"array": {"MAE": mae, "RMSE": rmse}, "classify": {}}

        # --- 分类 logits ---
        if self.center:
            P_center = P - P.mean(axis=1, keepdims=True)
            Z = -P_center / self.tau
        else:
            Z = -P / self.tau
        Z = Z - Z.max(axis=1, keepdims=True)
        yhat = Z.argmax(axis=1)
        Ytrue = torch.cat(self._ytrue, dim=0).numpy() if self._ytrue else T.argmin(axis=1)
        acc = float((yhat == Ytrue).mean())
        metrics["classify"]["acc"] = acc

        # --- 可选 CE ---
        if self.report_ce and len(Ytrue) == N:
            ce = float(np.mean(-Z[np.arange(N), Ytrue] + _logsumexp(Z, axis=1)))
            metrics["classify"]["cross_entropy"] = ce

        # --- 绘图 ---
        if self.save_plots:
            # 误差直方图
            fig = plt.figure()
            plt.hist(diff.reshape(-1), bins=40)
            plt.xlabel("Element-wise Error (pred - true)")
            plt.ylabel("Count")
            plt.title("Array Regression Error")
            _plot_save(fig, os.path.join(out_dir, "arr_error_hist.png"))

            # every dimension MAE
            per_dim_mae = np.mean(np.abs(diff), axis=0)
            fig = plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(per_dim_mae)), per_dim_mae)
            plt.xlabel("strategy")
            plt.ylabel("MAE")
            plt.title("Per-dimension MAE")
            plt.xticks(range(len(self.labels)), self.labels, rotation=45, ha='right')  
            plt.grid(True, alpha=0.3)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "arr_per_dim_mae.png"))
            
            
            #every dimension RMSE
            per_dim_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
            fig = plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(per_dim_rmse)), per_dim_rmse)
            plt.xlabel("Dimension")
            plt.ylabel("RMSE")
            plt.title("Per-dimension RMSE")
            plt.xticks(range(len(self.labels)), self.labels, rotation=45, ha='right')  # ✅ 使用标签
            plt.grid(True, alpha=0.3)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "arr_per_dim_rmse.png"))
            
            
            # 混淆矩阵
            cm = _conf_mat(Ytrue, yhat, self.num_classes)
            fig = plt.figure(figsize=(10, 8))  # 增大图片尺寸以容纳数字
            plt.imshow(cm, interpolation="nearest", cmap='Blues')

            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # 添加数字和百分比标注
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                            ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black',
                            fontsize=7)

            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()

            # 添加坐标轴标签
            nc = self.num_classes if self.num_classes else cm.shape[0]
            tick_marks = np.arange(nc)
            plt.xticks(tick_marks, self.labels[:nc], rotation=45, ha='right')  
            plt.yticks(tick_marks, self.labels[:nc])  


            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "cls_confusion_matrix.png"))
            metrics["classify"].update(_cls_report(cm))

        # --- 保存结果 ---
        _save_metrics(metrics, out_dir)
        _save_preds_targets(P, T, out_dir)
        return metrics



@register("evaluator", "arr_cls_1")
class ArrClsEvaluator(BaseEvaluator):
    """
    数组回归 + 分类：
      * 不使用 mask
      * 分类由回归数组派生：
          logits = -(y_arr_reg_pred - center) / tau
      * 计算 MAE / RMSE / 分类准确率 / 可选交叉熵
    """
    def __init__(self, output_dir, device=None, *,
                 tau: float = 1.0, center: bool = True,
                 report_ce: bool = True, save_plots: bool = True, num_classes: int = None):
        super().__init__(output_dir, device)
        self.tau = float(tau)
        self.center = bool(center)
        self.report_ce = bool(report_ce)
        self.save_plots = bool(save_plots)
        self.num_classes = num_classes
        if self.num_classes ==6:
            self.labels = scenario_id_list_6
        elif self.num_classes ==15:
            self.labels = scenario_id_list_15
        
    def _reset(self):
        self._arr_p: List[torch.Tensor] = []
        self.logits_p : List[torch.Tensor] = []
        self._arr_t: List[torch.Tensor] = []
        self._ytrue: List[torch.Tensor] = []

    def _accumulate(self, batch, model_out):
        P = model_out["pred_arr_reg"].detach().cpu()
        T = batch["y_arr_reg"].detach().cpu()
        L = model_out["logits"].detach().cpu()
        self._arr_p.append(P)
        self._arr_t.append(T)
        self.logits_p.append(L)
        if "y_cls" in batch:
            self._ytrue.append(batch["y_cls"].detach().cpu())

    def _finalize(self, out_dir: str):
        P = torch.cat(self._arr_p, dim=0).numpy()  # [N,K]
        T = torch.cat(self._arr_t, dim=0).numpy()
        L = torch.cat(self.logits_p, dim=0).numpy()
        N, K = P.shape

        # --- 数组回归 ---
        diff = P - T
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        metrics = {"array": {"MAE": mae, "RMSE": rmse}, "classify": {}}

        # --- 分类 logits ---
        if self.center:
            
            L_center = L - L.mean(axis=1, keepdims=True)
            Z = -L_center / self.tau
        else:
            Z = -P / self.tau
        Z = Z - Z.max(axis=1, keepdims=True)
        yhat = Z.argmax(axis=1)
        Ytrue = torch.cat(self._ytrue, dim=0).numpy() if self._ytrue else T.argmin(axis=1)
        acc = float((yhat == Ytrue).mean())
        metrics["classify"]["acc"] = acc

        # --- 可选 CE ---
        if self.report_ce and len(Ytrue) == N:
            ce = float(np.mean(-Z[np.arange(N), Ytrue] + _logsumexp(Z, axis=1)))
            metrics["classify"]["cross_entropy"] = ce

        # --- 绘图 ---
        if self.save_plots:
            # 误差直方图
            fig = plt.figure()
            plt.hist(diff.reshape(-1), bins=40)
            plt.xlabel("Element-wise Error (pred - true)")
            plt.ylabel("Count")
            plt.title("Array Regression Error")
            _plot_save(fig, os.path.join(out_dir, "arr_error_hist.png"))

            # every dimension MAE
            per_dim_mae = np.mean(np.abs(diff), axis=0)
            fig = plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(per_dim_mae)), per_dim_mae)
            plt.xlabel("strategy")
            plt.ylabel("MAE")
            plt.title("Per-dimension MAE")
            plt.xticks(range(len(self.labels)), self.labels, rotation=45, ha='right')  
            plt.grid(True, alpha=0.3)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "arr_per_dim_mae.png"))
            
            
            #every dimension RMSE
            per_dim_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
            fig = plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(per_dim_rmse)), per_dim_rmse)
            plt.xlabel("Dimension")
            plt.ylabel("RMSE")
            plt.title("Per-dimension RMSE")
            plt.xticks(range(len(self.labels)), self.labels, rotation=45, ha='right')  # ✅ 使用标签
            plt.grid(True, alpha=0.3)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "arr_per_dim_rmse.png"))
            
            
            # 混淆矩阵
            cm = _conf_mat(Ytrue, yhat, self.num_classes)
            fig = plt.figure(figsize=(10, 8))  # 增大图片尺寸以容纳数字
            plt.imshow(cm, interpolation="nearest", cmap='Blues')

            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # 添加数字和百分比标注
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                            ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black',
                            fontsize=7)

            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()

            # 添加坐标轴标签
            nc = self.num_classes if self.num_classes else cm.shape[0]
            tick_marks = np.arange(nc)
            plt.xticks(tick_marks, self.labels[:nc], rotation=45, ha='right')  
            plt.yticks(tick_marks, self.labels[:nc])  


            plt.tight_layout()
            _plot_save(fig, os.path.join(out_dir, "cls_confusion_matrix.png"))
            metrics["classify"].update(_cls_report(cm))

        # --- 保存结果 ---
        _save_metrics(metrics, out_dir)
        _save_preds_targets(P, T, out_dir)
        return metrics
    
    



def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _plot_save(fig, path: str):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def rmse_all_pairs(Yhat: np.ndarray, Y: np.ndarray) -> float:
    """
    Input:
        Yhat: [N,K] predicted time
        Y:    [N,K] true time
    Output:
        RMSE over all N*K entries
    """
    diff = Yhat - Y
    return float(np.sqrt(np.mean(diff ** 2)))


def rmse_true_min(Yhat: np.ndarray, Y: np.ndarray) -> float:
    """
    RMSE on per-instance true-optimal entry:
        compare Yhat[n, k*(n)] vs min_k Y[n,k]
    """
    k_star = np.argmin(Yhat, axis=1)                 # [N]
    Yhat_star = Y[np.arange(Y.shape[0]), k_star] 
    j_star = np.argmin(Y, axis=1) # [N]
    Y_star = Y[np.arange(Y.shape[0]), j_star]
    return float(np.sqrt(np.mean((Yhat_star - Y_star) ** 2)))


def selection_regret(Yhat: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """
    Decision regret for selecting argmin_k Yhat[n,k].

    Returns:
        regret_mean: mean of (Y[n, khat(n)] - min_k Y[n,k])
        regret_median, regret_p90
    """
    N, K = Y.shape
    k_hat = np.argmin(Yhat, axis=1)               # [N]
    y_sel = Y[np.arange(N), k_hat]                # true time of selected cfg
    y_opt = np.min(Y, axis=1)                     # [N]
    r = (y_sel - y_opt).clip(min=0.0)

    def _p(a, q): return float(np.percentile(a, q))

    return {
        "regret_mean": float(np.mean(r)),
        "regret_median": float(np.median(r)),
        "regret_p90": _p(r, 90),
    }



def build_virtual_strategy_array(Yhat: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the dynamic virtual strategy column by selecting, for each instance n,
    the strategy k*(n)=argmin_k Yhat[n,k], and then reading its TRUE runtime
    Y[n, k*(n)].

    Returns:
        Y_with_virtual: [N, K+1], last column is the virtual strategy's true runtime
        k_star: [N], predicted best strategy index per instance
    """
    k_star = np.argmin(Yhat, axis=1)
    y_virtual = Y[np.arange(Y.shape[0]), k_star]
    Y_with_virtual = np.concatenate([Y, y_virtual[:, None]], axis=1)
    return Y_with_virtual, k_star



def plot_near_optimal_coverage_with_virtual(
    Yhat: np.ndarray,
    Y: np.ndarray,
    out_path: str,
    labels: Optional[List[str]] = None,
    virtual_strategy_name: str = "Virtual_BestByPred",
    taus: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Reproduce the KPI notebook's near-optimal coverage CDF after appending a
    dynamic virtual strategy.

    The regret is computed instance-wise as
        regret_run = (T - T_min) / T_min,
    where T_min is the true best runtime among the original K strategies.

    The virtual strategy is built from the model prediction:
        k*(n) = argmin_k Yhat[n,k],
    and its runtime is the TRUE runtime Y[n, k*(n)].
    """
    if taus is None:
        taus = np.linspace(0.0, 1.0, 101)

    Y_with_virtual, k_star = build_virtual_strategy_array(Yhat, Y)
    y_opt = np.min(Y, axis=1)
    denom = np.maximum(y_opt, 1e-12)
    regret_matrix = (Y_with_virtual - y_opt[:, None]) / denom[:, None]

    K = Y.shape[1]
    base_labels = labels if (labels is not None and len(labels) == K) else [f"strategy{i+1}" for i in range(K)]
    strategy_labels = list(base_labels) + [virtual_strategy_name]

    cov_matrix = []
    for tau_val in taus:
        cov_val = (regret_matrix <= tau_val).mean(axis=0)
        cov_matrix.append(cov_val)
    cov_matrix = np.asarray(cov_matrix)

    fig = plt.figure(figsize=(10, 6))
    for i, strategy_name in enumerate(strategy_labels):
        if strategy_name == virtual_strategy_name:
            plt.plot(taus, cov_matrix[:, i], label=strategy_name, linewidth=3, color="black")
        else:
            plt.plot(taus, cov_matrix[:, i], label=strategy_name, alpha=0.6)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Near-optimal coverage $\mathrm{Cov}_\tau$")
    plt.title("Near-optimal coverage CDF including dynamic virtual strategy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    _plot_save(fig, out_path)

    return {
        "strategy_labels": strategy_labels,
        "taus": taus,
        "cov_matrix": cov_matrix,
        "Y_with_virtual": Y_with_virtual,
        "k_star": k_star,
        "virtual_strategy_name": virtual_strategy_name,
    }

def reg_plot_near_optimal_coverage_with_virtual(
    Yhat: np.ndarray,
    Y: np.ndarray,
    out_path: str,
    labels: Optional[List[str]] = None,
    virtual_strategy_name: str = "Virtual_BestByPred",
    taus: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Near-optimal coverage CDF after appending a dynamic virtual strategy.

    Now Yhat and Y are both regret values:
        Yhat[n, k] = predicted regret
        Y[n, k]    = true regret

    The virtual strategy is built from the model prediction:
        k*(n) = argmin_k Yhat[n, k]

    Its value is the TRUE regret:
        Y[n, k*(n)]
    """
    if taus is None:
        taus = np.linspace(0.0, 2.0, 101)

    Y_with_virtual, k_star = build_virtual_strategy_array(Yhat, Y)

    # Since Y is already regret, do NOT recompute regret from runtime.
    regret_matrix = Y_with_virtual

    K = Y.shape[1]
    base_labels = labels if (labels is not None and len(labels) == K) else [
        f"strategy{i+1}" for i in range(K)
    ]
    strategy_labels = list(base_labels) + [virtual_strategy_name]

    cov_matrix = []
    for tau_val in taus:
        cov_val = (regret_matrix <= tau_val).mean(axis=0)
        cov_matrix.append(cov_val)
    cov_matrix = np.asarray(cov_matrix)

    fig = plt.figure(figsize=(10, 6))
    for i, strategy_name in enumerate(strategy_labels):
        if strategy_name == virtual_strategy_name:
            plt.plot(
                taus,
                cov_matrix[:, i],
                label=strategy_name,
                linewidth=3,
                color="black",
            )
        else:
            plt.plot(
                taus,
                cov_matrix[:, i],
                label=strategy_name,
                alpha=0.6,
            )

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Coverage: $\mathbb{P}(\mathrm{regret} \leq \tau)$")
    plt.title("Regret coverage CDF including dynamic virtual strategy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    _plot_save(fig, out_path)

    return {
        "strategy_labels": strategy_labels,
        "taus": taus,
        "cov_matrix": cov_matrix,
        "Y_with_virtual": Y_with_virtual,
        "k_star": k_star,
        "virtual_strategy_name": virtual_strategy_name,
    }
def scatter_all_pairs(Yhat: np.ndarray, Y: np.ndarray, out_path: str, title: str):
    """
    Scatter plot for all (instance, cfg) pairs: x=true, y=pred.
    """
    x = Y.reshape(-1)
    y = Yhat.reshape(-1)

    fig = plt.figure()
    plt.scatter(x, y, s=6, alpha=0.25)
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel(r"True")
    plt.ylabel(r"Predicted")
    plt.title(title)
    _plot_save(fig, out_path)


def scatter_min_and_selected(
    Yhat: np.ndarray, Y: np.ndarray, out_path: str, title: str
):
    """
    Compare per-instance true regret under two decision rules:
      1) Model-selected strategy: argmin_k Yhat[n,k]
      2) One fixed global strategy: k_avg = argmin_k mean_n Y[n,k]

    Plot a scatter + boxplot for both regret distributions.

    Returns:
        Dict with summary statistics and chosen global strategy index.
    """
    N, K = Y.shape
    y_opt = np.min(Y, axis=1)

    # (1) Model-selected regret
    k_hat = np.argmin(Yhat, axis=1)
    y_sel_model = Y[np.arange(N), k_hat]
    regret_model = np.clip(y_sel_model - y_opt, a_min=0.0, a_max=None)

    # (2) Global fixed strategy selected by smallest true mean runtime on test set
    k_avg = int(np.argmin(np.mean(Y, axis=0)))
    y_sel_global = Y[:, k_avg]
    regret_global = np.clip(y_sel_global - y_opt, a_min=0.0, a_max=None)

    # Scatter + boxplot
    fig = plt.figure(figsize=(7.2, 5.6))
    x1 = np.random.normal(loc=1.0, scale=0.04, size=N)
    x2 = np.random.normal(loc=2.0, scale=0.04, size=N)
    plt.scatter(x1, regret_model, s=10, alpha=0.30, label="Model selected")
    plt.scatter(x2, regret_global, s=10, alpha=0.30, label=f"Global fixed strategy (k={k_avg})")

    plt.boxplot(
        [regret_model, regret_global],
        positions=[1, 2],
        widths=0.35,
        showfliers=False,
        medianprops={"linewidth": 2},
    )

    plt.xticks([1, 2], ["Model-selected", f"Global-best-mean (k={k_avg})"])
    plt.ylabel("True regret")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _plot_save(fig, out_path)

    def _summary(a: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
        }

    return {
        "global_best_mean_strategy": k_avg,
        "model_selected_regret": _summary(regret_model),
        "global_fixed_strategy_regret": _summary(regret_global),
    }


def sorted_heatmap_matrices(
    Y: np.ndarray, Yhat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort instances (columns) by average hardness across configs,
    and configs (rows) by average performance across instances (both on TRUE Y),
    following the paper's heatmap protocol【instances sorted by avg hardness; cfgs sorted by avg performance】.

    Returns:
        Y_sorted, Yhat_sorted, cfg_order, inst_order
    """
    # instance hardness: average over configs
    inst_score = np.mean(Y, axis=1)          # [N]
    inst_order = np.argsort(inst_score)      # easy -> hard

    # configuration performance: average over instances
    cfg_score = np.mean(Y, axis=0)           # [K]
    cfg_order = np.argsort(cfg_score)        # good -> bad

    Y_sorted = Y[inst_order, :][:, cfg_order].T        # [K,N] rows=config, cols=instance
    Yhat_sorted = Yhat[inst_order, :][:, cfg_order].T

    return Y_sorted, Yhat_sorted, cfg_order, inst_order


def plot_heatmap(
    M: np.ndarray,
    out_path: str,
    title: str,
    ytick_labels: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Heatmap for matrix M of shape [K,N] (rows=configs, cols=instances).
    Darker = faster if we use Greys_r on solve time (smaller is faster).
    """
    fig = plt.figure(figsize=(12, 4.8))
    plt.imshow(M, aspect="auto", cmap="Greys_r", vmin=vmin, vmax=vmax)
    plt.colorbar(label=r"y")
    plt.title(title)
    plt.xlabel("Instances (sorted easy → hard)")
    plt.ylabel("Configurations (sorted bad → good)")

    if ytick_labels is not None and len(ytick_labels) == M.shape[0]:
        plt.yticks(np.arange(M.shape[0]), ytick_labels)

    _plot_save(fig, out_path)


@register("evaluator", "arr_rsm_runtime")
class ArrRSMRuntimeEvaluator(BaseEvaluator):
    """
      (1) Scatter plots (true vs predicted),
      (2) RMSE regret 
      (3) Heatmaps with instances/configs sorted by mean difficulty/performance.
      (4) near-optimal coverage CDF with virtual strategy from model predictions.

    Expected tensors:
            - model_out["pred_arr_reg"] : [B,K] predicted
            - batch["y_arr_reg"]        : [B,K] true
    """
    def __init__(self, output_dir, device=None, *, 
                 save_plots: bool = True,num_classes: int = None):
        super().__init__(output_dir, device)
        self.num_classes = num_classes
        if self.num_classes ==6:
            self.labels = scenario_id_list_6
        elif self.num_classes ==15:
            self.labels = scenario_id_list_15
        self.save_plots = bool(save_plots)

    def _reset(self):
        self._Yhat: List[torch.Tensor] = []
        self._Y: List[torch.Tensor] = []
        self._recon = {}
        self._mode = None

    def _accumulate(self, batch, model_out):
        if "pred_arr_reg" in model_out and ("y_arr_reg" in batch or "y_arr_time" in batch):
            self._mode = "array"
            Yhat = model_out["pred_arr_reg"].detach().cpu()
            y_key = "y_arr_reg" if "y_arr_reg" in batch else "y_arr_time"
            Y = batch[y_key].detach().cpu()
            self._Yhat.append(Yhat)
            self._Y.append(Y)
            return

        # pointwise mode: reconstruct [N,K] from scalar predictions
        if "pred_y_reg" in model_out and ("y_arr_reg" in batch or "y_arr_time" in batch) and "scheme_id" in batch:
            self._mode = "pointwise"
            pred = model_out["pred_y_reg"].detach().cpu().view(-1)
            y_key = "y_arr_reg" if "y_arr_reg" in batch else "y_arr_time"
            Yfull = batch[y_key].detach().cpu()
            
            sid = batch["scheme_id"].detach().cpu().view(-1).long()

            if "instance_id" in batch:
                iid = batch["instance_id"].detach().cpu().view(-1).long()
            else:
                # fallback (less robust): use running row id
                start = len(self._recon)
                iid = torch.arange(start, start + pred.numel(), dtype=torch.long)

            B = pred.numel()
            for i in range(B):
                inst = int(iid[i].item())
                k = int(sid[i].item())
                y_row = Yfull[i].view(-1)
                K = int(y_row.numel())

                if inst not in self._recon:
                    self._recon[inst] = {
                        "y": y_row.clone(),
                        "yhat": torch.full((K,), float("nan"), dtype=y_row.dtype),
                        "mask": torch.zeros((K,), dtype=torch.bool),
                    }

                if 0 <= k < self._recon[inst]["yhat"].numel():
                    self._recon[inst]["yhat"][k] = pred[i]
                    self._recon[inst]["mask"][k] = True
            return

        raise KeyError(
            "arr_rsm_runtime expects either (pred_arr_reg, y_arr_reg|y_arr_time) "
            "or (pred_y_reg, scheme_id, y_arr_reg|y_arr_time)"
        )

    def _finalize(self, out_dir: str):
        _ensure_dir(out_dir)
        if self._mode == "array":
            Yhat = torch.cat(self._Yhat, dim=0).numpy()  # [N,K]
            Y = torch.cat(self._Y, dim=0).numpy()        # [N,K]
        elif self._mode == "pointwise":
            rows_y, rows_yhat = [], []
            dropped = 0
            for inst in sorted(self._recon.keys()):
                rec = self._recon[inst]
                if bool(rec["mask"].all()):
                    rows_y.append(rec["y"].view(1, -1))
                    rows_yhat.append(rec["yhat"].view(1, -1))
                else:
                    dropped += 1

            if len(rows_y) == 0:
                raise RuntimeError(
                    "No complete instances reconstructed in pointwise mode. "
                    "Please ensure split is grouped by instance_id."
                )

            Y = torch.cat(rows_y, dim=0).numpy()
            Yhat = torch.cat(rows_yhat, dim=0).numpy()
            if dropped > 0:
                print(f"[arr_rsm_runtime] Dropped incomplete instances: {dropped}")
        else:
            raise RuntimeError("arr_rsm_runtime received no valid batches")

        N, K = Y.shape
        print("===== TRUE DATA DIAGNOSTICS =====")
        print("Y overall std:", Y.std())
        spread = (Y.max(axis=1) - Y.min(axis=1))  # per-instance config spread
        print("per-instance spread median:", np.median(spread))
        print("per-instance spread mean:", spread.mean())
        print("================================")
        # ---------- Metrics ----------
        rmse1 = rmse_all_pairs(Yhat, Y)
        rmse2 = rmse_true_min(Yhat, Y)
        reg = selection_regret(Yhat, Y)

        metrics = {
            "rmse_all_pairs": rmse1,
            "rmse_true_min_entry": rmse2,
            "regret": reg,
            "N": int(N),
            "K": int(K),
        }

        # ---------- Plots ----------
        if self.save_plots:
            scatter_all_pairs(
                Yhat, Y,
                out_path=os.path.join(out_dir, "scatter_all_pairs.png"),
                title="All pairs: predicted vs true (solve time)"
            )

            regret_compare_stats = scatter_min_and_selected(
                Yhat, Y,
                out_path=os.path.join(out_dir, "regret_scatter_box_compare.png"),
                title="True regret comparison: model-selected vs global fixed strategy"
            )
            metrics["regret_compare"] = regret_compare_stats

            Y_sorted, Yhat_sorted, cfg_order, inst_order = sorted_heatmap_matrices(Y, Yhat)

            # reorder labels if provided
            ylabels = None
            if self.labels is not None and len(self.labels) == K:
                ylabels = [self.labels[i] for i in cfg_order]

            # Use one shared color scale for true/pred heatmaps for fair visual comparison.
            joint_min = float(min(np.min(Y_sorted), np.min(Yhat_sorted)))
            joint_max = float(max(np.max(Y_sorted), np.max(Yhat_sorted)))

            plot_heatmap(
                Y_sorted,
                out_path=os.path.join(out_dir, "heatmap_true.png"),
                title="True runtime matrix (sorted)",
                ytick_labels=ylabels,
                vmin=joint_min,
                vmax=joint_max,
            )
            plot_heatmap(
                Yhat_sorted,
                out_path=os.path.join(out_dir, "heatmap_pred.png"),
                title="Predicted runtime matrix (sorted)",
                ytick_labels=ylabels,
                vmin=joint_min,
                vmax=joint_max,
            )


            virtual_cov_artifacts = plot_near_optimal_coverage_with_virtual(
                Yhat,
                Y,
                out_path=os.path.join(out_dir, "near_optimal_coverage_with_virtual.png"),
                labels=self.labels if (self.labels is not None and len(self.labels) == K) else None,
                virtual_strategy_name="Virtual_BestByPred",
            )
            np.savez(
                os.path.join(out_dir, "near_optimal_coverage_with_virtual_data.npz"),
                Y_with_virtual=virtual_cov_artifacts["Y_with_virtual"],
                cov_matrix=virtual_cov_artifacts["cov_matrix"],
                taus=virtual_cov_artifacts["taus"],
                k_star=virtual_cov_artifacts["k_star"],
                strategy_labels=np.array(virtual_cov_artifacts["strategy_labels"], dtype=object),
            )
            metrics["virtual_strategy"] = {
                "name": virtual_cov_artifacts["virtual_strategy_name"],
                "mean_selected_true_runtime": float(np.mean(virtual_cov_artifacts["Y_with_virtual"][:, -1])),
                "mean_regret": float(np.mean((virtual_cov_artifacts["Y_with_virtual"][:, -1] - np.min(Y, axis=1)) / np.maximum(np.min(Y, axis=1), 1e-12))),
            }

        # ---------- Save ----------
        _save_metrics(metrics, out_dir)
        _save_preds_targets(Yhat, Y, out_dir)
        return metrics


@register("evaluator", "arr_rsm_reg")
class ArrRSMregEvaluator(BaseEvaluator):
    """
      Evaluator for regret prediction.
      Now both prediction and target are regret:
            reg = (runtime - min_runtime) / min_runtime
      Expected tensors:
            - model_out["pred_arr_reg"] : [B,K] predicted regret
            - batch["y_arr_reg"]        : [B,K] true regret
    """
    def __init__(self, output_dir, device=None, *,
                 save_plots: bool = True, num_classes: int = None):
        super().__init__(output_dir, device)
        self.num_classes = num_classes
        self.labels = None
        self.labels = scenario_id_list_15
        self.save_plots = bool(save_plots)

    def _reset(self):
        self._Yhat: List[torch.Tensor] = []
        self._Y: List[torch.Tensor] = []
        self._recon = {}
        self._mode = None

    def _accumulate(self, batch, model_out):
        if "pred_arr_reg" in model_out and "y_arr_reg" in batch:
            self._mode = "array"
            Yhat = model_out["pred_arr_reg"].detach().cpu()
            Y = batch["y_arr_reg"].detach().cpu()
            self._Yhat.append(Yhat)
            self._Y.append(Y)
            return

        if "pred_y_reg" in model_out and "y_arr_reg" in batch and "scheme_id" in batch:
            self._mode = "pointwise"
            pred = model_out["pred_y_reg"].detach().cpu().view(-1)
            Yfull = batch["y_arr_reg"].detach().cpu()
            sid = batch["scheme_id"].detach().cpu().view(-1).long()
            if "instance_id" in batch:
                iid = batch["instance_id"].detach().cpu().view(-1).long()
            else:
                start = len(self._recon)
                iid = torch.arange(start, start + pred.numel(), dtype=torch.long)

            B = pred.numel()
            for i in range(B):
                inst = int(iid[i].item())
                k = int(sid[i].item())
                y_row = Yfull[i].view(-1)
                K = int(y_row.numel())

                if inst not in self._recon:
                    self._recon[inst] = {
                        "y": y_row.clone(),
                        "yhat": torch.full((K,), float("nan"), dtype=y_row.dtype),
                        "mask": torch.zeros((K,), dtype=torch.bool),
                    }

                if 0 <= k < self._recon[inst]["yhat"].numel():
                    self._recon[inst]["yhat"][k] = pred[i]
                    self._recon[inst]["mask"][k] = True

            return

        raise KeyError(
            "arr_rsm_reg expects either "
            "(pred_arr_reg, y_arr_reg) or "
            "(pred_y_reg, scheme_id, y_arr_reg)"
        )

    def _finalize(self, out_dir: str):
        _ensure_dir(out_dir)

        if self._mode == "array":
            Yhat = torch.cat(self._Yhat, dim=0).numpy()  # [N,K], predicted regret
            Y = torch.cat(self._Y, dim=0).numpy()        # [N,K], true regret

        elif self._mode == "pointwise":
            rows_y, rows_yhat = [], []
            dropped = 0

            for inst in sorted(self._recon.keys()):
                rec = self._recon[inst]

                if bool(rec["mask"].all()):
                    rows_y.append(rec["y"].view(1, -1))
                    rows_yhat.append(rec["yhat"].view(1, -1))
                else:
                    dropped += 1

            if len(rows_y) == 0:
                raise RuntimeError(
                    "No complete instances reconstructed in pointwise mode. "
                    "Please ensure split is grouped by instance_id."
                )

            Y = torch.cat(rows_y, dim=0).numpy()
            Yhat = torch.cat(rows_yhat, dim=0).numpy()

            if dropped > 0:
                print(f"[arr_rsm_reg] Dropped incomplete instances: {dropped}")

        else:
            raise RuntimeError("arr_rsm_reg received no valid batches")

        N, K = Y.shape

        print("===== TRUE REGRET DATA DIAGNOSTICS =====")
        print("Y regret overall std:", Y.std())
        spread = Y.max(axis=1) - Y.min(axis=1)
        print("per-instance regret spread median:", np.median(spread))
        print("per-instance regret spread mean:", spread.mean())
        print("=======================================")

        # ---------- Metrics ----------
        rmse = rmse_all_pairs(Yhat, Y)

        # model chooses the config with smallest predicted regret
        k_pred = np.argmin(Yhat, axis=1)

        # since Y is already true regret, directly read selected true regret
        selected_true_reg = Y[np.arange(N), k_pred]

        metrics = {
            "rmse_all_pairs_reg": float(rmse),
            "mean_selected_true_reg": float(np.mean(selected_true_reg)),
            "median_selected_true_reg": float(np.median(selected_true_reg)),
            "max_selected_true_reg": float(np.max(selected_true_reg)),
            "N": int(N),
            "K": int(K),
        }

        # ---------- Plots ----------
        if self.save_plots:
            scatter_all_pairs(
                Yhat, Y,
                out_path=os.path.join(out_dir, "scatter_all_pairs.png"),
                title="All pairs: predicted vs true regret"
            )

            regret_compare_stats = scatter_min_and_selected(
                Yhat, Y,
                out_path=os.path.join(out_dir, "regret_scatter_box_compare.png"),
                title="True regret comparison: model-selected vs global fixed strategy"
            )
            metrics["regret_compare"] = regret_compare_stats

            Y_sorted, Yhat_sorted, cfg_order, inst_order = sorted_heatmap_matrices(Y, Yhat)

            ylabels = None
            if self.labels is not None and len(self.labels) == K:
                ylabels = [self.labels[i] for i in cfg_order]

            joint_min = float(min(np.min(Y_sorted), np.min(Yhat_sorted)))
            joint_max = float(max(np.max(Y_sorted), np.max(Yhat_sorted)))

            plot_heatmap(
                Y_sorted,
                out_path=os.path.join(out_dir, "heatmap_true.png"),
                title="True regret matrix (sorted)",
                ytick_labels=ylabels,
                vmin=joint_min,
                vmax=joint_max,
            )

            plot_heatmap(
                Yhat_sorted,
                out_path=os.path.join(out_dir, "heatmap_pred.png"),
                title="Predicted regret matrix (sorted)",
                ytick_labels=ylabels,
                vmin=joint_min,
                vmax=joint_max,
            )

            virtual_cov_artifacts = reg_plot_near_optimal_coverage_with_virtual(
                Yhat,
                Y,
                out_path=os.path.join(out_dir, "near_optimal_coverage_with_virtual.png"),
                labels=self.labels if (self.labels is not None and len(self.labels) == K) else None,
                virtual_strategy_name="Virtual_BestByPred",
            )

            np.savez(
                os.path.join(out_dir, "near_optimal_coverage_with_virtual_data.npz"),
                Y_with_virtual=virtual_cov_artifacts["Y_with_virtual"],
                cov_matrix=virtual_cov_artifacts["cov_matrix"],
                taus=virtual_cov_artifacts["taus"],
                k_star=virtual_cov_artifacts["k_star"],
                strategy_labels=np.array(virtual_cov_artifacts["strategy_labels"], dtype=object),
            )

            # Y_with_virtual[:, -1] is already selected true regret
            metrics["virtual_strategy"] = {
                "name": virtual_cov_artifacts["virtual_strategy_name"],
                "mean_selected_true_reg": float(np.mean(virtual_cov_artifacts["Y_with_virtual"][:, -1])),
                "median_selected_true_reg": float(np.median(virtual_cov_artifacts["Y_with_virtual"][:, -1])),
                "max_selected_true_reg": float(np.max(virtual_cov_artifacts["Y_with_virtual"][:, -1])),
            }

        # ---------- Save ----------
        _save_metrics(metrics, out_dir)
        _save_preds_targets(Yhat, Y, out_dir)

        return metrics
    
    
@register("evaluator", "rsm_reg")
class ArrRSMregEvaluator(BaseEvaluator):
    """
      Evaluator for regret prediction.
      Now both prediction and target are regret:
            reg = (runtime - min_runtime) / min_runtime
      Expected tensors:
            - model_out["pred_reg"] : [B,K] predicted regret
            - batch["y_reg"]        : [B,K] true regret
    """
    def __init__(self, output_dir, device=None, *,
                 save_plots: bool = True, num_classes: int = None):
        super().__init__(output_dir, device)
        self.num_classes = num_classes
        self.labels = None
        self.labels = scenario_id_list_15
        self.save_plots = bool(save_plots)

    def _reset(self):
        self._Yhat: List[torch.Tensor] = []
        self._Y: List[torch.Tensor] = []
        self._recon = {}
        self._mode = None

    def _accumulate(self, batch, model_out):
        if "pred_arr_reg" in model_out and "y_arr_reg" in batch:
            self._mode = "array"
            Yhat = model_out["pred_arr_reg"].detach().cpu()
            Y = batch["y_arr_reg"].detach().cpu()
            self._Yhat.append(Yhat)
            self._Y.append(Y)
            return

        if "pred_reg" in model_out and "y_arr_reg" in batch and "scheme_id" in batch:
            self._mode = "pointwise"
            pred = model_out["pred_reg"].detach().cpu().view(-1)
            Yfull = batch["y_arr_reg"].detach().cpu()
            sid = batch["scheme_id"].detach().cpu().view(-1).long()
            if "instance_id" in batch:
                iid = batch["instance_id"].detach().cpu().view(-1).long()
            else:
                start = len(self._recon)
                iid = torch.arange(start, start + pred.numel(), dtype=torch.long)

            B = pred.numel()
            for i in range(B):
                inst = int(iid[i].item())
                k = int(sid[i].item())
                y_row = Yfull[i].view(-1)
                K = int(y_row.numel())

                if inst not in self._recon:
                    self._recon[inst] = {
                        "y": y_row.clone(),
                        "yhat": torch.full((K,), float("nan"), dtype=y_row.dtype),
                        "mask": torch.zeros((K,), dtype=torch.bool),
                    }

                if 0 <= k < self._recon[inst]["yhat"].numel():
                    self._recon[inst]["yhat"][k] = pred[i]
                    self._recon[inst]["mask"][k] = True

            return

        raise KeyError(
            "arr_rsm_reg expects either "
            "(pred_arr_reg, y_arr_reg) or "
            "(pred_reg, scheme_id, y_reg)"
        )

    def _finalize(self, out_dir: str):
        _ensure_dir(out_dir)

        if self._mode == "array":
            Yhat = torch.cat(self._Yhat, dim=0).numpy()  # [N,K], predicted regret
            Y = torch.cat(self._Y, dim=0).numpy()        # [N,K], true regret

        elif self._mode == "pointwise":
            rows_y, rows_yhat = [], []
            dropped = 0

            for inst in sorted(self._recon.keys()):
                rec = self._recon[inst]

                if bool(rec["mask"].all()):
                    rows_y.append(rec["y"].view(1, -1))
                    rows_yhat.append(rec["yhat"].view(1, -1))
                else:
                    dropped += 1

            if len(rows_y) == 0:
                raise RuntimeError(
                    "No complete instances reconstructed in pointwise mode. "
                    "Please ensure split is grouped by instance_id."
                )

            Y = torch.cat(rows_y, dim=0).numpy()
            Yhat = torch.cat(rows_yhat, dim=0).numpy()

            if dropped > 0:
                print(f"[arr_rsm_reg] Dropped incomplete instances: {dropped}")

        else:
            raise RuntimeError("arr_rsm_reg received no valid batches")

        N, K = Y.shape

        print("===== TRUE REGRET DATA DIAGNOSTICS =====")
        print("Y regret overall std:", Y.std())
        spread = Y.max(axis=1) - Y.min(axis=1)
        print("per-instance regret spread median:", np.median(spread))
        print("per-instance regret spread mean:", spread.mean())
        print("=======================================")

        # ---------- Metrics ----------
        rmse = rmse_all_pairs(Yhat, Y)

        # model chooses the config with smallest predicted regret
        k_pred = np.argmin(Yhat, axis=1)

        # since Y is already true regret, directly read selected true regret
        selected_true_reg = Y[np.arange(N), k_pred]

        metrics = {
            "rmse_all_pairs_reg": float(rmse),
            "mean_selected_true_reg": float(np.mean(selected_true_reg)),
            "median_selected_true_reg": float(np.median(selected_true_reg)),
            "max_selected_true_reg": float(np.max(selected_true_reg)),
            "N": int(N),
            "K": int(K),
        }

        # ---------- Plots ----------
        if self.save_plots:
            scatter_all_pairs(
                Yhat, Y,
                out_path=os.path.join(out_dir, "scatter_all_pairs.png"),
                title="All pairs: predicted vs true regret"
            )

            regret_compare_stats = scatter_min_and_selected(
                Yhat, Y,
                out_path=os.path.join(out_dir, "regret_scatter_box_compare.png"),
                title="True regret comparison: model-selected vs global fixed strategy"
            )
            metrics["regret_compare"] = regret_compare_stats

            Y_sorted, Yhat_sorted, cfg_order, inst_order = sorted_heatmap_matrices(Y, Yhat)

            ylabels = None
            if self.labels is not None and len(self.labels) == K:
                ylabels = [self.labels[i] for i in cfg_order]

            joint_min = float(min(np.min(Y_sorted), np.min(Yhat_sorted)))
            joint_max = float(max(np.max(Y_sorted), np.max(Yhat_sorted)))

            plot_heatmap(
                Y_sorted,
                out_path=os.path.join(out_dir, "heatmap_true.png"),
                title="True regret matrix (sorted)",
                ytick_labels=ylabels,
                vmin=joint_min,
                vmax=joint_max,
            )

            plot_heatmap(
                Yhat_sorted,
                out_path=os.path.join(out_dir, "heatmap_pred.png"),
                title="Predicted regret matrix (sorted)",
                ytick_labels=ylabels,
                vmin=joint_min,
                vmax=joint_max,
            )
            #主要这个绘图函数在pointwise预测的时候会产生问题
            virtual_cov_artifacts = reg_plot_near_optimal_coverage_with_virtual(
                Yhat,
                Y,
                out_path=os.path.join(out_dir, "near_optimal_coverage_with_virtual.png"),
                labels=self.labels if (self.labels is not None and len(self.labels) == K) else None,
                virtual_strategy_name="Virtual_BestByPred",
            )

            np.savez(
                os.path.join(out_dir, "near_optimal_coverage_with_virtual_data.npz"),
                Y_with_virtual=virtual_cov_artifacts["Y_with_virtual"],
                cov_matrix=virtual_cov_artifacts["cov_matrix"],
                taus=virtual_cov_artifacts["taus"],
                k_star=virtual_cov_artifacts["k_star"],
                strategy_labels=np.array(virtual_cov_artifacts["strategy_labels"], dtype=object),
            )

            # Y_with_virtual[:, -1] is already selected true regret
            metrics["virtual_strategy"] = {
                "name": virtual_cov_artifacts["virtual_strategy_name"],
                "mean_selected_true_reg": float(np.mean(virtual_cov_artifacts["Y_with_virtual"][:, -1])),
                "median_selected_true_reg": float(np.median(virtual_cov_artifacts["Y_with_virtual"][:, -1])),
                "max_selected_true_reg": float(np.max(virtual_cov_artifacts["Y_with_virtual"][:, -1])),
            }

        # ---------- Save ----------
        _save_metrics(metrics, out_dir)
        _save_preds_targets(Yhat, Y, out_dir)

        return metrics
    