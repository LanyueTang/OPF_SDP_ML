import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 向上2层到 Lanyue/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from registries import register
import torch
import numpy as np
from gcn_utils.graph_varients import make_graph_varient

DIAG_SAVE_DIR = str(PROJECT_ROOT / "result" / "reader_diagnostics")

def convert_samples_to_torch(samples):
    """transfer numpy samples to torch samples"""
    torch_samples = []
    for sample in samples:
        torch_sample = {}
        for key, value in sample.items():
            if key == "A":
                edge_index, edge_weight = value
                torch_sample[key] = value# a 
            if key == "y_cls":
                    torch_sample[key] = torch.tensor(value).long()
            elif key == "y_reg":
                    torch_sample[key] = torch.tensor(value).float()
            elif isinstance(value, np.ndarray): # y_cls transfer to long
                torch_sample[key] = torch.from_numpy(value).float()
            else:
                torch_sample[key] = value
        torch_samples.append(torch_sample)
    return torch_samples

def load_samples_from_npz(npz_path):
    """from npz get samples"""
    print(f"📖 Loading samples from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    num_samples = int(data['num_samples'])
    
    samples = []
    for i in range(num_samples):
        sample = data[f'sample_{i}'].item()  
        samples.append(sample)
    print(f"✅ Loaded {len(samples)} samples")
    return samples

def analyze_and_plot_samples(samples, save_dir='/home/goatoine/Documents/Lanyue/models/GNN/result/'):
    """
    Analyze y_cls and y_reg distribution of samples and generate distribution plots

    Args:
        samples: list of dicts, each containing keys 'y_cls' and 'y_reg'
        save_dir: directory to save plots
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ✅ 
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

    # ===== 1. y_cls class distribution =====
    y_cls_values = [int(sample['y_cls']) for sample in samples]
    unique, counts = np.unique(y_cls_values, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"📈 y_cls class distribution: {class_dist}")

    # --- Plot y_cls histogram with scenario labels ---
    plt.figure(figsize=(14, 6))  # ✅ 增大宽度以容纳长标签
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='black', alpha=0.8)
    
    # ✅ 添加数值标注
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # ✅ 使用场景名称作为 x 轴标签
    plt.xticks(
        range(len(unique)), 
        [scenario_id_list_15[idx] if idx < len(scenario_id_list_15) else f"Class_{idx}" 
         for idx in unique],
        rotation=45,  # ✅ 旋转 45 度避免重叠
        ha='right'    # ✅ 右对齐
    )
    
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('y_cls Class Distribution by Scenario', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save class distribution plot
    save_path_cls = os.path.join(save_dir, 'y_cls_distribution.png')
    plt.savefig(save_path_cls, dpi=300, bbox_inches='tight')
    print(f"✅ y_cls distribution plot saved to: {save_path_cls}")
    plt.close()

    # ===== 2. y_reg statistics =====
    y_reg_values = [float(sample['y_reg']) for sample in samples]
    print(f"📊 y_reg statistics:")
    print(f"  Min:  {np.min(y_reg_values):.6f}")
    print(f"  Max:  {np.max(y_reg_values):.6f}")
    print(f"  Mean: {np.mean(y_reg_values):.6f}")
    print(f"  Std:  {np.std(y_reg_values):.6f}")

    # --- Plot y_reg histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(y_reg_values, bins=50, alpha=0.7, edgecolor='black', color='salmon')
    plt.xlabel('y_reg values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('y_reg Distribution Histogram', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path_reg = os.path.join(save_dir, 'y_reg_distribution.png')
    plt.savefig(save_path_reg, dpi=300, bbox_inches='tight')
    print(f"✅ y_reg distribution plot saved to: {save_path_reg}")
    plt.close()


def analyze_and_plot_true_regret(samples, save_dir=DIAG_SAVE_DIR, labels=None, prefix=""):
    """
    Plot true y_arr_reg for all samples and save diagnostics.

    Saved plots:
      - true_regret_heatmap_sorted.png  (rows=config, cols=instances)
      - true_regret_hist.png            (all entries)
      - true_regret_by_config_box.png   (per-config distribution)
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    y_rows = []
    for sample in samples:
        y = sample.get("y_arr_reg", None)
        if y is None:
            continue
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)
        y = y.reshape(-1)
        y_rows.append(y)

    if len(y_rows) == 0:
        print("⚠️ No y_arr_reg found, skip true regret plotting.")
        return

    lengths = [len(r) for r in y_rows]
    if len(set(lengths)) != 1:
        min_k = min(lengths)
        y_rows = [r[:min_k] for r in y_rows]
        print(f"⚠️ Inconsistent y_arr_reg lengths, truncated to K={min_k}")

    Y = np.stack(y_rows, axis=0)  # [N, K]
    N, K = Y.shape

    inst_score = np.mean(Y, axis=1)
    inst_order = np.argsort(inst_score)
    cfg_score = np.mean(Y, axis=0)
    cfg_order = np.argsort(cfg_score)
    Y_sorted = Y[inst_order, :][:, cfg_order].T  # [K, N]

    if labels is None or len(labels) != K:
        ytick_labels = [f"cfg_{i}" for i in cfg_order]
    else:
        ytick_labels = [labels[i] for i in cfg_order]

    name_prefix = f"{prefix}_" if prefix else ""

    fig = plt.figure(figsize=(12, 5))
    plt.imshow(Y_sorted, aspect="auto", cmap="Greys_r")
    plt.colorbar(label="true regret (y_arr_reg)")
    plt.title(f"True regret matrix (sorted)  N={N}, K={K}")
    plt.xlabel("Instances (sorted easy → hard)")
    plt.ylabel("Configurations (sorted good → bad)")
    plt.yticks(np.arange(K), ytick_labels)
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, f"{name_prefix}true_regret_heatmap_sorted.png")
    plt.savefig(heatmap_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4.5))
    plt.hist(Y.reshape(-1), bins=60, alpha=0.8, edgecolor="black")
    plt.xlabel("True regret (all entries)")
    plt.ylabel("Count")
    plt.title("True regret distribution")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    hist_path = os.path.join(save_dir, f"{name_prefix}true_regret_hist.png")
    plt.savefig(hist_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    box_data = [Y[:, i] for i in cfg_order]
    plt.boxplot(box_data, showfliers=False)
    plt.xticks(np.arange(1, K + 1), ytick_labels, rotation=30, ha="right")
    plt.ylabel("True regret")
    plt.title("True regret by configuration")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    box_path = os.path.join(save_dir, f"{name_prefix}true_regret_by_config_box.png")
    plt.savefig(box_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print("✅ True regret plots saved:")
    print(f"   - {heatmap_path}")
    print(f"   - {hist_path}")
    print(f"   - {box_path}")



def _uniform_sample_pointwise_by_instance(
    samples,
    *,
    bins: int = 8,
    instances_per_bin=None,
    score_mode: str = "mean",
    seed: int = 42,
):
    """
    Uniformly sample instances by target-score bins, then keep all pointwise rows
    from selected instances.

    score_mode is computed from per-instance y_arr_reg:
      - "mean" / "median" / "max"
    """
    if len(samples) == 0:
        return samples

    inst_to_rows = {}
    inst_score = {}
    for row in samples:
        if "instance_id" not in row:
            return samples

        iid = int(torch.as_tensor(row["instance_id"]).item())
        inst_to_rows.setdefault(iid, []).append(row)

        if iid in inst_score:
            continue

        y_arr = row.get("y_arr_reg", None)
        if y_arr is None:
            y_arr = row.get("y_reg", None)
        if y_arr is None:
            continue

        y = torch.as_tensor(y_arr, dtype=torch.float32).view(-1)
        if y.numel() == 0:
            continue

        if score_mode == "median":
            score = float(torch.median(y).item())
        elif score_mode == "max":
            score = float(torch.max(y).item())
        else:
            score = float(torch.mean(y).item())
        inst_score[iid] = score

    inst_ids = [iid for iid in inst_to_rows.keys() if iid in inst_score]
    if len(inst_ids) == 0:
        print("⚠️ Uniform instance sampling skipped: no valid instance scores.")
        return samples

    values = np.array([inst_score[iid] for iid in inst_ids], dtype=np.float64)
    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        print("⚠️ Uniform instance sampling skipped: target scores are constant.")
        return samples

    bins = max(2, int(bins))
    edges = np.linspace(vmin, vmax, bins + 1)
    buckets = [[] for _ in range(bins)]
    for iid in inst_ids:
        v = inst_score[iid]
        bid = int(np.searchsorted(edges, v, side="right") - 1)
        bid = max(0, min(bins - 1, bid))
        buckets[bid].append(iid)

    non_empty = [bucket for bucket in buckets if len(bucket) > 0]
    if len(non_empty) == 0:
        print("⚠️ Uniform instance sampling skipped: all bins are empty.")
        return samples

    non_empty_sizes = np.array([len(bucket) for bucket in non_empty], dtype=np.int64)
    if instances_per_bin is None:
        # robust default: avoid being dominated by a single extremely sparse bin
        k = max(1, int(np.floor(np.percentile(non_empty_sizes, 25))))
    else:
        k = max(1, int(instances_per_bin))

    rng = np.random.default_rng(seed)
    selected_ids = []
    for bucket in non_empty:
        if len(bucket) <= k:
            selected_ids.extend(bucket)
        else:
            pick = rng.choice(np.array(bucket), size=k, replace=False)
            selected_ids.extend([int(x) for x in pick.tolist()])

    selected_set = set(selected_ids)
    sampled = []
    for iid in selected_ids:
        sampled.extend(inst_to_rows[iid])

    before_inst = len(inst_to_rows)
    after_inst = len(selected_set)
    print(
        f"🔀 Uniform instance sampling: {before_inst} -> {after_inst} instances "
        f"(bins={bins}, score_mode={score_mode}, per_bin={k if instances_per_bin is None else instances_per_bin})"
    )
    print(f"🔀 Non-empty bin sizes: {non_empty_sizes.tolist()}")
    print(f"🔀 Pointwise rows after uniform instance sampling: {len(samples)} -> {len(sampled)}")
    return sampled






@register("reader", "opf_vA")   
class OPFReaderVA:
    def __init__(self, npz_path = None, adj_mode = "real", **kwargs):
        self.sample_file_path = npz_path
        self.adj_mode = adj_mode
        self.samples = self._load_v1()
            
    def _load_v1(self):
        sample_file = self.sample_file_path
        samples = load_samples_from_npz(sample_file)
        print(f"📊 summary: {len(samples)} samples")
        samples = convert_samples_to_torch(samples)
        samples = [s for s in samples if "original_0" in s.get("scenario_id", "")]
        # keep only one sample for each loadfile
        seen_files = set()
        unique_samples = []
        for sample in samples:
            source_file = sample.get('source_file', '')
            if source_file not in seen_files:
                seen_files.add(source_file)
                unique_samples.append(sample)

        print(f"🔧 After deduplication: {len(unique_samples)} samples (from {len(samples)})")
        samples = unique_samples
        analyze_and_plot_samples(samples)

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
        analyze_and_plot_true_regret(
            samples,
            save_dir=DIAG_SAVE_DIR,
            labels=scenario_id_list_15,
            prefix="opf_vA",
        )

        key_features = ["A_grid", "node_load", "y_cls", "y_arr_reg"]
        filtered_samples = []
        for sample in samples:
            filtered_sample = {}
            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")
            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")
        out_samples = []
        s = samples[0]
        A_grid = s["A_grid"] 
        num_nodes = s["node_load"].shape[0]
        # node_load在这里没有归一化在主程序里面归一化了
        A_hat = make_graph_varient(A_grid, num_nodes, mode=self.adj_mode)
        for s in samples:
            new_s = dict(s)
            new_s["A_grid"] = A_hat       
            out_samples.append(new_s)
        return out_samples

        return samples
    def load(self):
        return self.samples

@register("reader", "opf_reg_array")   
class OPFReaderVA_reg_array:
    def __init__(self, npz_path = None, adj_mode = "real", **kwargs):
        self.sample_file_path = npz_path
        self.adj_mode = adj_mode
        self.samples = self._load_v1()
        
          
    def _load_v1(self):
        sample_file = self.sample_file_path
        samples = load_samples_from_npz(sample_file)
        print(f"📊 summary: {len(samples)} samples")
        samples = convert_samples_to_torch(samples)
        samples = [s for s in samples if "original_0" in s.get("scenario_id", "")]
        # keep only one sample for each loadfile
        seen_files = set()
        unique_samples = []
        for sample in samples:
            source_file = sample.get('source_file', '')
            if source_file not in seen_files:
                seen_files.add(source_file)
                unique_samples.append(sample)

        print(f"🔧 After deduplication: {len(unique_samples)} samples (from {len(samples)})")
        samples = unique_samples
        analyze_and_plot_samples(samples)

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
        analyze_and_plot_true_regret(
            samples,
            save_dir=DIAG_SAVE_DIR,
            labels=scenario_id_list_15,
            prefix="opf_vA",
        )

        key_features = ["A_grid", "node_load", "y_cls", "y_arr_reg"]
        filtered_samples = []
        for sample in samples:
            filtered_sample = {}
            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")
            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")
        out_samples = []
        s = samples[0]
        A_grid = s["A_grid"] 
        num_nodes = s["node_load"].shape[0]
        # node_load在这里没有归一化在主程序里面归一化了
        A_hat = make_graph_varient(A_grid, num_nodes, mode=self.adj_mode)
        for s in samples:
            new_s = dict(s)
            new_s["A_grid"] = A_hat       
            out_samples.append(new_s)
        return out_samples

        return samples
    def load(self):
        return self.samples
    
@register("reader", "opf_runtime_array")   
class OPFReaderVA_reg_array:
    def __init__(self, npz_path = None, adj_mode = "real", **kwargs):
        self.sample_file_path = npz_path
        self.adj_mode = adj_mode
        self.samples = self._load_v1()
        
          
    def _load_v1(self):
        sample_file = self.sample_file_path
        samples = load_samples_from_npz(sample_file)
        print(f"📊 summary: {len(samples)} samples")
        samples = convert_samples_to_torch(samples)
        samples = [s for s in samples if "original_0" in s.get("scenario_id", "")]
        # keep only one sample for each loadfile
        seen_files = set()
        unique_samples = []
        for sample in samples:
            source_file = sample.get('source_file', '')
            if source_file not in seen_files:
                seen_files.add(source_file)
                unique_samples.append(sample)

        print(f"🔧 After deduplication: {len(unique_samples)} samples (from {len(samples)})")
        samples = unique_samples
        analyze_and_plot_samples(samples)

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
        analyze_and_plot_true_regret(
            samples,
            save_dir=DIAG_SAVE_DIR,
            labels=scenario_id_list_15,
            prefix="opf_vA",
        )

        key_features = ["A_grid", "node_load", "y_cls", "y_arr_time"]
        filtered_samples = []
        for sample in samples:
            filtered_sample = {}
            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")
            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")
        out_samples = []
        s = samples[0]
        A_grid = s["A_grid"] 
        num_nodes = s["node_load"].shape[0]
        # node_load在这里没有归一化在主程序里面归一化了
        A_hat = make_graph_varient(A_grid, num_nodes, mode=self.adj_mode)
        for s in samples:
            new_s = dict(s)
            new_s["A_grid"] = A_hat       
            out_samples.append(new_s)
        return out_samples

        return samples
    def load(self):
        return self.samples
    
@register("reader", "opf_reg_pointwise")
class OPFReaderVA_reg_pointwise:
    """
    Pointwise OPF reader.

    Each sample corresponds to one scheme/config of one OPF instance.

    Required output keys:
        A_grid
        node_load
        y_cls
        y_reg
        y_arr_reg
        instance_id   # integer id encoded from source_file
        scheme_id     # integer id encoded from scenario_id suffix
    """

    def __init__(self, npz_path=None, adj_mode="real", **kwargs):
        self.sample_file_path = npz_path
        self.adj_mode = adj_mode
        self.samples = self._load()

    def _load(self):
        sample_file = self.sample_file_path
        samples = load_samples_from_npz(sample_file)
        print(f"📊 summary: {len(samples)} samples")

        samples = convert_samples_to_torch(samples)
        samples = [s for s in samples if "original_0" in s.get("scenario_id", "")]

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

        # --------------------------------------------------
        # Build integer mappings
        # --------------------------------------------------
        scheme_to_id = {name: i for i, name in enumerate(scenario_id_list_15)}
        instance_to_id = {}

        for s in samples:
            source_file = s["source_file"]
            if source_file not in instance_to_id:
                instance_to_id[source_file] = len(instance_to_id)
            s["instance_id"] = torch.tensor(
                instance_to_id[source_file],
                dtype=torch.long,
            )
            scenario_id = s["scenario_id"]
            scheme_name = None
            for name in scenario_id_list_15:
                if scenario_id.endswith(name):
                    scheme_name = name
                    break

            s["scheme_id"] = torch.tensor(
                scheme_to_id[scheme_name],
                dtype=torch.long,
            )

        print(f"✅ Built instance_id from source_file: {len(instance_to_id)} unique instances")
        print(f"✅ Built scheme_id from scenario_id: {len(scheme_to_id)} schemes")

        # --------------------------------------------------
        # Diagnostics before feature filtering
        # --------------------------------------------------
        analyze_and_plot_true_regret(
            samples,
            save_dir=DIAG_SAVE_DIR,
            labels=scenario_id_list_15,
            prefix="opf_vA",
        )

        # --------------------------------------------------
        # Optional sanity check: each instance should have 15 schemes
        # --------------------------------------------------
        from collections import defaultdict
        instance_to_schemes = defaultdict(list)
        for s in samples:
            iid = int(s["instance_id"].item())
            sid = int(s["scheme_id"].item())
            instance_to_schemes[iid].append(sid)

        bad_instances = []
        expected = list(range(len(scenario_id_list_15)))

        for iid, scheme_ids in instance_to_schemes.items():
            if sorted(scheme_ids) != expected:
                bad_instances.append((iid, sorted(scheme_ids)))

        if len(bad_instances) > 0:
            print(f"⚠️ Found {len(bad_instances)} incomplete / abnormal instances.")
            print("Example bad instances:")
            for iid, scheme_ids in bad_instances[:5]:
                print(f"  instance_id={iid}, scheme_ids={scheme_ids}")
        else:
            print("✅ Sanity check passed: every instance has all 15 schemes.")

        # --------------------------------------------------
        # Keep only model-needed features
        # --------------------------------------------------
        key_features = [
            "A",
            "node_load",
            "y_cls",
            "y_reg",
            "y_arr_reg",
            "instance_id",
            "scheme_id",
        ]

        filtered_samples = []
        for sample in samples:
            filtered_sample = {}
            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")
            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")

        # s0 = samples[0]
        # A_grid = s0["A"]
        # num_nodes = s0["node_load"].shape[0]
        # # node_load is not normalized here; it is normalized in main program
        # A_hat = make_graph_varient(A_grid, num_nodes, mode=self.adj_mode)
        # out_samples = []
        # for s in samples:
        #     new_s = dict(s)
        #     new_s["A_grid"] = A_hat
        #     out_samples.append(new_s)

        return samples

    def load(self):
        return self.samples
    

import numpy as np
import torch
import networkx as nx
def edge_list_to_dense_numpy(A_raw):
    """
    A_raw can be:
        1) (edge_index, edge_weight)
        2) dense numpy array
        3) dense torch tensor
    """
    if isinstance(A_raw, (tuple, list)) and len(A_raw) == 2:
        edge_index, edge_weight = A_raw

        if torch.is_tensor(edge_index):
            edge_index = edge_index.detach().cpu().numpy()
        if torch.is_tensor(edge_weight):
            edge_weight = edge_weight.detach().cpu().numpy()

        edge_index = np.asarray(edge_index)
        edge_weight = np.asarray(edge_weight)

        N = int(edge_index.max()) + 1
        A = np.zeros((N, N), dtype=np.float32)

        src = edge_index[0].astype(np.int64)
        dst = edge_index[1].astype(np.int64)

        A[src, dst] = edge_weight.astype(np.float32)
        return A

    if torch.is_tensor(A_raw):
        return A_raw.detach().cpu().numpy().astype(np.float32)

    return np.asarray(A_raw, dtype=np.float32)


def build_clique_pool_list_from_A(A_raw, use_clique_tree=True, min_clique_size=2):
    import numpy as np
    import torch
    import networkx as nx

    if isinstance(A_raw, (tuple, list)) and len(A_raw) == 2:
        edge_index, edge_weight = A_raw

        if torch.is_tensor(edge_index):
            edge_index = edge_index.detach().cpu().numpy()
        else:
            edge_index = np.asarray(edge_index)

        N = int(edge_index.max()) + 1

        src = edge_index[0].astype(np.int64)
        dst = edge_index[1].astype(np.int64)

        edges = [(int(i), int(j)) for i, j in zip(src, dst) if i != j]

        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(edges)

    else:
        if torch.is_tensor(A_raw):
            A = A_raw.detach().cpu().numpy()
        else:
            A = np.asarray(A_raw)

        A_bin = (A > 0).astype(np.int32)
        np.fill_diagonal(A_bin, 0)
        G = nx.from_numpy_array(A_bin)

    N = G.number_of_nodes()

    clique_nodes = [sorted(list(c)) for c in nx.find_cliques(G)]
    clique_nodes = [c for c in clique_nodes if len(c) >= min_clique_size]

    if len(clique_nodes) == 0:
        clique_nodes = [[i] for i in range(N)]

    C = len(clique_nodes)
    clique_sets = [set(c) for c in clique_nodes]

    sep_weight = {}
    for i in range(C):
        for j in range(i + 1, C):
            s = len(clique_sets[i] & clique_sets[j])
            if s > 0:
                sep_weight[(i, j)] = float(s)

    if use_clique_tree and C > 1:
        CG = nx.Graph()
        CG.add_nodes_from(range(C))
        for (i, j), w in sep_weight.items():
            CG.add_edge(i, j, weight=w)

        if CG.number_of_edges() > 0:
            T = nx.maximum_spanning_tree(CG, weight="weight")
            A_clique_edges = []
            sep_edges = []

            for i, j, data in T.edges(data=True):
                A_clique_edges.append((int(i), int(j)))
                sep_edges.append((int(i), int(j), float(data["weight"])))

            return clique_nodes, A_clique_edges, sep_edges

    A_clique_edges = []
    sep_edges = []
    for (i, j), w in sep_weight.items():
        A_clique_edges.append((int(i), int(j)))
        sep_edges.append((int(i), int(j), float(w)))

    return clique_nodes, A_clique_edges, sep_edges



@register("reader", "opf_reg_pointwise_clique_pooling")
class OPFReaderVA_reg_pointwise:
    """
    Pointwise OPF reader with clique pooling preprocessing.

    Each sample contains:
        A
        node_load
        y_cls
        y_reg
        y_arr_reg
        instance_id
        scheme_id

    Additional clique pooling keys:
        clique_M
        A_clique
        sep_size
        clique_mask
    """

    def __init__(
        self,
        npz_path=None,
        adj_mode="real",
        use_clique_tree=True,
        min_clique_size=2,
        **kwargs,
    ):
        self.sample_file_path = npz_path
        self.adj_mode = adj_mode
        self.use_clique_tree = use_clique_tree
        self.min_clique_size = min_clique_size
        self.samples = self._load()

    def _load(self):
        sample_file = self.sample_file_path
        samples = load_samples_from_npz(sample_file)
        print(f"📊 summary: {len(samples)} samples")

        samples = convert_samples_to_torch(samples)
        samples = [s for s in samples if "original_0" in s.get("scenario_id", "")]

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

        scheme_to_id = {name: i for i, name in enumerate(scenario_id_list_15)}
        instance_to_id = {}

        for s in samples:
            source_file = s["source_file"]

            if source_file not in instance_to_id:
                instance_to_id[source_file] = len(instance_to_id)

            s["instance_id"] = torch.tensor(
                instance_to_id[source_file],
                dtype=torch.long,
            )

            scenario_id = s["scenario_id"]
            scheme_name = None

            for name in scenario_id_list_15:
                if scenario_id.endswith(name):
                    scheme_name = name
                    break

            if scheme_name is None:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")

            s["scheme_id"] = torch.tensor(
                scheme_to_id[scheme_name],
                dtype=torch.long,
            )

        print(f"✅ Built instance_id from source_file: {len(instance_to_id)} unique instances")
        print(f"✅ Built scheme_id from scenario_id: {len(scheme_to_id)} schemes")

        analyze_and_plot_true_regret(
            samples,
            save_dir=DIAG_SAVE_DIR,
            labels=scenario_id_list_15,
            prefix="opf_vA",
        )

        from collections import defaultdict

        instance_to_schemes = defaultdict(list)
        for s in samples:
            iid = int(s["instance_id"].item())
            sid = int(s["scheme_id"].item())
            instance_to_schemes[iid].append(sid)

        bad_instances = []
        expected = list(range(len(scenario_id_list_15)))

        for iid, scheme_ids in instance_to_schemes.items():
            if sorted(scheme_ids) != expected:
                bad_instances.append((iid, sorted(scheme_ids)))

        if len(bad_instances) > 0:
            print(f"⚠️ Found {len(bad_instances)} incomplete / abnormal instances.")
            print("Example bad instances:")
            for iid, scheme_ids in bad_instances[:5]:
                print(f"  instance_id={iid}, scheme_ids={scheme_ids}")
        else:
            print("✅ Sanity check passed: every instance has all 15 schemes.")

        # --------------------------------------------------
        # Build clique pooling data once
        # --------------------------------------------------
        print("🔧 Building clique pooling data from A ...")

        for idx, sample in enumerate(samples):
            if "A" not in sample:
                raise KeyError(f"Sample {idx} has no key 'A'.")

            A = sample["A"]
            A_dense = edge_list_to_dense_numpy(sample["A"])

            clique_nodes, A_clique_edges, sep_edges = build_clique_pool_list_from_A(
                sample["A"],
                use_clique_tree=self.use_clique_tree,
                min_clique_size=self.min_clique_size,
            )

            sample["clique_nodes"] = clique_nodes
            sample["A_clique_edges"] = A_clique_edges
            sample["sep_edges"] = sep_edges
            print(
                f"sample {idx}: "
                f"N={sample['node_load'].shape[0]}, "
                f"sample_instance_id={sample['scheme_id'].item()}, "
                f"C={len(sample['clique_nodes'])}, "
                f"clique_edges={len(sample['A_clique_edges'])}"
            )
        print("✅ Finished clique pooling preprocessing.")

        key_features = [
            "A",
            "node_load",
            "y_cls",
            "y_reg",
            "y_arr_reg",
            "instance_id",
            "scheme_id",
            "clique_nodes",
            "A_clique_edges",
            "sep_edges",
        ]

        filtered_samples = []

        for sample in samples:
            filtered_sample = {}

            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")

            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")

        return samples

    def load(self):
        return self.samples




@register("reader", "opf_vB")   #
class OPFReaderVB:
    def __init__(self,**kwargs):
        self.samples = self._load_v1()  
    def _load_v1(self):
        sample_file = "/home/goatoine/Documents/Lanyue/data/data_for_GCN/data_basic_GCN/case2746wop_1029_samples.npz"
        samples = load_samples_from_npz(sample_file)
        print(f"📊 总计: {len(samples)} 个样本")
        samples = convert_samples_to_torch(samples)
        # 过滤掉key值“global_vec”中存在nan的样本
        samples = [s for s in samples if not torch.isnan(s.get("global_vec", torch.tensor(0))).any()]
        # keep only one sample for each loadfile
        print(f"🔧 After deduplication: {len(samples)} samples (from {len(samples)})")
        samples = samples
        analyze_and_plot_samples(samples)
        key_features = ["A","node_load", "global_vec", "y_cls", "y_arr_reg"]
        filtered_samples = []
        for sample in samples:
            filtered_sample = {}
            for key in key_features:
                if key in sample:
                    filtered_sample[key] = sample[key]
                else:
                    print(f"⚠️ Warning: {key} not found in sample")
            filtered_samples.append(filtered_sample)

        samples = filtered_samples
        print(f"🔧 After feature filtering: kept {key_features}")
       
        return samples
    def load(self):
        return self.samples

