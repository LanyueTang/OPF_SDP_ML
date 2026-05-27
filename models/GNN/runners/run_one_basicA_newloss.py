import os, sys, yaml, torch
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import random
CONFIG_PATH = Path(os.environ.get(
    "GNN_CONFIG_PATH",
    str(PROJECT_ROOT / "configs" / "0522_debug.yaml")
))

try:
    from features.loads_phys_topo import LoadsPhysTopo
except ModuleNotFoundError:
    from features.loads_phys_topo import LoadsPhysTopo

from gcn_utils.seed import set_seed
import pickle
import torch 
import numpy as np 
from registries import build
from torch.utils.data import DataLoader
from data_loader.datasets.dataset_opf import OPFGraphDataset, make_collate_fn

from data_loader import reader  
from model_varients import gcn_new
from trainers import task
from trainers import loss 
from gcn_utils.normalize import GlobalNormalizer, normalize_inplace
from pathlib import Path
import torch

def build_run_paths(cfg):
    run_name = cfg["experiment"]["run_name"]
    root_dir = Path(cfg["experiment"]["root_dir"])
    run_dir = root_dir / "result" / run_name
    ckpt_dir = root_dir / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "log").mkdir(parents=True, exist_ok=True)

    cfg["evaluator"]["output_dir"] = str(run_dir)
    cfg["logger"]["log_dir"] = str(run_dir / "log")

    cfg["checkpoint"]["dir"] = str(ckpt_dir)
    cfg["checkpoint"]["save_full_model_path"] = str(ckpt_dir / "full_model.pth")
    cfg["checkpoint"]["save_training_path"] = str(ckpt_dir / "training_latest.pth")

    if cfg["checkpoint"].get("resume_training_path") is None:
        cfg["checkpoint"]["resume_training_path"] = cfg["checkpoint"]["save_training_path"]

    if cfg["checkpoint"].get("eval_model_path") is None:
        cfg["checkpoint"]["eval_model_path"] = cfg["checkpoint"]["save_full_model_path"]

    return cfg

def infer_class_counts_from_list(samples, num_classes: int):
    counts = torch.zeros(num_classes, dtype=torch.long)
    key_candidates = ("y_cls", "label", "y")
    for sample in samples:
        if isinstance(sample, dict):
            y = None
            for k in key_candidates:
                if k in sample:
                    y = sample[k]
                    break
        else:
            y = sample  

        if y is None:
            continue

        y = torch.as_tensor(y)

        if y.ndim == 0:
            idx = y.unsqueeze(0).to(torch.long)
        elif y.ndim == 1:
            idx = y.to(torch.long)
        elif y.ndim == 2:

            idx = y.argmax(dim=1).to(torch.long)
        else:
            raise ValueError(f"Unsupported label shape: {tuple(y.shape)}")

        idx = idx.clamp(min=0, max=num_classes - 1)

        bc = torch.bincount(idx, minlength=num_classes)
        counts[:len(bc)] += bc.to(torch.long)


    return counts.tolist()

#checkpoint utils
def save_full_model_checkpoint(path, model, cfg=None, extra=None):
    """
    Save all model weights.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
    }
    if cfg is not None:
        ckpt["cfg"] = cfg
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, str(path))
    print(f"[CKPT] Saved full model checkpoint to {path}")
def load_full_model_checkpoint(path, model, device="cpu", strict=True):
    """
    Load all model weights into an already-built model.
    """
    ckpt = torch.load(path, map_location=device)

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    print(f"[CKPT] Loaded full model checkpoint from {path}")
    return model

def save_training_checkpoint(path, model, task, epoch, cfg=None, extra=None):
    """
    Save training state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
    }

    if hasattr(task, "optimizer"):
        ckpt["optimizer_state"] = task.optimizer.state_dict()

    if cfg is not None:
        ckpt["cfg"] = cfg

    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, str(path))
    print(f"[CKPT] Saved training checkpoint to {path}")
    
def load_training_checkpoint(path, model, task, device="cpu", strict=True):
    """
    Load model + optimizer + epoch.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.to(device)
    if "optimizer_state" in ckpt and hasattr(task, "optimizer"):
        task.optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = ckpt.get("epoch", -1) + 1

    print(f"[CKPT] Resume training from {path}")
    print(f"[CKPT] Start epoch = {start_epoch}")

    return start_epoch

def infer_in_dim_from_ds(ds):
    raw0 = ds.samples[0]                 
    if ds.build_features is not None:    
        X0 = ds.build_features(raw0)
    else:
        X0 = raw0["X"]
    return X0.shape[-1]

def split_by_ratio(samples, train=0.6, val=0.2, test=0.2, seed=42, shuffle=True):
    assert abs(train + val + test - 1.0) < 1e-6
    idx = list(range(len(samples)))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
    n = len(samples)
    i_tr = int(train * n)
    i_va = int((train + val) * n)
    tr_idx, va_idx, te_idx = idx[:i_tr], idx[i_tr:i_va], idx[i_va:]
    get = lambda ids: [samples[i] for i in ids]
    return get(tr_idx), get(va_idx), get(te_idx)


def split_by_group(samples, group_key="instance_id", train=0.6, val=0.3, test=0.1, seed=42, shuffle=True):
    assert abs(train + val + test - 1.0) < 1e-6
    groups = {}
    for i, s in enumerate(samples):
        if isinstance(s, dict) and group_key in s:
            gid = int(torch.as_tensor(s[group_key]).item())
        else:
            gid = i
        groups.setdefault(gid, []).append(i)

    gids = list(groups.keys())
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(gids)

    n = len(gids)
    i_tr = int(train * n)
    i_va = int((train + val) * n)
    g_tr, g_va, g_te = gids[:i_tr], gids[i_tr:i_va], gids[i_va:]

    def collect(gid_list):
        out = []
        for gid in gid_list:
            out.extend(groups[gid])
        return [samples[j] for j in out]

    return collect(g_tr), collect(g_va), collect(g_te)

def main():
    cfg = yaml.safe_load(open(CONFIG_PATH, "r"))
    set_seed(cfg.get("seed", 42))
    cfg = build_run_paths(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    data_cfg = dict(cfg.get("data", {}))
    reader_name = data_cfg.pop("reader")
    reader = build("reader", reader_name, **data_cfg)
    samples  = reader.load()
    norm_style     = cfg["data"].get("norm", None)                   
    norm = normalize_inplace(samples, mode=norm_style, key="node_load") if norm_style else None
    train_ratio = 0.6
    val_ratio   = 0.3
    test_ratio  = 1.0 - train_ratio - val_ratio
    has_group = len(samples) > 0 and isinstance(samples[0], dict) and ("instance_id" in samples[0])
    if has_group:
        train_s, val_s, test_s = split_by_group(
            samples,
            group_key="instance_id",
            train=train_ratio,
            val=val_ratio,
            test=test_ratio,
            seed=cfg["seed"],
            shuffle=True,
        )
        print("[INFO] Using grouped split by instance_id")
    else:
        train_s, val_s, test_s = split_by_ratio(samples, train_ratio, val_ratio, test_ratio, seed=cfg["seed"], shuffle=True)
    pipeline = LoadsPhysTopo()
    train_ds = OPFGraphDataset(train_s,  pipeline.node_features)
    val_ds   = OPFGraphDataset(val_s,   pipeline.node_features)
    test_ds  = OPFGraphDataset(test_s,  pipeline.node_features)
    train_ds.collate_fn = make_collate_fn(train_ds)
    val_ds.collate_fn   = make_collate_fn(val_ds)   
    test_ds.collate_fn  = make_collate_fn(test_ds)

    print(f"[INFO] Dataset split:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=True,
                              collate_fn=getattr(train_ds, "collate_fn", None))
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=True,
                              collate_fn=getattr(val_ds, "collate_fn", None))
    test_loader   = DataLoader(test_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=True,
                              collate_fn=getattr(test_ds, "collate_fn", None))
    in_dim = infer_in_dim_from_ds(train_ds)
    print(f"[INFO] Input feature dimension: {in_dim}")
    # model / loss / task
    model_cfg = dict(cfg.get("model", {}))
    model_name = model_cfg.pop("name")
    model = build("model", model_name, in_dim=in_dim, **model_cfg)
    num_classes = cfg["model"].get("out_array_dim", 15)
    class_counts_auto = infer_class_counts_from_list(train_s, num_classes)

    print(f"[INFO] Class counts in training set (C={num_classes}): {class_counts_auto}")

    #loss_fn = build("loss", "arr_cls") 
    if cfg["loss"]["name"] == "arr_dec_softmin":
        loss_fn = build("loss", cfg["loss"]["name"],
                        tau=cfg["loss"].get("tau", 1.0),
                        center=cfg["loss"].get("center", True),
                        lam_reg=cfg["loss"].get("lam_reg", 1),
                        reg_mode=cfg["loss"].get("reg_mode", "mse"),
                        reduction=cfg["loss"].get("reduction", "mean"))  
    if cfg["loss"]["name"] == "arr_cls_cbce":
        loss_fn = build("loss", cfg["loss"]["name"],
                        w_arr=cfg["loss"].get("w_arr", 1.0),
                        w_cls=cfg["loss"].get("w_cls", 1.0),
                        tau=cfg["loss"].get("tau", 1),
                        beta=cfg["loss"].get("beta", 0.99),
                        class_counts=class_counts_auto,
                        normalize_weights=cfg["loss"].get("normalize_weights", True))
    else:
        loss_fn = build("loss", cfg["loss"]["name"])
        
    

    task = build("task", cfg["task"]["name"],
                model=model, 
                loss_fn=loss_fn,
                device=device,  
                lr=cfg.get("train", {}).get("lr", 1e-3), 
                log_dir=cfg.get("logger", {}).get("log_dir"))
    
    
    
    ckpt_cfg = cfg.get("checkpoint", {})
    epochs = cfg.get("train", {}).get("epochs", 5)
    start_epoch = 0
    resume = ckpt_cfg.get("resume", False)
    resume_path = ckpt_cfg.get("resume_training_path", None)
    save_training_path = ckpt_cfg.get("save_training_path", None)
    save_full_model_path = ckpt_cfg.get("save_full_model_path", None)
    save_every = ckpt_cfg.get("save_every", 5)
    eval_model_path = ckpt_cfg.get("eval_model_path", None)
    if resume:
        if resume_path is None:
            raise ValueError("resume=True but resume_path is None")
        start_epoch = load_training_checkpoint(
            resume_path,
            model,
            task,
            device
        )

    for ep in range(start_epoch, epochs):
        tr_loss = task.train_one_epoch(train_loader, ep)
        val_loss = task.validate(val_loader, ep)

        print(f"Epoch {ep} | loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

        if save_training_path is not None:
            if (ep + 1) % save_every == 0 or ep == epochs - 1:
                save_training_checkpoint(
                    save_training_path,
                    model,
                    task,
                    ep,
                    cfg=cfg,
                    extra={
                        "train_loss": tr_loss,
                        "val_loss": val_loss,
                    },
                )

    if save_full_model_path is not None:
        save_full_model_checkpoint(
            save_full_model_path,
            model,
            cfg=cfg,
            extra={
                "last_epoch": epochs - 1,
            },
        )
    
    #model test and eval
    task.plot_loss_curves(cfg)
    task.save_loss_data(cfg)
    test_metrics = task.test_with_evaluator(test_loader, cfg)
    print("[TEST]", test_metrics)

if __name__ == "__main__":
    main()

