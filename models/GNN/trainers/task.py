import os, sys, yaml, torch
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import matplotlib.pyplot as plt
import numpy as np
from registries import register
from registries import build
import evaluation.evaluators
from gcn_utils.model_logger import ModelLogger
from torch.optim import AdamW


class BaseTask:
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None,
                 return_all_H=False, track_gsmooth=False):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.return_all_H = bool(return_all_H)
        self.track_gsmooth = bool(track_gsmooth)
        self.Gsmooth_stds = []
        self.train_losses = []
        self.val_losses = []
        self.log_dir = log_dir
        self._init_optimizer(lr)

        if log_dir:
            self.logger = ModelLogger(self.model, self.opt, log_dir=log_dir)
            self.logger.record_model()
        else:
            self.logger = None

    def _init_optimizer(self, lr):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        v[kk] = vv.to(self.device)
        return batch

    def _log_layer_std(self, out):
        if not self.return_all_H:
            return
        H_list = out["H_list"]
        std_list = []
        for l, H in enumerate(H_list):
            std = H.std(dim=1).mean().item()
            print(f"Layer {l}: mean node feature std = {std:.6f}")
            std_list.append(std)
        if self.track_gsmooth:
            self.Gsmooth_stds.append(std_list)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total = 0.0
        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            self.opt.zero_grad()
            if self.return_all_H:
                out = self.model(batch, return_all_H=True)
                #self._log_layer_std(out)
            else:
                out = self.model(batch)
            loss = self.loss_fn(out, batch)
            loss.backward()
            if self.logger:
                self.logger.pre_step()

            self.opt.step()

            if self.logger:
                self.logger.log_gradients(epoch, step)

            total += loss.item()

        avg_loss = total / max(1, len(loader))
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(batch)
            loss = self.loss_fn(out, batch)
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        self.val_losses.append(avg_loss)
        return avg_loss

    def plot_loss_curves(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        if not self.train_losses:
            print("No training losses to plot")
            return

        plt.figure(figsize=(10, 6))

        epochs = range(1, len(self.train_losses) + 1)

        plt.plot(epochs, self.train_losses, "b-", label="Training Loss", linewidth=2)
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            plt.plot(val_epochs, self.val_losses, "r-", label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if len(self.train_losses) <= 20:
            for i, loss in enumerate(self.train_losses):
                plt.annotate(
                    f"{loss:.3f}",
                    (i + 1, loss),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

        plt.tight_layout()

        save_path = os.path.join(save_path, "loss_curves.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Loss curves saved to: {save_path}")

        print("\n📊 Loss Statistics:")
        print(f"Final Training Loss: {self.train_losses[-1]:.6f}")
        if self.val_losses:
            print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
            print(
                f"Best Validation Loss: {min(self.val_losses):.6f} "
                f"(Epoch {self.val_losses.index(min(self.val_losses)) + 1})"
            )

    def _extra_loss_columns(self):
        return []

    def save_loss_data(self, cfg):
        save_path = cfg["evaluator"]["output_dir"]
        save_path = os.path.join(save_path, "loss_data.csv")
        import csv

        extra_cols = self._extra_loss_columns()
        header = ["Epoch", "Train_Loss", "Val_Loss"] + [name for name, _ in extra_cols]

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            max_len = max(len(self.train_losses), len(self.val_losses))
            for i in range(max_len):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else ""
                val_loss = self.val_losses[i] if i < len(self.val_losses) else ""
                row = [i + 1, train_loss, val_loss]
                for _, values in extra_cols:
                    row.append(values[i] if i < len(values) else "")
                writer.writerow(row)

        print(f"Loss data saved to: {save_path}")

    @torch.no_grad()
    def test_with_evaluator(self, loader, cfg):
        name = cfg["evaluator"]["name"]
        output_dir = cfg["evaluator"]["output_dir"]
        num_classes = cfg["model"].get("out_array_dim", 15)
        evaluator = build("evaluator", name, output_dir=output_dir, num_classes=num_classes)
        return evaluator.run(self.model, loader, device=self.device, output_dir=output_dir)


@register("task", "opf_basicA_task")
class OPFBasicTask(BaseTask):
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            device=device,
            lr=lr,
            log_dir=log_dir,
            return_all_H=True,
            track_gsmooth=False,
        )


@register("task", "opf_basicA_task_Gsmooth")
class OPFBasicTask(BaseTask):
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        if hasattr(self.model, "head_array") and self.model.head_array is not None:
            print("🔒 Freezing head_array parameters...")
            for p in self.model.head_array.parameters():
                p.requires_grad = False
        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )
        self.train_losses = []
        self.val_losses = []
        self.Gsmooth_stds = []
        self.log_dir = log_dir
        self.return_all_H = True
        self.track_gsmooth = True

        if log_dir:
            self.logger = ModelLogger(self.model, self.opt, log_dir=log_dir)
            self.logger.record_model()
        else:
            self.logger = None

    def _extra_loss_columns(self):
        return [("Gsmooth_Stds", self.Gsmooth_stds)]


@register("task", "opf_basicA_dirl_task")
class OPFBasicTask(BaseTask):
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        gcn_backbone_params = []
        head_params = []

        for name, p in model.named_parameters():
            if "gcns" in name:
                gcn_backbone_params.append(p)
            else:
                head_params.append(p)

        backbone_lr = 3 * lr
        head_lr = 1 * lr

        self.opt = AdamW(
            [
                {"params": gcn_backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=1e-4,
        )

        self.train_losses = []
        self.val_losses = []
        self.Gsmooth_stds = []
        self.log_dir = log_dir
        self.return_all_H = False
        self.track_gsmooth = False

        if log_dir:
            self.logger = ModelLogger(self.model, self.opt, log_dir=log_dir)
            self.logger.record_model()
        else:
            self.logger = None


@register("task", "opf_globalB_task")
class OPFBasicTask(BaseTask):
    def __init__(self, model, loss_fn, device="cpu", lr=1e-3, log_dir=None):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            device=device,
            lr=lr,
            log_dir=log_dir,
            return_all_H=False,
            track_gsmooth=False,
        )
