from dataclasses import dataclass, field
import numpy as np
import torch
from skimage import measure
from einops import repeat, rearrange

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.ops import generate_dense_grid_points
from craftsman.utils.typing import *
from craftsman.utils.misc import get_rank


@craftsman.register("shape-autoencoder-system")
class ShapeAutoEncoderSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        sample_posterior: bool = True

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "xyz" in batch:
            if "sdf" in batch:
                bs = batch["sdf"].shape[0]
                rand_points = torch.cat([batch["xyz"].view(bs, -1, 3), batch["patch_xyz"].view(bs, -1, 3)], dim=1)
                target = torch.cat([batch["sdf"].view(bs, -1, 1), batch["patch_sdf"].view(bs, -1, 1)], dim=1).squeeze(-1)
                criteria = torch.nn.MSELoss()
            elif "occupancy" in batch:
                bs = batch["occupancy"].shape[0]
                rand_points = torch.cat([batch["xyz"].view(bs, -1, 3), batch["patch_xyz"].view(bs, -1, 3)], dim=1)
                target = torch.cat([batch["occupancy"].view(bs, -1, 1), batch["patch_occupancy"].view(bs, -1, 1)], dim=1).squeeze(-1)
                criteria = torch.nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError
        else:
            rand_points = batch["rand_points"]
            if "sdf" in batch:
                target = batch["sdf"]
                criteria = torch.nn.MSELoss()
            elif "occupancies" in batch:
                target = batch["occupancies"]
                criteria = torch.nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError

        # forward pass
        _, latents, posterior, logits = self.shape_model(
            batch["surface"][..., :3 + self.cfg.shape_model.point_feats], 
            rand_points, 
            sample_posterior=self.cfg.sample_posterior
        )

        if self.cfg.sample_posterior:
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

            return {
                "loss_logits": criteria(logits, target).mean(),
                "loss_kl": loss_kl,
                "logits": logits,
                "target": target,
                "latents": latents,
            }
        else:
            return {
                "loss_logits": criteria(logits, target).mean(),
                "latents": latents,
                "logits": logits,
            }

    def training_step(self, batch, batch_idx):
        """
        Description:

        Args:
            batch:
            batch_idx:
        Returns:
            loss:
        """
        out = self(batch)

        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        out = self(batch)
        # self.save_state_dict("latest-weights", self.state_dict())

        mesh_v_f, has_surface = self.shape_model.extract_geometry(out["latents"])
        self.save_mesh(
            f"it{self.true_global_step}/{batch['uid'][0]}.obj",
            mesh_v_f[0][0], mesh_v_f[0][1]
        )

        threshold = 0
        outputs = out["logits"]
        labels = out["target"]
        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        self.log("val/accuracy", accuracy)
        self.log("val/iou", iou)

        torch.cuda.empty_cache()

        return {"val/loss": out["loss_logits"], "val/accuracy": accuracy, "val/iou": iou}


    def on_validation_epoch_end(self):
        pass