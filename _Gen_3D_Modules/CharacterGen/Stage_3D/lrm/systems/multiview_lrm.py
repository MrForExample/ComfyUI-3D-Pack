from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange

from CharacterGen.Stage_3D import lrm
from CharacterGen.Stage_3D.lrm.models.mesh import Mesh
from CharacterGen.Stage_3D.lrm.systems.base import BaseLossConfig, BaseSystem
from CharacterGen.Stage_3D.lrm.utils.ops import binary_cross_entropy, get_plucker_rays
from CharacterGen.Stage_3D.lrm.utils.typing import *
from CharacterGen.Stage_3D.lrm.models.lpips import LPIPS
from CharacterGen.Stage_3D.lrm.utils.misc import time_recorder as tr


@dataclass
class MultiviewLRMLossConfig(BaseLossConfig):
    lambda_mse: Any = 0.0
    lambda_mse_coarse: Any = 0.0
    lambda_smooth_l1: Any = 0.0
    lambda_smooth_l1_coarse: Any = 0.0
    lambda_lpips: Any = 0.0
    lambda_lpips_coarse: Any = 0.0
    lambda_mask: Any = 0.0
    lambda_mask_coarse: Any = 0.0


class MultiviewLRM(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        #loss: MultiviewLRMLossConfig = MultiviewLRMLossConfig()

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)

        material_cls: str = ""
        material: dict = field(default_factory=dict)

        background_cls: str = ""
        background: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        resume_ckpt_path: str = ""

    cfg: Config

    def configure(self):
        super().configure()
        self.image_tokenizer = lrm.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        if self.cfg.image_tokenizer.modulation:
            self.camera_embedder = lrm.find(self.cfg.camera_embedder_cls)(
                self.cfg.camera_embedder
            )
        self.tokenizer = lrm.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = lrm.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = lrm.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = lrm.find(self.cfg.decoder_cls)(self.cfg.decoder)
        self.material = lrm.find(self.cfg.material_cls)(self.cfg.material)
        self.background = lrm.find(self.cfg.background_cls)(self.cfg.background)
        self.renderer = lrm.find(self.cfg.renderer_cls)(
            self.cfg.renderer, self.decoder, self.material, self.background
        )
        
        self.exporter = lrm.find(self.cfg.exporter_cls)(
            self.cfg.exporter, self.renderer
        )
        
    def on_fit_start(self):
        super().on_fit_start()
        self.lpips_loss_fn = LPIPS()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # batch["rgb_cond"]: B, N_cond, H, W, 3
        # batch["rgb"]: B, N_render, H, W, 3
        # batch["c2w_cond"]: B, N_cond, 4, 4
        # for single image input (like LRM), N_cond = 1

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        # Camera modulation
        camera_embeds: Optional[Float[Tensor, "B Nv Cc"]]
        if self.cfg.image_tokenizer.modulation:
            camera_embeds = self.camera_embedder(**batch)
        else:
            camera_embeds = None

        tr.start("image tokenizer")
        input_image_tokens: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(
            rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_embeds,
            plucker_rays=rearrange(
                get_plucker_rays(batch["rays_o_cond"], batch["rays_d_cond"]),
                "B Nv H W C -> B Nv C H W",
            )
            if "rays_o_cond" in batch
            else None,
        )
        tr.end("image tokenizer")

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=n_input_views
        )

        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size)

        tr.start("backbone")
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )
        tr.end("backbone")

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def forward_renderer_nerf(
        self, batch: Dict[str, Any], scene_codes
    ) -> Dict[str, Any]:
        tr.start("render")
        render_out = self.renderer(scene_codes, **batch)
        tr.end("render")
        return render_out

    def training_step(self, batch, batch_idx):
        scene_codes = self(batch)
        out = self.forward_renderer_nerf(batch, scene_codes)

        loss = 0.0

        for suffix in ["", "_coarse"]:
            if not f"comp_rgb{suffix}" in out:
                continue

            comp_rgb: Float[Tensor, "B Nv H W 3"] = out["comp_rgb{}".format(suffix)]
            gt_rgb: Float[Tensor, "B Nv H W 3"] = batch["rgb"]

            self.log(f"train/comp_rgb_min{suffix}", comp_rgb.min())

            loss_mse = F.mse_loss(comp_rgb, gt_rgb, reduction="mean")
            self.log(f"train/loss_mse{suffix}", loss_mse)
            loss += loss_mse * self.C(self.cfg.loss[f"lambda_mse{suffix}"])

            loss_smooth_l1 = F.smooth_l1_loss(
                comp_rgb, gt_rgb, beta=0.1, reduction="mean"
            )
            self.log(f"train/loss_smooth_l1{suffix}", loss_smooth_l1)
            loss += loss_smooth_l1 * self.C(self.cfg.loss[f"lambda_smooth_l1{suffix}"])

            if self.C(self.cfg.loss[f"lambda_lpips{suffix}"]) > 0:
                loss_lpips = self.lpips_loss_fn(
                    rearrange(comp_rgb, "B Nv H W C -> (B Nv) C H W"),
                    rearrange(gt_rgb, "B Nv H W C -> (B Nv) C H W"),
                    input_range=(0, 1),
                ).mean()
                self.log(f"train/loss_lpips{suffix}", loss_lpips)
                loss += loss_lpips * self.C(self.cfg.loss[f"lambda_lpips{suffix}"])

            loss_mask = binary_cross_entropy(
                out[f"opacity{suffix}"].clamp(1e-5, 1 - 1e-5), batch["mask"]
            )
            self.log(f"train/loss_mask{suffix}", loss_mask)
            loss += loss_mask * self.C(self.cfg.loss[f"lambda_mask{suffix}"])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(
            batch,
            out,
            extra=f"m{loss_mse:.2f}_l{loss_smooth_l1:.2f}_p{loss_lpips:.2f}_ma{loss_mask:.2f}",
        )

        return {"loss": loss}

    def get_input_visualizations(self, batch):
        return [
            {
                "type": "rgb",
                "img": rearrange(batch["rgb_cond"], "B N H W C -> (B H) (N W) C"),
                "kwargs": {"data_format": "HWC"},
            }
        ]

    def get_output_visualizations(self, batch, outputs):
        out = outputs
        images = []
        if "rgb" in batch:
            images += [
                {
                    "type": "rgb",
                    "img": rearrange(batch["rgb"], "B N H W C -> (B H) (N W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "grayscale",
                    "img": rearrange(batch["mask"], "B N H W C -> (B H) (N W) C")[
                        ..., 0
                    ],
                    "kwargs": {"cmap": None, "data_range": None},
                },
            ]
        for suffix in ["", "_coarse"]:
            if not f"comp_rgb{suffix}" in out:
                continue
            images += [
                {
                    "type": "rgb",
                    "img": rearrange(
                        out[f"comp_rgb{suffix}"], "B N H W C -> (B H) (N W) C"
                    ),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "grayscale",
                    "img": rearrange(
                        out[f"opacity{suffix}"], "B N H W C -> (B H) (N W) C"
                    )[..., 0],
                    "kwargs": {"cmap": None, "data_range": None},
                },
                {
                    "type": "grayscale",
                    "img": rearrange(
                        out[f"depth{suffix}"], "B N H W C -> (B H) (N W) C"
                    )[..., 0],
                    "kwargs": {"cmap": None, "data_range": None},
                },
            ]
        return images

    # def check_train(self, batch, outputs, **kwargs):
    #     self.on_check_train(batch, outputs, **kwargs)

    def on_check_train(self, batch, outputs, extra=""):
        self.save_image_grid(
            f"it{self.true_global_step}-train.jpg",
            self.get_output_visualizations(batch, outputs),
            name="train_step_output",
            step=self.true_global_step,
        )
        # self.save_image_grid(
        #     f"debug/it{self.true_global_step}-{self.global_rank}-{extra}.jpg",
        #     self.get_output_visualizations(batch, outputs),
        #     name="train_step_output",
        #     step=self.true_global_step,
        # )
        # self.save_json(
        #     f"debug_list/it{self.true_global_step}-{self.global_rank}-ids.json",
        #     batch["scene_id"],
        # )

    def validation_step(self, batch, batch_idx):
        scene_codes = self(batch)
        out = self.forward_renderer_nerf(batch, scene_codes)
        if (
            self.cfg.check_val_limit_rank > 0
            and self.global_rank < self.cfg.check_val_limit_rank
        ):
            self.save_image_grid(
                f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}-input.jpg",
                self.get_input_visualizations(batch),
                name=f"validation_step_input_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}.jpg",
                self.get_output_visualizations(batch, out),
                name=f"validation_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx):
        # not saved to wandb
        scene_codes = self(batch)
        out = self.forward_renderer_nerf(batch, scene_codes)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                self.save_image_grid(
                    f"it{self.true_global_step}-test/{batch['index'][b]}-input.jpg",
                    [
                        {
                            "type": "rgb",
                            "img": rearrange(
                                batch["rgb_cond"][b], "N H W C -> H (N W) C"
                            ),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ],
                )
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][b]}/{batch['view_index'][b,0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][b][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "grayscale",
                        "img": out["depth"][b][0, ..., 0],
                        "kwargs": {"cmap": None, "data_range": None},
                    },
                ],
            )

    def on_test_end(self):
        if self.global_rank == 0:
            self.save_img_sequences(
                f"it{self.true_global_step}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
            )
            
    def get_geometry(self, scene_code: torch.Tensor) -> Mesh:
        tr.start("Surface extraction")
        mesh: Mesh = self.renderer.isosurface(scene_code)
        tr.end("Surface extraction")
        return mesh
