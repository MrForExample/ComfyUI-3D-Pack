import numpy as np
import torch
from CRM_T2I_V2.imagedream.camera_utils import get_camera_for_index
from CRM_T2I_V2.imagedream.ldm.util import set_seed, add_random_background
from CRM_T2I_V2.libs.base_utils import do_resize_content
from CRM_T2I_V2.imagedream.ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms as TF


class ImageDreamDiffusion:
    def __init__(
        self,
        model,
        device,
        dtype,
        mode,
        num_frames,
        camera_views,
        ref_position,
        random_background=False,
        offset_noise=False,
        resize_rate=1,
        image_size=256,
        seed=1234,
    ) -> None:
        assert mode in ["pixel", "local"]
        size = image_size
        self.seed = seed
        batch_size = max(4, num_frames)

        sampler = DDIMSampler(model, device=device)

        # pre-compute camera matrices
        camera = [get_camera_for_index(i).squeeze() for i in camera_views]
        camera[ref_position] = torch.zeros_like(camera[ref_position])  # set ref camera to zero
        camera = torch.stack(camera)
        camera = camera.repeat(batch_size // num_frames, 1).to(device)

        self.image_transform = TF.Compose(
            [
                TF.Resize((size, size)),
                TF.ToTensor(),
                TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.dtype = dtype
        self.ref_position = ref_position
        self.mode = mode
        self.random_background = random_background
        self.resize_rate = resize_rate
        self.num_frames = num_frames
        self.size = size
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.sampler = sampler
        self.camera = camera
        self.offset_noise = offset_noise
        self.pixel_control = (self.mode == "pixel")
        
    @staticmethod
    def get_ip_conditions(
        model,
        c_,
        uc_=None,
        ip=None,
        num_frames=7,
        pixel_control=True,
        transform=None,
        dtype=torch.float32,
        device="cuda"
    ):
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, dtype=dtype):
            if ip is not None:
                ip_embed = model.get_learned_image_conditioning(ip).to(
                    device
                )  # shape: torch.Size([num_ip, 257, 1280]) mean: 0.06, std: 0.53, min: -6.83, max: 11.12
                ip_ = ip_embed.repeat_interleave(num_frames, dim=0)
                c_["ip"] = ip_
                if uc_ is not None:
                    uc_["ip"] = torch.zeros_like(ip_)

            if pixel_control:
                ip = torch.stack([transform(img) for img in ip]).to(
                    device
                )  # shape: torch.Size([num_ip, 3, 256, 256]) mean: 0.33, std: 0.37, min: -1.00, max: 1.00
                ip_img = model.get_first_stage_encoding(
                    model.encode_first_stage(ip)
                )  # shape: torch.Size([num_ip, 4, 32, 32]) mean: 0.23, std: 0.77, min: -4.42, max: 3.55
                c_["ip_img"] = ip_img
                if uc_ is not None:
                    uc_["ip_img"] = torch.zeros_like(ip_img)
                
        return c_, uc_
       
    @staticmethod
    def get_base_conditions(
        model,
        prompt,
        negative_prompt,
        batch_size=7,
        num_ip=1,
        num_frames=7,
        dtype=torch.float32,
        device="cuda",
        camera=None
    ):
        """
        Get all condition and uncondition embeddings/parameters
        
        Reture conditions and uncondition dictionary
        """
        
        if type(prompt) != list:
            prompt = [prompt]
        if type(negative_prompt) != list:
            negative_prompt = [negative_prompt]
        
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, dtype=dtype):
            c = model.get_learned_conditioning(prompt).to(
                device
            )  # shape: torch.Size([1, 77, 1024]) mean: -0.17, std: 1.02, min: -7.50, max: 13.05
            uc = model.get_learned_conditioning(negative_prompt).to(
                device
            )
            c_ = {"context": c.repeat(batch_size, 1, 1)}  # batch_size
            uc_ = {"context": uc.repeat(batch_size, 1, 1)}

            if camera is not None:
                c_["camera"] = uc_["camera"] = (
                    camera.repeat(num_ip, 1)  # shape: torch.Size([5, 16]) mean: 0.11, std: 0.49, min: -1.00, max: 1.00
                )
                c_["num_frames"] = uc_["num_frames"] = num_frames
                
        return c_, uc_

    @staticmethod
    def i2i(
        model,
        image_size,
        prompt,
        negative_prompt,
        sampler,
        ip=None,
        step=20,
        scale=5.0,
        batch_size=8,
        ddim_eta=0.0,
        dtype=torch.float32,
        device="cuda",
        camera=None,
        num_frames=4,
        pixel_control=False,
        transform=None,
        offset_noise=False,
        additional_residuals=None
    ):
        """ The function supports additional image prompt.
        Args:
            model (_type_): the image dream model
            image_size (_type_): size of diffusion output (standard 256)
            prompt (_type_): text prompt for the image (prompt in type str)
            negative_prompt (_type_): negative text prompt (prompt in type str)
            sampler (_type_): imagedream.ldm.models.diffusion.ddim.DDIMSampler
            ip (Image, optional): the image prompt. Defaults to None.
            step (int, optional): _description_. Defaults to 20.
            scale (float, optional): _description_. Defaults to 7.5.
            batch_size (int, optional): _description_. Defaults to 8.
            ddim_eta (float, optional): _description_. Defaults to 0.0.
            dtype (_type_, optional): _description_. Defaults to torch.float32.
            device (str, optional): _description_. Defaults to "cuda".
            camera (_type_, optional): camera info in tensor, shape: torch.Size([5, 16]) mean: 0.11, std: 0.49, min: -1.00, max: 1.00
            num_frames (int, optional): _num of frames (views) to generate
            pixel_control: whether to use pixel conditioning. Defaults to False, True when using pixel mode
            transform: Compose(
                Resize(size=(256, 256), interpolation=bilinear, max_size=None, antialias=warn)
                ToTensor()
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            )
        """
        if type(ip) != list:
            ip = [ip]

        num_ip = len(ip)
        batch_size *= num_ip
        
        c_, uc_ = ImageDreamDiffusion.get_base_conditions(model, prompt, negative_prompt, batch_size, num_ip, num_frames, dtype, device, camera)
        c_, uc_ = ImageDreamDiffusion.get_ip_conditions(model, c_, uc_, ip, num_frames, pixel_control, transform, dtype, device)
        
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, dtype=dtype):
            
            shape = [4, image_size // 8, image_size // 8]  # [4, 32, 32]
            if offset_noise:
                ip_img = c_["ip_img"]
                ref_mean = ip_img.mean(dim=(-1, -2), keepdim=True).repeat_interleave(num_frames, dim=0)
                time_steps = torch.randint(model.num_timesteps - 1, model.num_timesteps, (batch_size,), device=device)
                x_T = model.q_sample(torch.ones([batch_size] + shape, device=device) * ref_mean, time_steps)

            kwargs = {}
            kwargs["additional_residuals"] = additional_residuals
            samples_ddim, _ = (
                sampler.sample(  # shape: torch.Size([5, 4, 32, 32]) mean: 0.29, std: 0.85, min: -3.38, max: 4.43
                    S=step,
                    conditioning=c_,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_,
                    eta=ddim_eta,
                    x_T=x_T if offset_noise else None,
                    **kwargs
                )
            )

            x_sample = model.decode_first_stage(samples_ddim)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = x_sample.permute(0, 2, 3, 1) # (N, H, W, 3) in [0, 1]

        return x_sample

    def diffuse(self, t, negative_prompt, ip, n_test=2):
        set_seed(self.seed)
        ip = do_resize_content(ip, self.resize_rate)
        if self.random_background:
            ip = add_random_background(ip)

        images = []
        for _ in range(n_test):
            img = self.i2i(
                self.model,
                self.size,
                t,
                negative_prompt,
                self.sampler,
                ip=ip,
                step=50,
                scale=5,
                batch_size=self.batch_size,
                ddim_eta=0.0,
                dtype=self.dtype,
                device=self.device,
                camera=self.camera,
                num_frames=self.num_frames,
                pixel_control=self.pixel_control,
                transform=self.image_transform,
                offset_noise=self.offset_noise,
            )
            img = np.concatenate(img, 1)
            img = np.concatenate((img, ip.resize((self.size, self.size))), axis=1)
            images.append(img)
        set_seed()  # unset random and numpy seed
        return images


class ImageDreamDiffusionStage2:
    def __init__(
        self,
        model,
        device,
        dtype,
        num_frames,
        camera_views,
        ref_position,
        random_background=False,
        offset_noise=False,
        resize_rate=1,
        mode="pixel",
        image_size=256,
        seed=1234,
    ) -> None:
        assert mode in ["pixel", "local"]

        size = image_size
        self.seed = seed
        batch_size = max(4, num_frames)

        neg_texts = "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear."
        uc = model.get_learned_conditioning([neg_texts]).to(device)
        sampler = DDIMSampler(model)

        # pre-compute camera matrices
        camera = [get_camera_for_index(i).squeeze() for i in camera_views]
        if ref_position is not None:
            camera[ref_position] = torch.zeros_like(camera[ref_position])  # set ref camera to zero
        camera = torch.stack(camera)
        camera = camera.repeat(batch_size // num_frames, 1).to(device)

        self.image_transform = TF.Compose(
            [
                TF.Resize((size, size)),
                TF.ToTensor(),
                TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dtype = dtype
        self.mode = mode
        self.ref_position = ref_position
        self.random_background = random_background
        self.resize_rate = resize_rate
        self.num_frames = num_frames
        self.size = size
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.sampler = sampler
        self.uc = uc
        self.camera = camera
        self.offset_noise = offset_noise

    @staticmethod
    def i2iStage2(
        model,
        image_size,
        prompt,
        uc,
        sampler,
        pixel_images,
        ip=None,
        step=20,
        scale=5.0,
        batch_size=8,
        ddim_eta=0.0,
        dtype=torch.float32,
        device="cuda",
        camera=None,
        num_frames=4,
        pixel_control=False,
        transform=None,
        offset_noise=False,
    ):
        ip_raw = ip
        if type(prompt) != list:
            prompt = [prompt]
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, dtype=dtype):
            c = model.get_learned_conditioning(prompt).to(
                device
            )  # shape: torch.Size([1, 77, 1024]) mean: -0.17, std: 1.02, min: -7.50, max: 13.05
            c_ = {"context": c.repeat(batch_size, 1, 1)}  # batch_size
            uc_ = {"context": uc.repeat(batch_size, 1, 1)}

            if camera is not None:
                c_["camera"] = uc_["camera"] = (
                    camera  # shape: torch.Size([5, 16]) mean: 0.11, std: 0.49, min: -1.00, max: 1.00
                )
                c_["num_frames"] = uc_["num_frames"] = num_frames

            if ip is not None:
                ip_embed = model.get_learned_image_conditioning(ip).to(
                    device
                )  # shape: torch.Size([1, 257, 1280]) mean: 0.06, std: 0.53, min: -6.83, max: 11.12
                ip_ = ip_embed.repeat(batch_size, 1, 1)
                c_["ip"] = ip_
                uc_["ip"] = torch.zeros_like(ip_)

            if pixel_control:
                assert camera is not None
                
            transed_pixel_images = torch.stack([transform(i).to(device) for i in pixel_images])
            latent_pixel_images = model.get_first_stage_encoding(model.encode_first_stage(transed_pixel_images))

            c_["pixel_images"] = latent_pixel_images
            uc_["pixel_images"] = torch.zeros_like(latent_pixel_images)

            shape = [4, image_size // 8, image_size // 8]  # [4, 32, 32]
            if offset_noise:
                ref = transform(ip_raw).to(device)
                ref_latent = model.get_first_stage_encoding(model.encode_first_stage(ref[None, :, :, :]))
                ref_mean = ref_latent.mean(dim=(-1, -2), keepdim=True)
                time_steps = torch.randint(model.num_timesteps - 1, model.num_timesteps, (batch_size,), device=device)
                x_T = model.q_sample(torch.ones([batch_size] + shape, device=device) * ref_mean, time_steps)

            samples_ddim, _ = (
                sampler.sample(  # shape: torch.Size([5, 4, 32, 32]) mean: 0.29, std: 0.85, min: -3.38, max: 4.43
                    S=step,
                    conditioning=c_,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_,
                    eta=ddim_eta,
                    x_T=x_T if offset_noise else None,
                )
            )
            x_sample = model.decode_first_stage(samples_ddim)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

        return list(x_sample.astype(np.uint8))

    @torch.no_grad()
    def diffuse(self, t, ip, pixel_images, n_test=2):
        set_seed(self.seed)
        ip = do_resize_content(ip, self.resize_rate)
        pixel_images = [do_resize_content(i, self.resize_rate) for i in pixel_images]

        if self.random_background:
            bg_color = np.random.rand() * 255
            ip = add_random_background(ip, bg_color)
            pixel_images = [add_random_background(i, bg_color) for i in pixel_images]

        images = []
        for _ in range(n_test):
            img = self.i2iStage2(
                self.model,
                self.size,
                t,
                self.uc,
                self.sampler,
                pixel_images=pixel_images,
                ip=ip,
                step=50,
                scale=5,
                batch_size=self.batch_size,
                ddim_eta=0.0,
                dtype=self.dtype,
                device=self.device,
                camera=self.camera,
                num_frames=self.num_frames,
                pixel_control=(self.mode == "pixel"),
                transform=self.image_transform,
                offset_noise=self.offset_noise,
            )
            img = np.concatenate(img, 1)
            img = np.concatenate(
                (img, ip.resize((self.size, self.size)), *[i.resize((self.size, self.size)) for i in pixel_images]),
                axis=1,
            )
            images.append(img)
        set_seed()  # unset random and numpy seed
        return images
