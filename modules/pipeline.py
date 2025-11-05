import abc
import os

from modules import clip

import vae


class VAEWrapper(abc.ABC):
    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.vae, name)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, latents):
        return NotImplementedError


class WanxVAEWrapper(VAEWrapper):
    def __init__(self, vae, clip):
        super(WanxVAEWrapper, self).__init__()
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.clip = clip
        if clip is not None:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def encode(self, x, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        x = self.vae.encode(
            x, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )  # already scaled
        return x  # torch.stack(x, dim=0)

    def clip_img(self, x):
        x = self.clip(x)
        return x

    def decode(
        self, latents, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        videos = self.vae.decode(
            latents,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return videos  # self.vae.decode(videos, dim=0) # already scaled

    def to(self, device, dtype):
        # Move VAE module to the requested device/dtype.
        self.vae = self.vae.to(device, dtype)

        # Move CLIP encoder as well when available.
        if self.clip is not None:
            self.clip = self.clip.to(device, dtype)

        return self


def get_wanx_vae_wrapper(model_path, weight_dtype):
    vae_model = vae.WanVAE(
        pretrained_path=os.path.join(model_path, "Wan2.1_VAE.pth")
    ).to(weight_dtype)

    clip_model = clip.CLIPModel(
        checkpoint_path=os.path.join(
            model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        ),
        tokenizer_path=os.path.join(model_path, "xlm-roberta-large"),
    )
    return WanxVAEWrapper(vae_model, clip_model)
