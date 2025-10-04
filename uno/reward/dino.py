import json
import math
import os
from typing import Literal, TypeAlias

import torch
from accelerate import PartialState
from torchvision import transforms
from tqdm.auto import tqdm
import PIL.Image
from PIL import Image
from transformers import BitImageProcessor, Dinov2Model
import numpy as np

ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32

class Dinov2Score:
    # NOTE: noqa, in version 1, the performance of the official repository and HuggingFace is inconsistent.
    def __init__(
        self,
        model_or_name_path,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = Dinov2Model.from_pretrained(model_or_name_path, torch_dtype=torch_dtype).to(device)
        self.model.eval()
        self.processor = BitImageProcessor.from_pretrained(model_or_name_path)
        print("********DINO MODEL INIT ************")

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model(inputs["pixel_values"].to(self.device, dtype=self.dtype)).last_hidden_state[:, 0, :]
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)
        return image_features

    def dino_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."

        images1_features = self.get_image_features(images1, norm=True)
        images2_features = self.get_image_features(images2, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (images1_features * images2_features).sum(axis=-1)
        return score.sum(0).float(), len(images1)

def single_eval_dino_score(
    image_ref_dirs: list | str | ImageType,
    image_dirs: list | str | ImageType,
    dino_score: Dinov2Score | None = None,
) -> float:
    if dino_score is None:
        dino_score = Dinov2Score()
    if isinstance(image_ref_dirs, str):
        image_ref_dirs = [image_ref_dirs]

    image_refs = []
    images = []

    for image_ref_dir in image_ref_dirs:
        image_refs.append(PIL.Image.open(image_ref_dir).convert("RGB"))

    if isinstance(image_dirs, ImageType):
        images.append(image_dirs)
    else:
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        for image_dir in image_dirs:
            images.append(PIL.Image.open(image_dir).convert("RGB"))

    score = dino_score.dino_score(image_refs, images)[0]

    return score

if __name__ == "__main__":

    image1_ref = "/home/zw.hzw/model_list/dinov3-main/val_images/dog3_ref.png"
    image1_gen = "/home/zw.hzw/model_list/dinov3-main/val_images/dog3_gen.png"

    image2_ref = "/home/zw.hzw/model_list/dinov3-main/val_images/candle_ref.png"
    image2_gen = "/home/zw.hzw/model_list/dinov3-main/val_images/candle_gen.png"

    device = "cuda:0"
    # print("decoded_image type:", type(img_obj))
    dino_score = Dinov2Score(model_or_name_path = "/home/zw.hzw/checkpoints/dinov2-base")
    score = single_eval_dino_score(image1_ref, image1_gen, dino_score)
    print(score)
    score = single_eval_dino_score(image2_ref, image2_gen, dino_score)
    print(score)