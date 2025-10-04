import json
import math
import os
from typing import Literal, TypeAlias

import torch
from accelerate import PartialState
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
import PIL.Image
from PIL import Image
import numpy as np

ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32

class CLIPScore:

    def __init__(
        self,
        model_or_name_path,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = CLIPModel.from_pretrained(model_or_name_path, torch_dtype=torch_dtype).to(device)
        print("-------------CLIP_MODEL INIT ALREADY----------")
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_or_name_path)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)

    @torch.no_grad()
    def get_text_features(self, text: str | list[str], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
        inputs = self.processor(text=text, padding=True, return_tensors="pt", max_length=77, truncation=True)
        text_features = self.model.get_text_features(
            inputs["input_ids"].to(self.device),
            inputs["attention_mask"].to(self.device),
        )
        if norm:
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device, dtype=self.dtype))
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def clipi_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
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

    @torch.no_grad()
    def clipt_score(self, texts: str | list[str], images: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(texts, list):
            texts = [texts]
        if not isinstance(images, list):
            images = [images]
        assert len(texts) == len(images), f"Number of texts ({len(texts)}) and images {(len(images))} should be same."

        texts_features = self.get_text_features(texts, norm=True)
        images_features = self.get_image_features(images, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (texts_features * images_features).sum(axis=-1)
        # print("SCORE", score)
        return score.sum(0).float(), len(texts)

def single_eval_clipt_score(
    texts: list | str,
    image_dirs: list | str | ImageType,
    clip_score: CLIPScore | None = None,
) -> float:
    if clip_score is None:
        clip_score = CLIPScore()
    if isinstance(texts, str):
        texts = [texts]

    if isinstance(image_dirs, ImageType):
        score = clip_score.clipt_score(texts, image_dirs)[0]
    else:
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        images = []
        for image_dir in image_dirs:
            images.append(PIL.Image.open(image_dir).convert("RGB"))
        score = clip_score.clipt_score(texts, images)[0]

    return score

def single_eval_clipi_score(
    image_ref_dirs: list | str | ImageType,
    image_dirs: list | str | ImageType,
    clip_score: CLIPScore | None = None,
) -> float:
    if clip_score is None:
        clip_score = CLIPScore()

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

    score = clip_score.clipi_score(image_refs, images)[0]
    return score

if __name__ == "__main__":
    texts = "two dolls with braided hair, hats, and pink accents. two dolls with braided hair, hats, and pink accents two dolls with braided hair, hats, and pink accents two dolls with braided hair, hats, and pink accents two dolls with braided hair, hats, and pink accents two dolls with braided hair, hats, and pink accents two dolls with braided hair, hats, and pink accents"
    image = "/data/oss_bucket_0/ziwei/datasets/syncd/36611/1.png"
    img_obj = Image.open("/data/oss_bucket_0/ziwei/datasets/syncd/36611/1.png").convert("RGB")
    print("decoded_image type:", type(img_obj))
    device = "cuda:0"
    clip_score = CLIPScore(model_or_name_path = "/home/zw.hzw/checkpoints/clip-vit-large-patch14")
    score = single_eval_clipt_score(texts, img_obj, clip_score)
    score_clipi = single_eval_clipi_score("/data/oss_bucket_0/ziwei/datasets/syncd/36611/0.png", img_obj, clip_score)
    print(score)