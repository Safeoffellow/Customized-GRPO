# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader

def bucket_images(images: list[torch.Tensor], resolution: int = 512):
    bucket_override=[
        # h    w
        (256, 768),
        (320, 768),
        (320, 704),
        (384, 640),
        (448, 576),
        (512, 512),
        (576, 448),
        (640, 384),
        (704, 320),
        (768, 320),
        (768, 256)
    ]
    bucket_override = [(int(h / 512 * resolution), int(w / 512 * resolution)) for h, w in bucket_override]
    bucket_override = [(h // 16 * 16, w // 16 * 16) for h, w in bucket_override]

    aspect_ratios = [image.shape[-2] / image.shape[-1] for image in images]
    mean_aspect_ratio = np.mean(aspect_ratios)
    
    new_h, new_w = bucket_override[0]
    min_aspect_diff = np.abs(new_h / new_w - mean_aspect_ratio)
    for h, w in bucket_override:
        aspect_diff = np.abs(h / w - mean_aspect_ratio)
        if aspect_diff < min_aspect_diff:
            min_aspect_diff = aspect_diff
            new_h, new_w = h, w
    
    images = [TVF.resize(image, (new_h, new_w)) for image in images]
    images = torch.stack(images, dim=0)
    return images

class FluxPairedDatasetV2(Dataset):
    def __init__(self, json_file: str, resolution: int, resolution_ref: int | None = None):
        super().__init__()
        self.json_file = json_file
        self.resolution = resolution
        self.resolution_ref = resolution_ref if resolution_ref is not None else resolution
        self.image_root = os.path.dirname(json_file)

        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]
        image_paths = [data_dict["image_path"]] if "image_path" in data_dict else data_dict["image_paths"]
        txt = data_dict["prompt"]
        image_tgt_path = data_dict.get("image_tgt_path", None)
        # image_tgt_path = data_dict.get("image_paths", None)[0]  # TODO: for debugging delete it when release paired data pipeline
        ref_imgs = [
            Image.open(os.path.join(self.image_root, path)).convert("RGB")
            for path in image_paths
        ]
        ref_imgs = [self.transform(img) for img in ref_imgs]
        img = None
        if image_tgt_path is not None:
            img = Image.open(os.path.join(self.image_root, image_tgt_path)).convert("RGB")
            img = self.transform(img)

        return {
            "img": img,
            "txt": txt,
            "ref_imgs": ref_imgs,
        }

    def __len__(self):
        return len(self.data_dicts)
    
    def collate_fn(self, batch):
        img = [data["img"] for data in batch]
        txt = [data["txt"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        assert all([len(ref_imgs[0]) == len(ref_imgs[i]) for i in range(len(ref_imgs))])

        n_ref = len(ref_imgs[0])

        img = bucket_images(img, self.resolution)
        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = bucket_images(ref_imgs_i, self.resolution_ref)
            ref_imgs_new.append(ref_imgs_i)

        return {
            "txt": txt,
            "img": img,
            "ref_imgs": ref_imgs_new,
        }

class FluxPairedDataset_My(Dataset):
    def __init__(self, json_file: str,resolution_ref: int, image_base: str):
        super().__init__()
        self.json_file = json_file
        self.resolution_ref = resolution_ref
        self.image_root = image_base

        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]
        if  isinstance(data_dict["ref_img"], list):
            ref_img_paths = data_dict["ref_img"]
        else:
            ref_img_paths = [data_dict["ref_img"]]
        text = data_dict["text"]
        ref_imgs = [
            Image.open(os.path.join(self.image_root, path)).convert("RGB")
            for path in ref_img_paths
        ]
        ref_imgs = [self.transform(img) for img in ref_imgs]

        ref_img_paths_new = [os.path.join(self.image_root, path) for path in ref_img_paths]

        # print("ref_imgs 的结构:")
        
        # if hasattr(ref_imgs, 'shape'):
        #     print(f"  ref_imgs: shape = {ref_imgs.shape}")
        # else:
        #     print(f"  ref_imgs: type = {type(ref_imgs)}, len={len(ref_imgs)}")

        return {
            "text": text,
            "ref_imgs": ref_imgs,
            "ref_img_paths": ref_img_paths_new
        }

    def __len__(self):
        return len(self.data_dicts)
    
    def collate_fn(self, batch):
        text = [data["text"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        ref_img_paths = [data["ref_img_paths"] for data in batch]
        
        n_ref = len(ref_imgs[0])

        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = bucket_images(ref_imgs_i, self.resolution_ref)
            ref_imgs_new.append(ref_imgs_i)

        # # 打印所有 ref_imgs_new 的shape
        # print("ref_imgs_new 的结构:")
        # for i, imgs in enumerate(ref_imgs_new):
        #     if hasattr(imgs, 'shape'):
        #         print(f"  ref_imgs_new[{i}]: shape = {imgs.shape}")
        #     else:
        #         print(f"  ref_imgs_new[{i}]: type = {type(imgs)}, len={len(imgs)}")
        return {
            "text": text,
            "ref_imgs": ref_imgs_new,
            "ref_img_paths": ref_img_paths
        }

class FluxPairedDataset_without_text(Dataset):
    def __init__(self, json_file: str,resolution_ref: int, image_base: str):
        super().__init__()
        self.json_file = json_file
        self.resolution_ref = resolution_ref
        self.image_root = image_base

        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]
        if  isinstance(data_dict["ref_img"], list):
            ref_img_paths = data_dict["ref_img"]
        else:
            ref_img_paths = [data_dict["ref_img"]]
        text = data_dict["text"]
        ref_imgs = [
            Image.open(os.path.join(self.image_root, path)).convert("RGB")
            for path in ref_img_paths
        ]
        ref_imgs = [self.transform(img) for img in ref_imgs]

        ref_img_paths_new = [os.path.join(self.image_root, path) for path in ref_img_paths]

        # print("ref_imgs 的结构:")
        
        # if hasattr(ref_imgs, 'shape'):
        #     print(f"  ref_imgs: shape = {ref_imgs.shape}")
        # else:
        #     print(f"  ref_imgs: type = {type(ref_imgs)}, len={len(ref_imgs)}")

        return {
            "text": text,
            "ref_imgs": ref_imgs,
            "ref_img_paths": ref_img_paths_new,
            "txt_save_path": data_dict["txt_save_path"],
            "vec_save_path": data_dict["vec_save_path"],
            "ids": data_dict["id"]
        }

    def __len__(self):
        return len(self.data_dicts)
    
    def collate_fn(self, batch):
        text = [data["text"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        ref_img_paths = [data["ref_img_paths"] for data in batch]
        txt_save_path = [data["txt_save_path"] for data in batch]
        vec_save_path = [data["vec_save_path"] for data in batch]
        ids = [data["ids"] for data in batch]
        n_ref = len(ref_imgs[0])

        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = bucket_images(ref_imgs_i, self.resolution_ref)
            ref_imgs_new.append(ref_imgs_i)

        # 加载txt/vec并stack为tensor
        encoder_hidden_states_list = []
        pool_prompt_embed_list = []
        for txt_p, vec_p in zip(txt_save_path, vec_save_path):
            # 兼容 torch.save(tensor) 或 torch.save({'txt':tensor})
            txt_obj = torch.load(txt_p, map_location='cpu')
            vec_obj = torch.load(vec_p, map_location='cpu')
            # 支持dict或张量
            if isinstance(txt_obj, dict):
                txt_tensor = txt_obj.get("txt", list(txt_obj.values())[0])
            else:
                txt_tensor = txt_obj
            if isinstance(vec_obj, dict):
                vec_tensor = vec_obj.get("vec", list(vec_obj.values())[0])
            else:
                vec_tensor = vec_obj
            # print("Temp txt_tensor:", txt_tensor.shape)
            # print("Temp txt_tensor:", txt_tensor.shape)

            if txt_tensor.dim() == 3 and txt_tensor.size(0) == 1:
                txt_tensor = txt_tensor.squeeze(0)

            if vec_tensor.dim() == 2 and vec_tensor.size(0) == 1:
                vec_tensor = vec_tensor.squeeze(0)
            encoder_hidden_states_list.append(txt_tensor)
            pool_prompt_embed_list.append(vec_tensor)
        # stack
        encoder_hidden_states = torch.stack(encoder_hidden_states_list, dim=0)
        pool_prompt_embed = torch.stack(pool_prompt_embed_list, dim=0)
        
        # print("dataset:encoder_hidden_states.shape:", encoder_hidden_states.shape)
        # print("dataset:pool_prompt_embed.shape:",pool_prompt_embed.shape)
        # return
        return {
            "text": text,
            "ref_imgs": ref_imgs_new,
            "ref_img_paths": ref_img_paths,
            "encoder_hidden_states": encoder_hidden_states,
            "pool_prompt_embed": pool_prompt_embed,
            "ids": ids
        }

class FluxPairedDataset_DreamSingle_Eval(Dataset):
    def __init__(self, json_file: str,resolution_ref: int, image_base: str):
        super().__init__()
        self.json_file = json_file
        self.resolution_ref = resolution_ref
        self.image_root = image_base

        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]
        if  isinstance(data_dict["image_paths"], list):
            ref_img_paths = data_dict["image_paths"]
        else:
            ref_img_paths = [data_dict["image_paths"]]
        text = data_dict["prompt"]
        ref_imgs = [
            Image.open(os.path.join(self.image_root, path)).convert("RGB")
            for path in ref_img_paths
        ]
        ref_imgs = [self.transform(img) for img in ref_imgs]

        # ref_img_paths_new = [os.path.join(self.image_root, path) for path in ref_img_paths]

        # print("ref_imgs 的结构:")
        
        # if hasattr(ref_imgs, 'shape'):
        #     print(f"  ref_imgs: shape = {ref_imgs.shape}")
        # else:
        #     print(f"  ref_imgs: type = {type(ref_imgs)}, len={len(ref_imgs)}")

        return {
            "text": text,
            "ref_imgs": ref_imgs,
        }

    def __len__(self):
        return len(self.data_dicts)
    
    def collate_fn(self, batch):
        text = [data["text"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        
        n_ref = len(ref_imgs[0])

        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = bucket_images(ref_imgs_i, self.resolution_ref)
            ref_imgs_new.append(ref_imgs_i)

        return {
            "text": text,
            "ref_imgs": ref_imgs_new,
        }

class FluxPairedDataset_SFT(Dataset):
    def __init__(self, json_file: str, resolution: int,resolution_ref: int, image_base: str):
        super().__init__()
        self.json_file = json_file
        self.resolution = resolution
        self.resolution_ref = resolution_ref
        self.image_root = image_base

        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]
        
        img = None
        image_tgt_path = data_dict["img"]
        if image_tgt_path is not None:
            img = Image.open(os.path.join(self.image_root, image_tgt_path)).convert("RGB")
            img = self.transform(img)

        if  isinstance(data_dict["ref_img"], list):
            ref_img_paths = data_dict["ref_img"]
        else:
            ref_img_paths = [data_dict["ref_img"]]
        text = data_dict["text"]
        ref_imgs = [
            Image.open(os.path.join(self.image_root, path)).convert("RGB")
            for path in ref_img_paths
        ]
        ref_imgs = [self.transform(img) for img in ref_imgs]

        return {
            "text": text,
            "img": img,
            "ref_imgs": ref_imgs,
        }

    def __len__(self):
        return len(self.data_dicts)
    
    def collate_fn(self, batch):
        text = [data["text"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        img = [data["img"] for data in batch]
        
        n_ref = len(ref_imgs[0])

        img = bucket_images(img, self.resolution)
        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = bucket_images(ref_imgs_i, self.resolution_ref)
            ref_imgs_new.append(ref_imgs_i)

        return {
            "text": text,
            "img": img,
            "ref_imgs": ref_imgs_new,
        }

class FluxPairedDataset_DreamSingle(Dataset):
    def __init__(self, json_file: str, resolution_ref: int, image_base: str):
        super().__init__()
        self.json_file = json_file
        self.resolution_ref = resolution_ref
        self.image_root = image_base

        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]
        if  isinstance(data_dict["image_paths"], list):
            ref_img_paths = data_dict["image_paths"]
        else:
            ref_img_paths = [data_dict["image_paths"]]
        text = data_dict["prompt"]
        ref_imgs = [
            Image.open(os.path.join(self.image_root, path)).convert("RGB")
            for path in ref_img_paths
        ]
        ref_imgs = [self.transform(img) for img in ref_imgs]

        return {
            "text": text,
            "ref_imgs": ref_imgs,
        }

    def __len__(self):
        return len(self.data_dicts)
    
    def collate_fn(self, batch):
        text = [data["text"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]

        n_ref = len(ref_imgs[0])

        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = bucket_images(ref_imgs_i, self.resolution_ref)
            ref_imgs_new.append(ref_imgs_i)

        return {
            "text": text,
            "ref_imgs": ref_imgs_new,
        }

if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser()
    # parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--json_file", type=str, default="/data/oss_bucket_0/ziwei/datasets/syncd_meta/train_1w_prompt_embedding.json")
    parser.add_argument("--image_base", type=str, default="/data/oss_bucket_0/ziwei/datasets/syncd/")
    args = parser.parse_args()
    dataset = FluxPairedDataset_without_text(args.json_file, 512, args.image_base)
    dataloder = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(dataloder))
    print(f"batch type = {type(batch)}")
    print(f"batch = {batch}")