import csv
import logging
import os
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
import torch

import cv2

logger = logging.getLogger(__name__)

def load_pairs_from_csv(base_dir: str, csv_filename: str) -> List[Dict]:
    csv_path = os.path.join(base_dir, csv_filename)
    rows = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_a_path = os.path.join(base_dir, row["scene_id"], row["image_a"])
            image_b_path = os.path.join(base_dir, row["scene_id"], row["image_b"])
            rows.append(
                {
                    "image_a_path": image_a_path,
                    "image_b_path": image_b_path,
                    "scene_id": row.get("scene_id"),
                    "pair_id": f"{row.get('scene_id', 'unknown')}_{row.get('image_a', '')}_{row.get('image_b', '')}",
                }
            )
    return rows

def load_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image

class FederatedMatcherDataset(Dataset):
    def __init__(self, data_pairs: List[Dict], transform=None):
        self.data_pairs = data_pairs
        self.transform = transform

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]

        image_a = load_image(pair["image_a_path"])
        image_b = load_image(pair["image_b_path"])

        image_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

        if self.transform:
            image_a_gray = self.transform(image_a_gray)
            image_b_gray = self.transform(image_b_gray)

        sample = {
            "image_a": torch.from_numpy(image_a_gray).float().unsqueeze(0) / 255.0,
            "image_b": torch.from_numpy(image_b_gray).float().unsqueeze(0) / 255.0,
            "pair_id": pair["pair_id"],
            "scene_id": pair.get("scene_id", "unknown")
        }

        return sample

def create_federated_dataloader(data_pairs: List[Dict], batch_size: int = 4,
                               shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = FederatedMatcherDataset(data_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_federated_batch
    )
    return dataloader

def collate_federated_batch(batch):
    batch_dict = {
        "image_a":   torch.stack([item["image_a"] for item in batch]),
        "image_b":   torch.stack([item["image_b"] for item in batch]),
        "pair_ids":  [item["pair_id"]  for item in batch],
        "scene_ids": [item["scene_id"] for item in batch],
    }
    return batch_dict
