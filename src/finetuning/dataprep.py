import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomFeatureDataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.img_name = data_dict["img_name"]
        self.features = np.array(data_dict["features"], dtype="float32")
        self.Label = data_dict.get(
            "Label", None
        )  # Handle cases where Label might be missing

    def __getitem__(self, idx):
        features = self.features[idx]
        img_name = self.img_name[idx]

        if self.mode in ["train", "val"]:
            label = np.array(
                self.Label[idx], dtype=np.int64
            ).squeeze()  # Ensure proper label shape
            return features, label, img_name
        return features, img_name

    def __len__(self):
        return len(self.img_name)

def create_data_loader_from_df(
    df, mode="test", batch_size=1, num_workers=2, pin_memory=True
):

    # Prepare the samples
    samples = {
        "features": df["vector"].tolist(),
        "img_name": df["name"].tolist(),
        "Label": df["label"].tolist(),  # Label is optional for test mode
    }

    # Create the dataset
    dataset = CustomFeatureDataset(samples, mode)

    # Create and return the DataLoader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )