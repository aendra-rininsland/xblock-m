#!/usr/bin/env python
import os
import torch
import json
from timm import create_model
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import os
from constants import THRESHOLD
import torchvision.transforms as T

# from dotenv import load_dotenv
import numpy as np

# import ai_edge_torch
# import numpy
# import torchvision
import datasets
import torch.nn as nn

# load_dotenv()  # take environment variables from .env.
dataset = datasets.load_dataset("howdyaendra/xblock-social-screenshots", token=True)

# split up training into training + validation
splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

img_size = (224, 224)

train_tfms = T.Compose(
    [
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

valid_tfms = T.Compose(
    [
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def train_transforms(batch):
    # convert all images in batch to RGB to avoid grayscale or transparent images
    batch["image"] = [x.convert("RGB") for x in batch["image"]]
    # apply torchvision.transforms per sample in the batch
    inputs = [train_tfms(x) for x in batch["image"]]
    batch["pixel_values"] = inputs

    # one-hot encoding the labels
    labels = torch.tensor([[x] for x in batch["label"]])
    batch["labels"] = nn.functional.one_hot(labels, num_classes=num_classes)
    batch["labels"] = batch["labels"].sum(dim=1)

    return batch


def valid_transforms(batch):
    # convert all images in batch to RGB to avoid grayscale or transparent images
    batch["image"] = [x.convert("RGB") for x in batch["image"]]
    # apply torchvision.transforms per sample in the batch
    inputs = [valid_tfms(x) for x in batch["image"]]
    batch["pixel_values"] = inputs

    # one-hot encoding the labels
    labels = torch.tensor([[x] for x in batch["label"]])
    batch["labels"] = nn.functional.one_hot(labels, num_classes=num_classes)
    batch["labels"] = batch["labels"].sum(dim=1)

    return batch


train_dataset = train_ds.with_transform(train_transforms)
valid_dataset = val_ds.with_transform(valid_transforms)
test_dataset = dataset["train"].with_transform(valid_transforms)

len(train_dataset), len(valid_dataset), len(test_dataset)

# Set the number of threads to 1
torch.set_num_threads(1)

# Use environment variables
NUM_WORKERS = 50
MODEL_NAME = os.getenv("MODEL_NAME", "swin_s3_base_224-xblockm-timm")
MODEL_PATH = os.getenv("MODEL_PATH", "./model")

# Check if CUDA (GPU) is available; if not, default to CPU
device = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define model details
model_id = f"howdyaendra/{MODEL_NAME}"
cache_dir = "./models"

# Download model files
model_weights_path = hf_hub_download(
    repo_id=model_id, filename="model.safetensors", cache_dir=cache_dir
)
config_path = hf_hub_download(
    repo_id=model_id, filename="config.json", cache_dir=cache_dir
)
# Load configuration
with open(config_path) as f:
    config = json.load(f)
print(config)
num_classes = config.get("num_classes", 13)
# Create the model and load weights
model_name = "swin_s3_base_224"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

samples = test_dataset.shuffle().select(np.arange(2, 2))


def create_model_instance():
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.to(device)
    # Load weights
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


xblockm = create_model_instance()
sample_inputs = (torch.randn(1, 3, 224, 224),)
torch_output = xblockm(*sample_inputs)
print(torch_output)
