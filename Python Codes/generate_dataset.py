"""
===========================================================
generate_dataset.py
===========================================================

Generate a synthetic regression dataset

X ~ N(0,I)

↓

Frozen Pretrained Transformer

↓

Y_clean

↓

Y = Y_clean + Gaussian Noise

↓

Save

    transformer_dataset.csv
    transformer_dataset.pt

Author:
-----------------------------------------------------------
Designed for Apple Silicon (MPS), CUDA and CPU.

===========================================================
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn





##########################################################################
# Device
##########################################################################

def get_device():

    if torch.backends.mps.is_available():
        print("\nUsing Apple Metal (MPS)")
        return torch.device("mps")

    elif torch.cuda.is_available():
        print("\nUsing CUDA")
        print(torch.cuda.get_device_name(0))
        return torch.device("cuda")

    else:
        print("\nUsing CPU")
        return torch.device("cpu")


device = get_device()
##########################################################################
# Random Seed
##########################################################################

def seed_everything(seed=1234):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



##########################################################################
# Frozen Transformer
##########################################################################

import torch
import torch.nn as nn

class FrozenTransformer(nn.Module):

    def __init__(
        self,
        input_dim=16,
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = self.embedding(x)
        return self.encoder(x)

    


##########################################################################
# Gaussian Generator
##########################################################################

def generate_gaussian_samples(
    n_samples,
    dimension,
    device,
):

    return torch.randn(
        n_samples,
        dimension,
        device=device,
    )







##########################################################################
# Noise
##########################################################################

def add_noise(
    Y,
    sigma,
):

    noise = sigma * torch.randn_like(Y)

    return Y + noise



##########################################################################
# Batch Generator
##########################################################################

@torch.no_grad()
def generate_outputs(
    model,
    X,
    batch_size,
    device,
):

    """
    Compute

        Y = Transformer(X)

    in mini-batches.
    """

    outputs = []

    model.eval()

    N = X.shape[0]

    for start in tqdm(
        range(0, N, batch_size),
        desc="Generating",
    ):

        stop = min(start + batch_size, N)

        batch = X[start:stop]

        #######################################################
        # transformer expects
        #
        # (batch,sequence,input_dimension)
        #
        # every sample is treated as a sequence
        # of length one
        #######################################################

        batch = batch.unsqueeze(1)

        batch = batch.to(device)

        y = model(batch)

        y = y.squeeze(1)

        outputs.append(y.cpu())

    outputs = torch.cat(outputs, dim=0)

    return outputs




##########################################################################
# Save Dataset
##########################################################################

def save_dataset(
    X,
    Y,
    output_dir='.../Mixture of Experts/Simulations/Generated_Datasets',
    dataset_name="transformer_dataset_in32_out32",
    save_csv=True,
    save_pt=True,
):
    """
    Save the generated dataset.

    Parameters
    ----------
    X : torch.Tensor
        Shape (N, input_dim)

    Y : torch.Tensor
        Shape (N, output_dim)

    output_dir : str
        Directory to save files

    dataset_name : str
        Base filename

    save_csv : bool
        Save CSV version

    save_pt : bool
        Save PyTorch version
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ########################################################
    # Convert to CPU
    ########################################################

    X = X.detach().cpu()
    Y = Y.detach().cpu()

    ########################################################
    # Save .pt
    ########################################################

    if save_pt:

        pt_path = output_dir / f"{dataset_name}.pt"

        torch.save(
            {
                "X": X,
                "Y": Y,
                "num_samples": X.shape[0],
                "input_dimension": X.shape[1],
                "output_dimension": Y.shape[1],
            },
            pt_path,
        )

        print(f"Saved PyTorch dataset to\n{pt_path}")

    ########################################################
    # Save CSV
    ########################################################

    if save_csv:

        X_np = X.numpy()
        Y_np = Y.numpy()

        x_columns = [
            f"X{i+1}"
            for i in range(X_np.shape[1])
        ]

        y_columns = [
            f"Y{i+1}"
            for i in range(Y_np.shape[1])
        ]

        df = pd.DataFrame(
            np.concatenate(
                [X_np, Y_np],
                axis=1,
            ),
            columns=x_columns + y_columns,
        )

        csv_path = output_dir / f"{dataset_name}.csv"

        df.to_csv(
            csv_path,
            index=False,
            float_format="%.8f",
        )

        print(f"Saved CSV dataset to\n{csv_path}")

    ########################################################

    print("--------------------------------------")
    print("Dataset Summary")
    print("--------------------------------------")
    print(f"Samples          : {X.shape[0]}")
    print(f"Input dimension  : {X.shape[1]}")
    print(f"Output dimension : {Y.shape[1]}")
    print("--------------------------------------")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int, default=1000000)

    parser.add_argument("--input_dim", type=int, default=32)

    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--noise", type=float, default=0.0005)

    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


if __name__ == "__main__":

    #########################################################
    # Parse arguments
    #########################################################

    args = parse_args()

    #########################################################
    # Seed
    #########################################################

    seed_everything(args.seed)

    #########################################################
    # Device
    #########################################################

    device = get_device()

    #########################################################
    # Build transformer
    #########################################################

    model = FrozenTransformer(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
    )

    model.to(device)

    #########################################################
    # Generate Gaussian data
    #########################################################

    print("\nGenerating Gaussian samples...")

    X = generate_gaussian_samples(
        args.n_samples,
        args.input_dim,
        device,
    )

    #########################################################
    # Transformer
    #########################################################

    print("\nGenerating transformer outputs...")

    Y = generate_outputs(
        model,
        X,
        args.batch_size,
        device,
    )

    #########################################################
    # Add noise
    #########################################################

    print("\nAdding Gaussian noise...")

    Y = add_noise(
        Y,
        args.noise,
    )

    #########################################################
    # Save
    #########################################################

    save_dataset(
        X.cpu(),
        Y.cpu(),
    )

    print("\nFinished!")
