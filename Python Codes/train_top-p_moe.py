"""
===========================================================
train_moe.py
===========================================================

Train a Feed-forward Mixture of Experts regression model.

Input:

        X

        |

        MoE

        |

        Y_hat


Evaluate:

        Test RMSE vs number of training samples


Supports:

- Apple Silicon MPS
- CUDA
- CPU

===========================================================
"""

import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

###########################################################################
# Device
###########################################################################

def get_device():
    if torch.backends.mps.is_available():
        print("\nUsing Apple Metal (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("\nUsing CUDA")
        return torch.device("cuda")
    else:
        print("\nUsing CPU")
        return torch.device("cpu")

def seed_everything(seed = 7649297):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


###########################################################################
# Load Dataset
############################################################################

def load_dataset(dataset_path):
    data = torch.load(dataset_path)
    X = data["X"].float()
    Y = data["Y"].float()
    print(f"--------------------------------------------------------")
    print(f"Loaded dataset from {dataset_path} with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"Target shape: {Y.shape}")
    print(f"--------------------------------------------------------")
    return X, Y


###########################################################################
#Standardize 
###########################################################################

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(0, keepdim = True)
        self.std = X.std(0, keepdim = True)+1e-8
    def transform(self, X):
        return (X - self.mean)/self.std
    def inverse(self, X):
        return X*self.std + self.mean        


############################################################################
# Expert Network
############################################################################


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

############################################################################
# Mixture of Experts Network
############################################################################

#Fix Topk=r

given_p=0.7
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_experts=4, top_p=given_p):
        super().__init__()
        self.num_experts = num_experts
        self.top_p = top_p
        if not(0.0< top_p <=1.0):
            raise ValueError(
                f"top_p must satisfy 0 < top_p <= 1.0. Got top_p={top_p}."
            )
        self.experts = nn.ModuleList([Expert(input_dim, output_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x, return_components=False):
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=1)

        sorted_probs, sorted_indices = torch.sort(gate_probs, dim=1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=1)

        exclusive_cumsum = cumulative_probs - sorted_probs
        keep_mask_sorted = exclusive_cumsum < self.top_p
        keep_mask_sorted[:,0] = True

        keep_mask = torch.zeros_like(gate_probs, dtype = torch.bool)
        keep_mask.scatter_(1, sorted_indices, keep_mask_sorted)

        masked_logits = gate_logits.masked_fill(~keep_mask, float("-inf"))
        gate_weights = torch.softmax(masked_logits, dim=1)

        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)

        if return_components:
            gate_outputs = gate_weights.unsqueeze(-1)
            return (output, expert_outputs, gate_outputs, keep_mask)


        return output

############################################################################
# Computing the Bounding Curve for the Mixture of Experts
############################################################################   

def compute_empirical_l2_measures(model, X_train, C=1.0, L=1.0):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        _, expert_outputs, gate_outputs, keep_mask = model(X_train.to(device), return_components=True)
        expert_l2 = torch.sqrt(torch.mean(torch.sum(expert_outputs**2, dim=-1), dim=0))
        max_expert_l2 = expert_l2.max().item()
        gate_l2 = torch.sqrt(torch.mean(torch.sum(gate_outputs**2, dim=-1))).item()

        r_eff = keep_mask.sum(dim=1).float().mean().item()

        # Complexity Bound
        n=X_train.shape[0]
        d_in = X_train.shape[1]
        d_out = expert_outputs.shape[-1]
        K = model.num_experts
        r = r_eff
        p = model.top_p
        denom = d_in * (d_out+K)
        a = (L * gate_l2 * n) / (p * d_in *(K + d_out))
        b = (K * (K + 1) * max_expert_l2 * n) / (p * p * d_in * (K-1) * (K+d_out))
        bound = math.sqrt((denom/n)  
                 + (d_in * d_out * math.log(a))/n
                 + (d_in * K * math.log(b))/n)





        return expert_l2.cpu(), max_expert_l2, gate_l2, bound, r_eff


############################################################################
# Gate Weights Outputs on the test/specific set
############################################################################  

def compute_gate_weight_outputs(model, X_eval, eps=1e-12):
     device = next(model.parameters()).device
     model.eval()
     with torch.no_grad():
        _, _, gate_outputs, _ = model(X_eval.to(device), return_components=True)
        eff_probs = gate_outputs.squeeze(-1)  # (B, K), zeros for pruned experts, sums to 1
        avg_gate_weights = eff_probs.mean(dim=0).cpu().numpy().tolist()
 
        gate_logits = model.gate(X_eval.to(device))
        raw_gate_probs = torch.softmax(gate_logits, dim=1)  # (B, K), full support
        avg_raw_gate_weights = raw_gate_probs.mean(dim=0).cpu().numpy().tolist()
 
        # Per-sample Shannon entropy H(p) = -sum p*log(p), clamped to avoid log(0);
        # 0*log(0) terms contribute ~0 anyway so clamping doesn't bias the result.
        raw_entropy = -(raw_gate_probs * torch.log(raw_gate_probs.clamp_min(eps))).sum(dim=1)
        avg_raw_entropy = raw_entropy.mean().item()
 
        eff_entropy = -(eff_probs * torch.log(eff_probs.clamp_min(eps))).sum(dim=1)
        avg_effective_entropy = eff_entropy.mean().item()
 
     return avg_gate_weights, avg_raw_gate_weights, avg_raw_entropy, avg_effective_entropy      


############################################################################
# Training Loop
############################################################################  

def train_model(model, X_train, Y_train, X_test, Y_test, device, num_epochs = 30, lr = 1e-3, batch_size = 1024):
    device = get_device()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr = lr)
    loss_fn = nn.MSELoss()
    best_rmse = float("inf")
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0])
        for i in range(0, len(permutation), batch_size):
            indices = permutation[i:i+batch_size]
            xb = X_train[indices].to(device)
            yb = Y_train[indices].to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation

        model.eval()
        with torch.no_grad():
            pred_train = model(X_train.to(device))
            rmse_train = torch.sqrt(torch.mean((pred_train-Y_train.to(device))**2))
            pred_test = model(X_test.to(device))
            test_rmse = torch.sqrt(torch.mean((pred_test-Y_test.to(device))**2))
        if test_rmse.item()<best_rmse:
            best_rmse = test_rmse.item()
            best_train_rmse = rmse_train.item()
    return best_rmse, best_train_rmse


#############################################################################
# The Main Funcition
#############################################################################

def main():
    seed_everything()
    device = get_device()
    d_in = 32
    d_out = 16
    X,Y= load_dataset(f'/Users/sg63684/Desktop/PhD Stuffs/Mixture of Experts/Simulations/Generated_Datasets/General_Case/transformer_dataset_in{d_in}_out{d_out}.pt')

    sx = StandardScaler()
    sy = StandardScaler()

    sx.fit(X)
    sy.fit(Y)

    X = sx.transform(X)
    Y = sy.transform(Y)

    n = X.shape[0]
    perm = torch.randperm(n)
    train_size = int(0.8*n)
    test_size = n-train_size

    test_idx = perm[:test_size]

    X_test = X[test_idx]
    Y_test = Y[test_idx]

    train_idx = perm[test_size:]

    train_sizes = [int(train_size*i/100) for i in range(1, 101)]  # 1% to 100% of training data in increments of 1%
    #train_sizes = [int(train_size*0.1), int(train_size*0.2), int(train_size*0.3), int(train_size*0.4), int(train_size*0.5), int(train_size*0.6), int(train_size*0.7), int(train_size*0.8), int(train_size*0.9), train_size]

    rmse_list = []
    train_rmse_list = []
    bound_list = []
    results = []

    best_model = None
    best_rmse = float("inf")
    for size in train_sizes:
        print(f"\n-------------------------------------------------------------------------")
        print(f"Training with {size} samples...")
        idx = train_idx[:size]
        X_train = X[idx]
        Y_train = Y[idx]

        model = MixtureOfExperts(input_dim=X.shape[1], output_dim=Y.shape[1], hidden_dim=256, num_experts=4, top_p=given_p)
        test_rmse, train_rmse = train_model(model, X_train, Y_train, X_test, Y_test, device=device)
        expert_l2_device, max_expert_l2, gate_l2, bound, r_eff = compute_empirical_l2_measures(model, X_train)
        expert_l2_device_np = expert_l2_device.numpy().tolist()

        (avg_gate_weights_test, avg_raw_gate_weights_test,
          avg_raw_entropy_test, avg_effective_entropy_test) = compute_gate_weight_outputs(model, X_test)
        results.append({
            "train_size": size,
            "test_rmse": test_rmse,
            "train_rmse": train_rmse,
            "expert_l2_device": expert_l2_device_np,
            "max_expert_l2": max_expert_l2,
            "gate_l2": gate_l2,
            "bound": bound,
            "no_of_experts_chosen": r_eff,
            "avg_gate_weights_test": avg_gate_weights_test,
            "avg_raw_gate_weights_test": avg_raw_gate_weights_test,
            "avg_raw_entorpy_test": avg_raw_entropy_test,
            "avg_effective_entropy_test": avg_effective_entropy_test
        })
        print(f"\nEmpirical L2 measures - Expert L2 Device: {expert_l2_device}, Max Expert L2: {max_expert_l2}, Gate L2: {gate_l2}")
        print(f"\n Number of Experts Chosen:{math.ceil(r_eff)}")
        print(f"\nAverage effective gate weights (test set): {avg_gate_weights_test}")
        print(f"\nAverage raw softmax gate weights (test set): {avg_raw_gate_weights_test}")
        print(f"\nAverage raw gate entropy (test set, nats): {avg_raw_entropy_test:.4f} (max possible: {math.log(model.num_experts):.4f})")
        print(f"\nAverage effective gate entropy (test set, nats): {avg_effective_entropy_test:.4f}")
        print(f"\nTest RMSE for {size} samples: {test_rmse}")
        print(f"Train RMSE for {size} samples: {train_rmse}")
        print(f"Bound for {size} samples: {bound}")
        rmse_list.append(test_rmse)
        train_rmse_list.append(train_rmse)
        bound_list.append(bound)
        if test_rmse<best_rmse:
            best_rmse = test_rmse
            best_train_rmse = train_rmse
            best_model = model
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"/Users/sg63684/Desktop/PhD Stuffs/Mixture of Experts/Simulations/Results_CSV/Top_P_Case/in{d_in}_out{d_out}.csv", index=False)
    print("\nSaved statistics to results/moe_training_statistics.csv")       

    Path("models").mkdir(parents=True, exist_ok=True)  
    torch.save(best_model.state_dict(), "models/best_moe_model.pt")

    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, rmse_list, marker='o')
    plt.plot(train_sizes, bound_list, marker='x')
    plt.plot(train_sizes, train_rmse_list, marker='s')
    plt.legend(["Test RMSE", "Bound", "Train RMSE"])
    plt.xlabel("Number of Training Samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Number of Training Samples")
    plt.grid()
    plt.savefig(f'/Users/sg63684/Desktop/PhD Stuffs/Mixture of Experts/Simulations/New_Figures/Top_P_case/in{d_in}_out{d_out}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__=="__main__":    
    main()

         

                


        
