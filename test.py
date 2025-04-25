import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import numpy as np
import argparse
import copy
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler # Still import, but use conditionally
import random
import platform # To check for macOS

try:
    import geomloss # Import geomloss for Sinkhorn
except ImportError:
    print("Error: geomloss library not found.")
    print("Please install it using: pip install geomloss")
    exit()


# --- Set Seed Function ---
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # MPS seeding is less straightforward, manual seed is usually sufficient
    # Optional: Ensures deterministic behavior in CuDNN (can impact performance)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

# --- Directory Setup ---
def setup_experiment_dirs(experiment_name, use_coreset, coreset_method=None, seed=None, divergence_type="kl"):
    """Create necessary directories for saving results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coreset_str = f"_coreset_{coreset_method}" if use_coreset else ""
    seed_str = f"_seed{seed}" if seed is not None else ""
    # Use alpha=1 for VCL paper comparison if KL, otherwise note Sinkhorn
    exp_prefix = f"{experiment_name}_{divergence_type}"
    exp_name = f"{exp_prefix}{coreset_str}{seed_str}_{timestamp}"
    base_dir = os.path.join("results_combined", exp_name)
    model_dir = os.path.join(base_dir, "models")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return base_dir, model_dir, plot_dir

# --- Bayesian Layer ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Parameters for the posterior q_t
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))
        # Parameters for the prior q_{t-1} (stored as buffers)
        self.register_buffer('weight_prior_mu', torch.zeros(out_features, in_features))
        self.register_buffer('weight_prior_log_var', torch.zeros(out_features, in_features))
        self.register_buffer('bias_prior_mu', torch.zeros(out_features))
        self.register_buffer('bias_prior_log_var', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_log_var, -6) # Initialize with low variance
        bound = 1 / np.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_log_var, -6) # Initialize with low variance

    def forward(self, x, sample=True):
        # Sample weights and biases if training or explicitly requested
        if self.training or sample:
            # Reparameterization trick
            weight_eps = torch.randn_like(self.weight_log_var)
            weight_std = torch.exp(0.5 * self.weight_log_var)
            weight = self.weight_mu + weight_eps * weight_std

            bias_eps = torch.randn_like(self.bias_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias = self.bias_mu + bias_eps * bias_std
        else:
            # Use mean parameters for evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _kl_divergence(self, mu_q, log_var_q, mu_p, log_var_p):
        """Calculates KL divergence between two diagonal Gaussians."""
        var_q = torch.exp(log_var_q)
        var_p = torch.exp(log_var_p)
        # Add small epsilon for numerical stability if prior variance is zero
        var_p = torch.clamp(var_p, min=1e-8)
        kl_div = 0.5 * torch.sum(
            log_var_p - log_var_q +
            (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
        )
        # Ensure KL is non-negative
        return torch.clamp(kl_div, min=0)

    def _sample_params(self, mu, log_var, num_samples):
        """Samples parameters using the reparameterization trick and flattens them."""
        std = torch.exp(0.5 * log_var)
        # Generate random noise for sampling
        # Ensure eps is generated on the same device as mu
        eps = torch.randn(num_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        # Sample using mu + eps * std
        samples = mu.unsqueeze(0) + eps * std.unsqueeze(0) # Shape: (num_samples, *mu.shape)
        # Flatten parameters for each sample
        samples_flat = samples.view(num_samples, -1) # Shape: (num_samples, num_params)
        return samples_flat

    def kl_divergence(self):
        """Calculates the analytical KL divergence (original VCL)."""
        return self._kl_divergence(
            self.weight_mu, self.weight_log_var,
            self.weight_prior_mu, self.weight_prior_log_var
        ) + self._kl_divergence(
            self.bias_mu, self.bias_log_var,
            self.bias_prior_mu, self.bias_prior_log_var
        )

    def sinkhorn_divergence(self, num_samples, sinkhorn_eps, sinkhorn_blur, sinkhorn_scaling):
        """Calculates Sinkhorn divergence between posterior and prior samples."""
        # Sample from current posterior q_t (requires grad)
        weight_samples_q = self._sample_params(self.weight_mu, self.weight_log_var, num_samples)
        bias_samples_q = self._sample_params(self.bias_mu, self.bias_log_var, num_samples)
        # Concatenate weights and biases into a single parameter vector per sample
        params_q = torch.cat([weight_samples_q, bias_samples_q], dim=1) # Shape: (num_samples, num_params_total)

        # Sample from prior q_{t-1} (detach, no grad needed for prior samples)
        with torch.no_grad():
            weight_samples_p = self._sample_params(self.weight_prior_mu, self.weight_prior_log_var, num_samples)
            bias_samples_p = self._sample_params(self.bias_prior_mu, self.bias_prior_log_var, num_samples)
            params_p = torch.cat([weight_samples_p, bias_samples_p], dim=1) # Shape: (num_samples, num_params_total)

        # Define Sinkhorn loss using geomloss
        # p=2 means using squared Euclidean distance as the cost function C(x,y) = ||x-y||^2
        # blur is the entropy regularization strength (epsilon in OT literature)
        # scaling helps stabilize computations
        # debias=True uses an unbiased estimator, recommended for Sinkhorn
        sinkhorn_loss = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=sinkhorn_blur, # Typically epsilon = blur^p -> blur = sqrt(epsilon) for p=2
            scaling=sinkhorn_scaling, # Recommended value 0.5 or 0.7
            debias=True,
            backend="tensorized" # Can be faster on GPU/MPS
        )

        # Compute divergence
        # Ensure samples are on the same device (should be handled by _sample_params)
        div = sinkhorn_loss(params_q, params_p)

        # Clamp at 0 as debiased Sinkhorn can sometimes be slightly negative for identical inputs
        return torch.clamp(div, min=0)


    def update_prior(self):
        """Updates the prior parameters (q_{t-1}) with the current posterior parameters (q_t)."""
        self.weight_prior_mu.data.copy_(self.weight_mu.data.clone().detach())
        self.weight_prior_log_var.data.copy_(self.weight_log_var.data.clone().detach())
        self.bias_prior_mu.data.copy_(self.bias_mu.data.clone().detach())
        self.bias_prior_log_var.data.copy_(self.bias_log_var.data.clone().detach())

# --- Multi-Head Bayesian MLP ---
class BayesianMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_tasks=1, num_classes_per_task=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        # Shared layers
        self.fc1 = BayesianLinear(input_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, hidden_size)
        # Task-specific heads
        self.heads = nn.ModuleList([
            BayesianLinear(hidden_size, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id, sample=True):
        if task_id >= len(self.heads):
             raise ValueError(f"Task ID {task_id} out of range for {len(self.heads)} heads.")
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = self.heads[task_id](x, sample)
        return x

    def divergence(self, current_task_id, divergence_type="kl", **kwargs):
        """Calculates total divergence (KL or Sinkhorn) for relevant layers."""
        total_div = 0.0 # Initialize as float
        if divergence_type == "kl":
            # Ensure results are floats or tensors that support addition
            total_div += self.fc1.kl_divergence()
            total_div += self.fc2.kl_divergence()
            for i in range(current_task_id + 1):
                if i < len(self.heads):
                    total_div += self.heads[i].kl_divergence()
        elif divergence_type == "sinkhorn":
            # Extract Sinkhorn parameters from kwargs
            num_samples = kwargs.get("sinkhorn_samples", 100) # Default samples
            sinkhorn_eps = kwargs.get("sinkhorn_eps", 0.1)     # Default epsilon
            sinkhorn_blur = kwargs.get("sinkhorn_blur", sinkhorn_eps**0.5) # Default blur = sqrt(eps) for p=2
            sinkhorn_scaling = kwargs.get("sinkhorn_scaling", 0.7) # Default scaling
            # Calculate divergence for shared layers
            total_div += self.fc1.sinkhorn_divergence(num_samples, sinkhorn_eps, sinkhorn_blur, sinkhorn_scaling)
            total_div += self.fc2.sinkhorn_divergence(num_samples, sinkhorn_eps, sinkhorn_blur, sinkhorn_scaling)
            # Add divergence for relevant heads
            for i in range(current_task_id + 1):
                if i < len(self.heads):
                    total_div += self.heads[i].sinkhorn_divergence(num_samples, sinkhorn_eps, sinkhorn_blur, sinkhorn_scaling)
        else:
            raise ValueError(f"Unknown divergence type: {divergence_type}")

        return total_div

    def update_priors(self, current_task_id):
        """Updates priors for shared layers and heads up to the current task."""
        self.fc1.update_prior()
        self.fc2.update_prior()
        for i in range(current_task_id + 1):
             if i < len(self.heads):
                self.heads[i].update_prior()
        print(f"Updated priors for shared layers and heads up to task {current_task_id}")

# --- Coreset Manager ---
# CoresetManager code remains the same
class CoresetManager:
    def __init__(self, selection_method='random', coreset_size=200):
        self.selection_method = selection_method
        self.coreset_size = coreset_size
        self.task_coresets = {} # Store {'dataset': dataset, 'indices': indices}

    def select_coreset(self, dataset):
        num_samples = len(dataset)
        coreset_num = min(self.coreset_size, num_samples)
        if coreset_num <= 0: return []

        if self.selection_method == 'random':
            return self._random_selection(dataset, coreset_num)
        elif self.selection_method == 'k-center':
            return self._k_center_selection(dataset, coreset_num)
        else:
            raise ValueError(f"Unknown coreset selection method: {self.selection_method}")

    def _random_selection(self, dataset, coreset_num):
        indices = np.random.choice(len(dataset), size=coreset_num, replace=False)
        return indices.tolist()

    def _k_center_selection(self, dataset, coreset_num):
        num_samples = len(dataset)
        if num_samples == 0: return []
        # Use smaller batch size for k-center feature extraction to avoid OOM
        data_loader = DataLoader(dataset, batch_size=min(1024, num_samples), shuffle=False)
        all_data_list = []
        original_indices = list(range(num_samples)) # Assume direct indexing works for simplicity

        current_offset = 0
        temp_original_indices = []
        print("Extracting features for k-center...")
        for data, _ in tqdm(data_loader, leave=False):
            all_data_list.append(data.view(data.size(0), -1).cpu().numpy())
            # Handle potential Subset indexing
            if isinstance(dataset, Subset):
                 batch_indices = dataset.indices[current_offset : current_offset + len(data)]
            elif isinstance(dataset, ConcatDataset):
                 # This case is complex, fallback to simple indexing for now
                 batch_indices = list(range(current_offset, current_offset + len(data)))
            else: # Assume simple dataset
                 batch_indices = list(range(current_offset, current_offset + len(data)))

            temp_original_indices.extend(batch_indices)
            current_offset += len(data)

        # Use actual indices if possible
        if hasattr(dataset, 'indices') and isinstance(dataset, Subset):
             original_indices = dataset.indices
        elif len(temp_original_indices) == num_samples:
             original_indices = temp_original_indices
        else:
             print("Warning: Could not reliably determine original indices for k-center. Using range.")
             original_indices = list(range(num_samples))


        if not all_data_list: return []
        X = np.concatenate(all_data_list, axis=0)
        n_samples_loaded = X.shape[0]
        if n_samples_loaded == 0: return []
        print(f"Running k-center on {n_samples_loaded} samples...")

        selected_indices_in_X = []
        first_center_idx_in_X = np.random.randint(0, n_samples_loaded)
        selected_indices_in_X.append(first_center_idx_in_X)
        centers = X[[first_center_idx_in_X]]
        min_distances = pairwise_distances(X, centers, metric='euclidean', n_jobs=-1).min(axis=1)

        for _ in tqdm(range(1, coreset_num), desc="k-center selection", leave=False):
            if len(selected_indices_in_X) >= n_samples_loaded: break
            new_center_idx_in_X = np.argmax(min_distances)
            # Ensure uniqueness
            if new_center_idx_in_X in selected_indices_in_X:
                 sorted_dist_indices = np.argsort(min_distances)[::-1]
                 for idx in sorted_dist_indices:
                     if idx not in selected_indices_in_X:
                         new_center_idx_in_X = idx
                         break
                 else:
                     print(f"Warning: k-center could not find {coreset_num} unique points.")
                     break

            selected_indices_in_X.append(new_center_idx_in_X)
            # Optimization: update distances incrementally
            new_center_distances = pairwise_distances(X, X[[new_center_idx_in_X]], metric='euclidean', n_jobs=-1).flatten()
            min_distances = np.minimum(min_distances, new_center_distances)

        # Map selected indices in X back to original dataset indices
        final_selected_indices = [original_indices[i] for i in selected_indices_in_X]
        print("k-center selection finished.")
        return final_selected_indices


    def add_coreset_for_task(self, dataset, task_id):
        print(f"Selecting coreset for task {task_id} using {self.selection_method}...")
        indices = self.select_coreset(dataset)
        self.task_coresets[task_id] = {'dataset': dataset, 'indices': indices}
        print(f"Coreset size for task {task_id}: {len(indices)}")
        if not indices: return None
        return Subset(dataset, indices)

    def get_combined_past_coreset_subset(self, current_task_id):
        past_subsets = []
        task_ids_in_coreset = []
        for task_id, coreset_info in self.task_coresets.items():
            if task_id < current_task_id:
                 if coreset_info['indices']:
                     past_subsets.append(Subset(coreset_info['dataset'], coreset_info['indices']))
                     task_ids_in_coreset.append(task_id)
        if not past_subsets: return None
        print(f"Creating combined past coreset from tasks {task_ids_in_coreset}")
        return ConcatDataset(past_subsets)

# --- Data Loading ---
# get_permuted_mnist and SplitMNISTTaskDataset remain the same
def get_permuted_mnist(task_id, permute_seed=42):
    torch.manual_seed(permute_seed + task_id)
    permutation = torch.randperm(784)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

class SplitMNISTTaskDataset(Dataset):
    def __init__(self, base_dataset, task_id):
        if not 0 <= task_id <= 4:
            raise ValueError("Split MNIST task_id must be between 0 and 4.")
        self.base_dataset = base_dataset
        self.task_id = task_id
        self.digit1 = task_id * 2
        self.digit2 = task_id * 2 + 1
        self.indices = []
        self.remapped_targets = []
        # print(f"Filtering dataset for task {task_id} (digits {self.digit1}/{self.digit2})...")
        targets = self._get_targets(base_dataset)
        for idx, target in enumerate(targets):
            target_val = target.item() if isinstance(target, torch.Tensor) else target
            if target_val == self.digit1:
                self.indices.append(idx)
                self.remapped_targets.append(0)
            elif target_val == self.digit2:
                self.indices.append(idx)
                self.remapped_targets.append(1)
        # print(f"Task {task_id}: Found {len(self.indices)} samples.")

    def _get_targets(self, dataset):
        if hasattr(dataset, 'targets'): return dataset.targets
        elif hasattr(dataset, 'labels'): return dataset.labels
        elif hasattr(dataset, 'targets') and isinstance(dataset.targets, list): return dataset.targets # Handle list case
        elif hasattr(dataset, 'labels') and isinstance(dataset.labels, list): return dataset.labels # Handle list case
        else:
            # Fallback: Iterate through dataset if necessary (slow)
            print("Warning: Accessing targets iteratively.")
            loader = DataLoader(dataset, batch_size=1024)
            all_targets = []
            for _, targets_batch in loader:
                all_targets.extend(targets_batch.tolist())
            return all_targets


    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        data, _ = self.base_dataset[original_idx]
        target = self.remapped_targets[idx]
        return data, target


def get_split_mnist(task_id, data_root='./data'):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_task_dataset = SplitMNISTTaskDataset(mnist_train, task_id)
    test_task_dataset = SplitMNISTTaskDataset(mnist_test, task_id)
    return train_task_dataset, test_task_dataset

# --- Training and Evaluation Functions ---
def train_task_and_past_coresets(
    model, task_id, train_loader, optimizer, device,
    divergence_type, sinkhorn_params, # Added divergence params
    past_coreset_loader=None, num_epochs=100
):
    model.train()
    # Conditional AMP setup
    use_amp = (device.type == 'cuda') # Only enable AMP for CUDA
    scaler = GradScaler() if use_amp else None

    print(f"Training Task {task_id} using {divergence_type} divergence on {device}...")
    n_task_samples = 0
    try:
        n_task_samples = len(train_loader.dataset)
    except TypeError:
        print("Warning: Cannot determine size of current task dataset. Divergence scaling might be approximate.")
        n_task_samples = sum(1 for _ in train_loader) * train_loader.batch_size if train_loader.batch_size else 50000

    if n_task_samples == 0: n_task_samples = 1 # Avoid division by zero

    for epoch in range(num_epochs):
        epoch_loss = 0
        total_data_count = 0

        # Train on current task data D_t
        if len(train_loader.dataset) > 0:
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Task {task_id} Data", leave=False):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                optimizer.zero_grad()
                # Autocast only if using CUDA
                with autocast(enabled=use_amp):
                    output = model(data, task_id=task_id, sample=True)
                    ce_loss = F.cross_entropy(output, target)
                    # Calculate divergence (KL or Sinkhorn)
                    div_loss = model.divergence(
                        current_task_id=task_id,
                        divergence_type=divergence_type,
                        **sinkhorn_params # Pass Sinkhorn params if needed
                    )
                    # Scale divergence by number of samples in current task dataset
                    div_loss_scaled = div_loss / n_task_samples
                    loss = ce_loss + div_loss_scaled

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item() * batch_size
                total_data_count += batch_size

        # Train on past coreset data C_{t-1} (only divergence term matters)
        if past_coreset_loader is not None and len(past_coreset_loader.dataset) > 0:
            for data, target in tqdm(past_coreset_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Past Coreset", leave=False):
                 data, target = data.to(device), target.to(device)
                 batch_size = data.size(0)
                 optimizer.zero_grad()
                 # Autocast only if using CUDA
                 with autocast(enabled=use_amp):
                     # No CE loss for past coreset, only divergence
                     div_loss = model.divergence(
                         current_task_id=task_id,
                         divergence_type=divergence_type,
                         **sinkhorn_params
                     )
                     div_loss_scaled = div_loss / n_task_samples # Scale by N_t
                     loss = div_loss_scaled # Only divergence loss

                 # Only backprop if loss requires grad (it should if divergence is computed)
                 if isinstance(loss, torch.Tensor) and loss.requires_grad:
                     if use_amp and scaler is not None:
                         scaler.scale(loss).backward()
                         scaler.step(optimizer)
                         scaler.update()
                     else:
                         loss.backward()
                         optimizer.step()
                 # Accumulate loss (note: loss here is just scaled divergence)
                 epoch_loss += loss.item() * batch_size
                 total_data_count += batch_size

        avg_epoch_loss = epoch_loss / total_data_count if total_data_count > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
             print(f"Epoch {epoch+1}/{num_epochs}, Task {task_id} Combined Loss: {avg_epoch_loss:.6f}")


def refine_with_current_coreset(
    model_tilde_q, task_id, current_coreset_loader, device,
    divergence_type, sinkhorn_params, # Added divergence params
    num_refine_epochs=10, lr_refine=1e-4
):
    if current_coreset_loader is None or len(current_coreset_loader.dataset) == 0:
        print("No current coreset provided or coreset is empty. Skipping refinement.")
        return copy.deepcopy(model_tilde_q)

    print(f"Refining posterior for task {task_id} using current coreset (size: {len(current_coreset_loader.dataset)})...")
    model_q = copy.deepcopy(model_tilde_q)
    model_q.train()
    optimizer_refine = optim.Adam(model_q.parameters(), lr=lr_refine)
    # Conditional AMP setup
    use_amp = (device.type == 'cuda') # Only enable AMP for CUDA
    scaler = GradScaler() if use_amp else None

    # Update priors before refinement - crucial!
    model_q.update_priors(current_task_id=task_id)
    coreset_size = len(current_coreset_loader.dataset)
    if coreset_size == 0: coreset_size = 1 # Avoid division by zero

    for epoch in range(num_refine_epochs):
        for data, target in tqdm(current_coreset_loader, desc=f"Refine Epoch {epoch+1}/{num_refine_epochs}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer_refine.zero_grad()
            # Autocast only if using CUDA
            with autocast(enabled=use_amp):
                output = model_q(data, task_id=task_id, sample=True)
                ce_loss = F.cross_entropy(output, target)
                # Calculate divergence (KL or Sinkhorn)
                div_loss = model_q.divergence(
                    current_task_id=task_id,
                    divergence_type=divergence_type,
                    **sinkhorn_params
                )
                # Scale divergence by coreset size during refinement
                div_loss_scaled = div_loss / coreset_size
                loss = ce_loss + div_loss_scaled

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer_refine)
                scaler.update()
            else:
                loss.backward()
                optimizer_refine.step()

    print("Refinement complete.")
    model_q.eval()
    return model_q


def evaluate(model, task_id, test_loader, device):
    # Evaluate code remains the same
    model.eval()
    test_loss = 0
    correct = 0
    actual_samples_processed = 0
    try:
        num_samples = len(test_loader.dataset)
        if num_samples == 0:
            print(f"Warning: Test dataset for task {task_id} is empty.")
            return 0.0, 0.0
    except TypeError:
        pass # Ignore if dataset has no len

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Evaluating Task {task_id}", leave=False):
            data, target = data.to(device), target.to(device)
            # Use mean parameters for evaluation (sample=False)
            output = model(data, task_id=task_id, sample=False)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            actual_samples_processed += data.size(0)

    if actual_samples_processed == 0: return 0.0, 0.0
    test_loss /= actual_samples_processed
    accuracy = correct / actual_samples_processed
    return test_loss, accuracy

# --- Save/Plot Functions ---
# save_checkpoint and plot_results remain the same
def save_checkpoint(model, optimizer, task_id, path):
    torch.save({
        'task_id': task_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def plot_results(task_accuracies, plot_dir, experiment_name):
    if not task_accuracies: print("No accuracies to plot."); return
    num_tasks_learned = len(task_accuracies)
    if num_tasks_learned == 0: print("No accuracies recorded."); return
    num_tasks_evaluated = 0
    for accs_at_t in task_accuracies: num_tasks_evaluated = max(num_tasks_evaluated, len(accs_at_t))
    if num_tasks_evaluated == 0: print("No evaluation results found."); return

    # Determine base directory name for saving plot
    base_dir_name = os.path.basename(plot_dir.replace('/plots','')) # Get unique experiment folder name

    plt.figure(figsize=(15, 8))
    num_plots = num_tasks_evaluated + 1
    cols = 3
    rows = (num_plots + cols - 1) // cols
    task_data_collected = {i: [] for i in range(num_tasks_evaluated)}
    for t, accs_at_t in enumerate(task_accuracies):
        for task_idx, acc in enumerate(accs_at_t):
             if task_idx < num_tasks_evaluated: task_data_collected[task_idx].append((t + 1, acc))

    for i in range(num_tasks_evaluated):
        ax = plt.subplot(rows, cols, i + 1)
        if task_data_collected[i]:
             time_steps, accs = zip(*task_data_collected[i])
             ax.plot(time_steps, accs, 'o-', label=f'Task {i}')
        else: ax.plot([],[], 'o-', label=f'Task {i} (no data)')
        ax.set_xlabel('Tasks Learned'); ax.set_ylabel('Accuracy'); ax.set_title(f'Task {i} Accuracy')
        ax.set_ylim(0, 1.05); ax.set_xticks(range(1, num_tasks_learned + 1)); ax.grid(True); ax.legend()

    ax = plt.subplot(rows, cols, num_tasks_evaluated + 1)
    avg_accuracies = [np.mean(accs) for accs in task_accuracies if accs]
    if avg_accuracies: ax.plot(range(1, len(avg_accuracies) + 1), avg_accuracies, 'o-', color='red', label='Average Accuracy')
    ax.set_xlabel('Tasks Learned'); ax.set_ylabel('Average Accuracy'); ax.set_title('Average Accuracy Across All Learned Tasks')
    ax.set_ylim(0, 1.05); ax.set_xticks(range(1, num_tasks_learned + 1)); ax.grid(True); ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'{experiment_name} - VCL Performance', fontsize=16)
    plot_path_png = os.path.join(plot_dir, f"{base_dir_name}_accuracies.png") # Use base dir name
    plot_path_pdf = os.path.join(plot_dir, f"{base_dir_name}_accuracies.pdf")
    try: plt.savefig(plot_path_png); plt.savefig(plot_path_pdf); print(f"Plot saved to {plot_path_png} and {plot_path_pdf}")
    except Exception as e: print(f"Error saving plot: {e}")
    plt.close()


# --- Experiment Runners ---
def run_permuted_mnist_experiment(
    num_tasks=10, hidden_size=100, use_coreset=False,
    coreset_method='random', coreset_size=200, batch_size=256,
    num_epochs=100, num_refine_epochs=10, lr=1e-3, lr_refine=1e-4,
    seed=None, device=None,
    divergence_type="kl", sinkhorn_params=None # Added divergence args
):
    experiment_name = "permuted_mnist"
    base_dir, model_dir, plot_dir = setup_experiment_dirs(
        experiment_name, use_coreset, coreset_method, seed, divergence_type
    )
    model = BayesianMLP(input_size=784, hidden_size=hidden_size, num_tasks=1, num_classes_per_task=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Running Permuted MNIST with {divergence_type.upper()} Divergence ---")
    print(f"Config: {num_tasks=}, {hidden_size=}, {use_coreset=}, {coreset_method=}, {coreset_size=}, {batch_size=}, {num_epochs=}, {num_refine_epochs=}, {lr=}, {lr_refine=}, {seed=}")
    if divergence_type == 'sinkhorn': print(f"Sinkhorn Params: {sinkhorn_params}")

    coreset_manager = None
    if use_coreset:
        coreset_manager = CoresetManager(selection_method=coreset_method, coreset_size=coreset_size)

    all_task_accuracies = []
    # --- Results Dictionary ---
    results = {"config": {
         "experiment": experiment_name, "num_tasks": num_tasks, "hidden_size": hidden_size,
         "divergence_type": divergence_type,
         "sinkhorn_params": sinkhorn_params if divergence_type == 'sinkhorn' else None,
         "use_coreset": use_coreset, "coreset_method": coreset_method if use_coreset else None,
         "coreset_size": coreset_size if use_coreset else None, "batch_size": batch_size,
         "num_epochs": num_epochs, "num_refine_epochs": num_refine_epochs if use_coreset else 0,
         "learning_rate": lr, "refine_learning_rate": lr_refine if use_coreset else 0, "seed": seed
        }, "task_accuracies": [], "avg_accuracies": []
    }

    # Determine DataLoader settings based on device
    pin_memory = (device.type == 'cuda') # Pin memory only for CUDA
    num_workers = 4 if device.type == 'cuda' else 0 # Use workers only for CUDA

    for task_id in range(num_tasks):
        print(f"\n{'='*20} Task {task_id} {'='*20}")
        train_dataset, _ = get_permuted_mnist(task_id)
        current_model_task_id = 0 # Single head for Permuted MNIST

        worker_init = lambda worker_id: np.random.seed(seed + worker_id) if seed is not None else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=pin_memory, num_workers=num_workers,
                                worker_init_fn=worker_init)

        past_coreset_loader = None
        if use_coreset and task_id > 0:
             combined_past_coreset = coreset_manager.get_combined_past_coreset_subset(current_task_id=task_id)
             if combined_past_coreset and len(combined_past_coreset) > 0:
                 past_coreset_loader = DataLoader(combined_past_coreset, batch_size=min(batch_size, len(combined_past_coreset)), shuffle=True,
                                                  pin_memory=pin_memory, num_workers=num_workers,
                                                  worker_init_fn=worker_init)
                 print(f"Using past coreset data (size: {len(combined_past_coreset)}) for training task {task_id}.")

        # Pass divergence parameters to training function
        train_task_and_past_coresets(
            model, current_model_task_id, train_loader, optimizer, device,
            divergence_type, sinkhorn_params,
            past_coreset_loader, num_epochs
        )

        current_coreset_subset = None
        current_coreset_loader = None
        if use_coreset:
            current_coreset_subset = coreset_manager.add_coreset_for_task(train_dataset, task_id)
            if current_coreset_subset and len(current_coreset_subset) > 0:
                 current_coreset_loader = DataLoader(current_coreset_subset, batch_size=min(batch_size, len(current_coreset_subset)), shuffle=True,
                                                     pin_memory=pin_memory, num_workers=num_workers,
                                                     worker_init_fn=worker_init)

        if use_coreset and current_coreset_loader:
            # Pass divergence parameters to refinement function
            eval_model = refine_with_current_coreset(
                model, current_model_task_id, current_coreset_loader, device,
                divergence_type, sinkhorn_params,
                num_refine_epochs, lr_refine
            )
        else:
            eval_model = model # Use the model trained without refinement

        accuracies_after_task_t = []
        print(f"Evaluating performance after task {task_id}...")
        for prev_task_id in range(task_id + 1):
            _, prev_test_dataset = get_permuted_mnist(prev_task_id)
            prev_test_loader = DataLoader(prev_test_dataset, batch_size=batch_size, shuffle=False,
                                          pin_memory=pin_memory, num_workers=num_workers)
            _, acc = evaluate(eval_model, current_model_task_id, prev_test_loader, device)
            accuracies_after_task_t.append(acc)
            print(f"  Task {prev_task_id} Test Accuracy: {acc:.4f}")

        all_task_accuracies.append(accuracies_after_task_t)
        avg_acc = np.mean(accuracies_after_task_t) if accuracies_after_task_t else 0
        results["task_accuracies"].append(accuracies_after_task_t)
        results["avg_accuracies"].append(avg_acc)
        # Save results incrementally
        with open(os.path.join(base_dir, "results.json"), "w") as f: json.dump(results, f, indent=2)

        # Update priors *after* evaluation and potential refinement
        model.update_priors(current_model_task_id)
        save_checkpoint(model, optimizer, task_id, os.path.join(model_dir, f"task_{task_id}_final_checkpoint.pt"))

    final_avg_acc = results["avg_accuracies"][-1] if results["avg_accuracies"] else 0
    print(f"\nFinal Average Accuracy (Permuted MNIST, {divergence_type.upper()}): {final_avg_acc:.4f}")
    plot_results(all_task_accuracies, plot_dir, os.path.basename(base_dir)) # Use unique dir name for plot title
    return all_task_accuracies, results


def run_split_mnist_experiment(
    num_tasks=5, hidden_size=256, use_coreset=False,
    coreset_method='random', coreset_size=40, num_epochs=120,
    num_refine_epochs=20, lr=1e-3, lr_refine=1e-4,
    seed=None, device=None,
    divergence_type="kl", sinkhorn_params=None # Added divergence args
):
    experiment_name = "split_mnist"
    base_dir, model_dir, plot_dir = setup_experiment_dirs(
        experiment_name, use_coreset, coreset_method, seed, divergence_type
    )
    # Multi-headed model for Split MNIST
    model = BayesianMLP(input_size=784, hidden_size=hidden_size, num_tasks=num_tasks, num_classes_per_task=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Running Split MNIST with {divergence_type.upper()} Divergence ---")
    print(f"Config: {num_tasks=}, {hidden_size=}, {use_coreset=}, {coreset_method=}, {coreset_size=}, batch_size='Full', {num_epochs=}, {num_refine_epochs=}, {lr=}, {lr_refine=}, {seed=}")
    if divergence_type == 'sinkhorn': print(f"Sinkhorn Params: {sinkhorn_params}")

    coreset_manager = None
    if use_coreset:
        coreset_manager = CoresetManager(selection_method=coreset_method, coreset_size=coreset_size)

    all_task_accuracies = []
    # --- Results Dictionary ---
    results = {"config": {
         "experiment": experiment_name, "num_tasks": num_tasks, "hidden_size": hidden_size,
         "divergence_type": divergence_type,
         "sinkhorn_params": sinkhorn_params if divergence_type == 'sinkhorn' else None,
         "use_coreset": use_coreset, "coreset_method": coreset_method if use_coreset else None,
         "coreset_size": coreset_size if use_coreset else None, "batch_size": "Full",
         "num_epochs": num_epochs, "num_refine_epochs": num_refine_epochs if use_coreset else 0,
         "learning_rate": lr, "refine_learning_rate": lr_refine if use_coreset else 0, "seed": seed
        }, "task_accuracies": [], "avg_accuracies": []
    }

    # Determine DataLoader settings based on device
    pin_memory = (device.type == 'cuda') # Pin memory only for CUDA
    # Use 0 workers for CPU/MPS, 4 for CUDA (can be tuned)
    num_workers = 4 if device.type == 'cuda' else 0

    for task_id in range(num_tasks):
        print(f"\n{'='*20} Task {task_id} (Digits {task_id*2}/{task_id*2+1}) {'='*20}")
        train_dataset, _ = get_split_mnist(task_id)

        current_batch_size = len(train_dataset)
        if current_batch_size == 0:
             print(f"Warning: Task {task_id} training set is empty. Skipping.")
             # Maintain accuracy structure if skipping a task
             if all_task_accuracies: all_task_accuracies.append(all_task_accuracies[-1])
             results["task_accuracies"].append([])
             results["avg_accuracies"].append(results["avg_accuracies"][-1] if results["avg_accuracies"] else 0.0)
             continue

        worker_init = lambda worker_id: np.random.seed(seed + worker_id) if seed is not None else None
        # Use full batch for Split MNIST training data, no workers needed
        train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True,
                                pin_memory=pin_memory, num_workers=0)

        past_coreset_loader = None
        if use_coreset and task_id > 0:
             combined_past_coreset = coreset_manager.get_combined_past_coreset_subset(current_task_id=task_id)
             if combined_past_coreset and len(combined_past_coreset) > 0:
                 # Use smaller batches for coreset training
                 past_coreset_batch_size = min(128, len(combined_past_coreset))
                 past_coreset_loader = DataLoader(combined_past_coreset, batch_size=past_coreset_batch_size, shuffle=True,
                                                  pin_memory=pin_memory, num_workers=num_workers,
                                                  worker_init_fn=worker_init)
                 print(f"Using past coreset data (size: {len(combined_past_coreset)}) for training task {task_id}.")

        # Pass divergence parameters to training function
        train_task_and_past_coresets(
            model, task_id, train_loader, optimizer, device,
            divergence_type, sinkhorn_params,
            past_coreset_loader, num_epochs
        )

        current_coreset_subset_obj = None
        current_coreset_loader = None
        if use_coreset:
            current_coreset_subset_obj = coreset_manager.add_coreset_for_task(train_dataset, task_id)
            if current_coreset_subset_obj and len(current_coreset_subset_obj) > 0:
                 # Use smaller batches for coreset refinement
                 refine_batch_size = min(128, len(current_coreset_subset_obj))
                 current_coreset_loader = DataLoader(current_coreset_subset_obj, batch_size=refine_batch_size, shuffle=True,
                                                     pin_memory=pin_memory, num_workers=num_workers,
                                                     worker_init_fn=worker_init)

        if use_coreset and current_coreset_loader:
            # Pass divergence parameters to refinement function
            eval_model = refine_with_current_coreset(
                model, task_id, current_coreset_loader, device,
                divergence_type, sinkhorn_params,
                num_refine_epochs, lr_refine
            )
        else:
            eval_model = model # Use the model trained without refinement

        accuracies_after_task_t = []
        print(f"Evaluating performance after task {task_id}...")
        for prev_task_id in range(task_id + 1):
            _, prev_test_dataset = get_split_mnist(prev_task_id)
            eval_batch_size = 256 # Use standard batch size for eval
            prev_test_loader = DataLoader(prev_test_dataset, batch_size=eval_batch_size, shuffle=False,
                                          pin_memory=pin_memory, num_workers=num_workers)
            _, acc = evaluate(eval_model, prev_task_id, prev_test_loader, device)
            accuracies_after_task_t.append(acc)
            print(f"  Task {prev_task_id} Test Accuracy: {acc:.4f}")

        all_task_accuracies.append(accuracies_after_task_t)
        avg_acc = np.mean(accuracies_after_task_t) if accuracies_after_task_t else 0
        results["task_accuracies"].append(accuracies_after_task_t)
        results["avg_accuracies"].append(avg_acc)
        # Save results incrementally
        with open(os.path.join(base_dir, "results.json"), "w") as f: json.dump(results, f, indent=2)

        # Update priors *after* evaluation and potential refinement
        model.update_priors(task_id)
        save_checkpoint(model, optimizer, task_id, os.path.join(model_dir, f"task_{task_id}_final_checkpoint.pt"))

    final_avg_acc = results["avg_accuracies"][-1] if results["avg_accuracies"] else 0
    print(f"\nFinal Average Accuracy (Split MNIST, {divergence_type.upper()}): {final_avg_acc:.4f}")
    plot_results(all_task_accuracies, plot_dir, os.path.basename(base_dir)) # Use unique dir name for plot title
    return all_task_accuracies, results


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Continual Learning with KL or Sinkhorn Divergence")
    parser.add_argument("--experiment", type=str, default="permuted",
                        choices=["permuted", "split"],
                        help="Which experiment to run (permuted or split MNIST)")
    parser.add_argument("--divergence", type=str, default="kl", choices=["kl", "sinkhorn"],
                        help="Type of divergence for regularization (kl or sinkhorn)")

    # General arguments (used if not overridden by experiment defaults)
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks (for Permuted MNIST, Split MNIST is fixed at 5)")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden layer size (default 100)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default 256, SplitMNIST uses Full)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs (default 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--coreset_size", type=int, default=200, help="Coreset size per task (default 200)")

    # Common arguments
    parser.add_argument("--use_coreset", action="store_true", help="Whether to use coresets")
    parser.add_argument("--coreset_method", type=str, default="random", choices=["random", "k-center"], help="Coreset selection method")
    parser.add_argument("--num_refine_epochs", type=int, default=10, help="Number of epochs for coreset refinement step (tune)")
    parser.add_argument("--lr_refine", type=float, default=1e-4, help="Learning rate for coreset refinement step (tune)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU usage (CUDA and MPS)") # Changed from no_cuda

    # Sinkhorn specific arguments
    parser.add_argument("--sinkhorn_samples", type=int, default=100, help="Number of samples per distribution for Sinkhorn")
    parser.add_argument("--sinkhorn_eps", type=float, default=0.1, help="Epsilon (regularization strength) for Sinkhorn")
    parser.add_argument("--sinkhorn_blur", type=float, default=None, help="Blur parameter (sqrt(eps) default) for Sinkhorn")
    parser.add_argument("--sinkhorn_scaling", type=float, default=0.7, help="Scaling parameter for Sinkhorn stability")


    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        set_seed(args.seed)
    else:
        print("Running without fixed random seed.")

    # --- Device Selection Logic ---
    if not args.no_gpu:
        # Check for MPS (Apple Silicon GPU)
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
             device = torch.device("mps")
             print("Using MPS (Apple Silicon GPU)")
        # Check for CUDA (NVIDIA GPU)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("GPU not available (CUDA or MPS), using CPU.")
    else:
        device = torch.device("cpu")
        print("GPU usage disabled (--no_gpu), using CPU.")
    # --- End Device Selection ---


    print(f"Selected experiment: {args.experiment}")
    print(f"Using divergence: {args.divergence.upper()}")

    # Prepare Sinkhorn parameters dictionary
    sinkhorn_params = None
    if args.divergence == "sinkhorn":
        # Calculate blur from epsilon if not provided
        sinkhorn_blur = args.sinkhorn_blur if args.sinkhorn_blur is not None else args.sinkhorn_eps**0.5
        sinkhorn_params = {
            "sinkhorn_samples": args.sinkhorn_samples,
            "sinkhorn_eps": args.sinkhorn_eps,
            "sinkhorn_blur": sinkhorn_blur,
            "sinkhorn_scaling": args.sinkhorn_scaling,
        }
        print(f"Sinkhorn Parameters: Samples={args.sinkhorn_samples}, Eps={args.sinkhorn_eps:.4f}, Blur={sinkhorn_blur:.4f}, Scaling={args.sinkhorn_scaling:.2f}")


    if args.experiment == "permuted":
        run_permuted_mnist_experiment(
            num_tasks=args.num_tasks,
            hidden_size=args.hidden_size,
            use_coreset=args.use_coreset,
            coreset_method=args.coreset_method,
            coreset_size=args.coreset_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_refine_epochs=args.num_refine_epochs,
            lr=args.lr,
            lr_refine=args.lr_refine,
            seed=args.seed,
            device=device,
            divergence_type=args.divergence, # Pass divergence type
            sinkhorn_params=sinkhorn_params   # Pass sinkhorn params
        )
    elif args.experiment == "split":
        # Override specific args with Split MNIST paper values
        split_hidden_size = 256
        split_coreset_size = 40
        split_epochs = 120
        split_lr = 1e-3
        num_split_tasks = 5

        print(f"NOTE: For Split MNIST, using {num_split_tasks=}, {split_hidden_size=}, {split_coreset_size=}, {split_epochs=}, {split_lr=}.")
        print(f"      Batch size for training will be Full dataset size.")
        print(f"      Ignoring command-line --num_tasks, --hidden_size, --batch_size, --num_epochs, --lr, --coreset_size if provided.")

        run_split_mnist_experiment(
            num_tasks=num_split_tasks,
            hidden_size=split_hidden_size,
            use_coreset=args.use_coreset,
            coreset_method=args.coreset_method,
            coreset_size=split_coreset_size,
            num_epochs=split_epochs,
            num_refine_epochs=args.num_refine_epochs,
            lr=split_lr,
            lr_refine=args.lr_refine,
            seed=args.seed,
            device=device,
            divergence_type=args.divergence, # Pass divergence type
            sinkhorn_params=sinkhorn_params   # Pass sinkhorn params
        )
    else:
        print(f"Unknown experiment: {args.experiment}")
