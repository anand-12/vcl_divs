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
from torch.cuda.amp import autocast, GradScaler
import random

# --- Set Seed Function ---
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Optional: Ensures deterministic behavior in CuDNN (can impact performance)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

# --- Directory Setup ---
def setup_experiment_dirs(experiment_name, alpha, use_coreset, coreset_method=None, seed=None):
    """Create necessary directories for saving results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coreset_str = f"_coreset_{coreset_method}" if use_coreset else ""
    seed_str = f"_seed{seed}" if seed is not None else ""
    alpha_str = f"alpha{alpha:.2f}".replace('.', 'p') # Format alpha for filename
    exp_name = f"{experiment_name}_{alpha_str}{coreset_str}{seed_str}_{timestamp}"
    base_dir = os.path.join("results_combined", exp_name)
    model_dir = os.path.join(base_dir, "models")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return base_dir, model_dir, plot_dir

# --- Bayesian Layer (with Amari Alpha-Divergence) ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('weight_prior_mu', torch.zeros(out_features, in_features))
        self.register_buffer('weight_prior_log_var', torch.zeros(out_features, in_features))
        self.register_buffer('bias_prior_mu', torch.zeros(out_features))
        self.register_buffer('bias_prior_log_var', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_log_var, -6)
        bound = 1 / np.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_log_var, -6)

    def forward(self, x, sample=True):
        if self.training or sample:
            weight_eps = torch.randn_like(self.weight_log_var)
            weight_std = torch.exp(0.5 * self.weight_log_var)
            weight = self.weight_mu + weight_eps * weight_std
            bias_eps = torch.randn_like(self.bias_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias = self.bias_mu + bias_eps * bias_std
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _kl_divergence(self, mu_q, log_var_q, mu_p, log_var_p):
        """ KL(q || p) for two diagonal Gaussians """
        var_q = torch.exp(log_var_q)
        var_p = torch.exp(log_var_p)
        var_p = torch.clamp(var_p, min=1e-8)
        kl_div = 0.5 * torch.sum(
            log_var_p - log_var_q +
            (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
        )
        return torch.clamp(kl_div, min=0)

    def _amari_alpha_divergence_single_gaussian(self, mu_q, log_var_q, mu_p, log_var_p, alpha):
        """
        Computes Amari's alpha divergence D_alpha(P || Q) for diagonal Gaussians P, Q.
        P = N(mu_p, var_p), Q = N(mu_q, var_q)
        Formula: D_alpha(P || Q) = (4 / (1 - alpha^2)) * (1 - Integral)
        Integral = sqrt(var_q / var_beta) * exp( (1-alpha^2)/8 * (mu_p-mu_q)^2 / var_beta )
        where var_beta = (1+alpha)/2 * var_q + (1-alpha)/2 * var_p
        """
        if abs(alpha) == 1.0:
            raise ValueError("Alpha cannot be exactly 1 or -1 for Amari alpha-divergence formula.")

        var_q = torch.exp(log_var_q)
        var_p = torch.exp(log_var_p)

        # Prevent division by zero or instability near alpha = +/- 1
        alpha_sq = alpha * alpha
        denominator = alpha_sq - 1.0
        if abs(denominator) < 1e-6:
             # fallback to KL
             return torch.tensor(float('inf'), device=mu_q.device)


        # Calculate var_beta = beta * var_q + (1-beta) * var_p
        # beta = (1+alpha)/2, 1-beta = (1-alpha)/2
        var_beta = 0.5 * (1.0 + alpha) * var_q + 0.5 * (1.0 - alpha) * var_p
        var_beta = torch.clamp(var_beta, min=1e-8) # Ensure positivity

        # Calculate log of the integral term for numerical stability
        log_var_q = log_var_q
        log_var_beta = torch.log(var_beta)
        mu_diff_sq = (mu_p - mu_q).pow(2)

        # Log Integral = 0.5 * (log_var_q - log_var_beta) + ( (1-alpha^2)/8 * mu_diff_sq / var_beta )
        log_integral_term = 0.5 * (log_var_q - log_var_beta) + (denominator / 8.0) * (mu_diff_sq / var_beta)

        # Clamp the exponent before exp to avoid potential overflow/underflow
        log_integral_term = torch.clamp(log_integral_term, max=50.0, min=-50.0) # Adjust bounds if needed

        integral_term = torch.exp(log_integral_term)

        # Calculate divergence
        divergence = (4.0 / denominator) * (1.0 - integral_term)

        # Sum over all dimensions
        return torch.sum(torch.clamp(divergence, min=0)) # Ensure non-negativity


    def divergence(self, alpha=1.0):
        """ Calculate divergence D(q || p_prior) using KL or Amari alpha """
        # Use KL if alpha is close to 1 (forward KL)
        if abs(alpha - 1.0) < 1e-4:
            # print("Using KL divergence (alpha approx 1)")
            return self._kl_divergence(
                self.weight_mu, self.weight_log_var,
                self.weight_prior_mu, self.weight_prior_log_var
            ) + self._kl_divergence(
                self.bias_mu, self.bias_log_var,
                self.bias_prior_mu, self.bias_prior_log_var
            )
        # Use Reverse KL if alpha is close to -1
        elif abs(alpha + 1.0) < 1e-4:
             # print("Using Reverse KL divergence (alpha approx -1)")
             # Calculate KL(prior || posterior)
             return self._kl_divergence(
                 self.weight_prior_mu, self.weight_prior_log_var,
                 self.weight_mu, self.weight_log_var,
             ) + self._kl_divergence(
                 self.bias_prior_mu, self.bias_prior_log_var,
                 self.bias_mu, self.bias_log_var,
             )
        # Use Amari Alpha-Divergence otherwise
        else:
            # print(f"Using Amari Alpha divergence (alpha={alpha})")
            # Note: The formula is D_alpha(P || Q). Here P=prior, Q=posterior.
            # We want D_alpha(Q || P_prior)
            # D_alpha(Q || P) = D_{-alpha}(P || Q)
            effective_alpha = -alpha
            if abs(abs(effective_alpha) - 1.0) < 1e-4: # Check again after flipping sign
                 # This should not happen if initial checks passed, but for safety
                 print(f"Warning: Effective alpha {-alpha} near +/- 1, falling back to KL.")
                 return self.divergence(alpha=1.0) # Fallback to standard KL

            try:
                div_w = self._amari_alpha_divergence_single_gaussian(
                    self.weight_prior_mu, self.weight_prior_log_var, # P = prior
                    self.weight_mu, self.weight_log_var,             # Q = posterior
                    effective_alpha                                  # Use -alpha
                )
                div_b = self._amari_alpha_divergence_single_gaussian(
                    self.bias_prior_mu, self.bias_prior_log_var,       # P = prior
                    self.bias_mu, self.bias_log_var,                   # Q = posterior
                    effective_alpha                                    # Use -alpha
                )
                return div_w + div_b
            except ValueError as e:
                 print(f"Error computing alpha-divergence with alpha={alpha}: {e}. Falling back to KL.")
                 return self.divergence(alpha=1.0)


    def update_prior(self):
        self.weight_prior_mu.data.copy_(self.weight_mu.data.clone().detach())
        self.weight_prior_log_var.data.copy_(self.weight_log_var.data.clone().detach())
        self.bias_prior_mu.data.copy_(self.bias_mu.data.clone().detach())
        self.bias_prior_log_var.data.copy_(self.bias_log_var.data.clone().detach())

# --- Multi-Head Bayesian MLP (Updated divergence call) ---
class BayesianMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_tasks=1, num_classes_per_task=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.fc1 = BayesianLinear(input_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, hidden_size)
        self.heads = nn.ModuleList([
            BayesianLinear(hidden_size, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id, sample=True):
        # ... (forward remains the same) ...
        if task_id >= len(self.heads):
             raise ValueError(f"Task ID {task_id} out of range for {len(self.heads)} heads.")
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = self.heads[task_id](x, sample)
        return x

    def divergence(self, current_task_id, alpha=1.0):
        """ Calculate total divergence D(q || p_prior) using specified alpha """
        # Pass alpha to layer divergence methods
        total_div = self.fc1.divergence(alpha) + self.fc2.divergence(alpha)
        for i in range(current_task_id + 1):
             if i < len(self.heads):
                 total_div += self.heads[i].divergence(alpha)
        return total_div

    def update_priors(self, current_task_id):
        # ... (update_priors remains the same) ...
        self.fc1.update_prior()
        self.fc2.update_prior()
        for i in range(current_task_id + 1):
             if i < len(self.heads):
                self.heads[i].update_prior()
        print(f"Updated priors for shared layers and heads up to task {current_task_id}")


# --- Coreset Manager (Unchanged) ---
class CoresetManager:
    # ... (CoresetManager code remains the same) ...
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
        else: raise ValueError(f"Unknown method: {self.selection_method}")

    def _random_selection(self, dataset, coreset_num):
        indices = np.random.choice(len(dataset), size=coreset_num, replace=False)
        return indices.tolist()

    def _k_center_selection(self, dataset, coreset_num):
        num_samples = len(dataset)
        if num_samples == 0: return []
        data_loader = DataLoader(dataset, batch_size=min(5000, num_samples), shuffle=False)
        all_data_list = []; original_indices = list(range(num_samples))
        current_offset = 0; temp_original_indices = []
        for data, _ in data_loader:
            all_data_list.append(data.view(data.size(0), -1).cpu().numpy())
            batch_indices = list(range(current_offset, current_offset + len(data)))
            temp_original_indices.extend(batch_indices)
            current_offset += len(data)
        if len(temp_original_indices) == num_samples: original_indices = temp_original_indices
        elif hasattr(dataset, 'indices'): original_indices = dataset.indices

        if not all_data_list: return []
        X = np.concatenate(all_data_list, axis=0)
        n_samples_loaded = X.shape[0]
        if n_samples_loaded == 0: return []

        selected_indices_in_X = []
        first_center_idx_in_X = np.random.randint(0, n_samples_loaded)
        selected_indices_in_X.append(first_center_idx_in_X)
        centers = X[[first_center_idx_in_X]]
        min_distances = pairwise_distances(X, centers).min(axis=1)

        for _ in range(1, coreset_num):
            if len(selected_indices_in_X) >= n_samples_loaded: break
            new_center_idx_in_X = np.argmax(min_distances)
            if new_center_idx_in_X in selected_indices_in_X:
                 sorted_dist_indices = np.argsort(min_distances)[::-1]
                 for idx in sorted_dist_indices:
                     if idx not in selected_indices_in_X: new_center_idx_in_X = idx; break
                 else: print(f"Warning: k-center could not find {coreset_num} unique points."); break
            selected_indices_in_X.append(new_center_idx_in_X)
            centers = np.vstack([centers, X[new_center_idx_in_X]])
            new_center_distances = pairwise_distances(X, X[[new_center_idx_in_X]]).flatten()
            min_distances = np.minimum(min_distances, new_center_distances)
        final_selected_indices = [original_indices[i] for i in selected_indices_in_X]
        return final_selected_indices

    def add_coreset_for_task(self, dataset, task_id):
        print(f"Selecting coreset for task {task_id} using {self.selection_method}...")
        indices = self.select_coreset(dataset)
        self.task_coresets[task_id] = {'dataset': dataset, 'indices': indices}
        print(f"Coreset size for task {task_id}: {len(indices)}")
        return Subset(dataset, indices)

    def get_combined_past_coreset_subset(self, current_task_id):
        past_subsets = []
        for task_id, coreset_info in self.task_coresets.items():
            if task_id < current_task_id and coreset_info['indices']:
                 past_subsets.append(Subset(coreset_info['dataset'], coreset_info['indices']))
        if not past_subsets: return None
        print(f"Creating combined past coreset from tasks {[tid for tid in self.task_coresets if tid < current_task_id]}")
        return ConcatDataset(past_subsets)

# --- Data Loading (Unchanged) ---
def get_permuted_mnist(task_id, permute_seed=42):
    # ... (get_permuted_mnist code remains the same) ...
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
    # ... (SplitMNISTTaskDataset remains the same) ...
    def __init__(self, base_dataset, task_id):
        if not 0 <= task_id <= 4: raise ValueError("Split MNIST task_id must be between 0 and 4.")
        self.base_dataset = base_dataset; self.task_id = task_id
        self.digit1 = task_id * 2; self.digit2 = task_id * 2 + 1
        self.indices = []; self.remapped_targets = []
        targets = self._get_targets(base_dataset)
        for idx, target in enumerate(targets):
            target_val = target.item() if isinstance(target, torch.Tensor) else target
            if target_val == self.digit1: self.indices.append(idx); self.remapped_targets.append(0)
            elif target_val == self.digit2: self.indices.append(idx); self.remapped_targets.append(1)
    def _get_targets(self, dataset):
        if hasattr(dataset, 'targets'): return dataset.targets
        elif hasattr(dataset, 'labels'): return dataset.labels
        else: raise AttributeError("Dataset missing .targets or .labels")
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]; data, _ = self.base_dataset[original_idx]
        target = self.remapped_targets[idx]; return data, target

def get_split_mnist(task_id, data_root='./data'):
    # ... (get_split_mnist remains the same) ...
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_task_dataset = SplitMNISTTaskDataset(mnist_train, task_id)
    test_task_dataset = SplitMNISTTaskDataset(mnist_test, task_id)
    return train_task_dataset, test_task_dataset


# --- Training and Evaluation Functions (Pass alpha) ---

def train_task_and_past_coresets(model, task_id, train_loader, optimizer, device, past_coreset_loader=None, alpha=1.0, num_epochs=100):
    """ Trains model (updates \tilde{q}_t). Uses specified alpha divergence. """
    model.train()
    scaler = GradScaler() if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'

    print(f"Training Task {task_id} with alpha={alpha}...")
    n_task_samples = 0
    try: n_task_samples = len(train_loader.dataset)
    except TypeError: n_task_samples = 50000 # Estimate if needed

    for epoch in range(num_epochs):
        epoch_loss = 0
        total_data_count = 0

        # Train on current task data D_t
        if n_task_samples > 0:
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Task {task_id} Data", leave=False):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                optimizer.zero_grad()
                with autocast(enabled=use_amp):
                    output = model(data, task_id=task_id, sample=True)
                    ce_loss = F.cross_entropy(output, target)
                    # Use specified alpha for divergence, scale by N_t
                    div_loss = model.divergence(current_task_id=task_id, alpha=alpha)
                    div_loss_scaled = div_loss / n_task_samples if n_task_samples > 0 else 0
                    loss = ce_loss + div_loss_scaled
                if use_amp: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else: loss.backward(); optimizer.step()
                epoch_loss += loss.item() * batch_size
                total_data_count += batch_size

        # Train on past coreset data C_{t-1} (Apply only divergence penalty)
        if past_coreset_loader is not None and len(past_coreset_loader.dataset) > 0:
            for data, target in tqdm(past_coreset_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Past Coreset", leave=False):
                 data, target = data.to(device), target.to(device)
                 batch_size = data.size(0)
                 optimizer.zero_grad()
                 with autocast(enabled=use_amp):
                     ce_loss = 0 # Ignore CE loss
                     # Use specified alpha for divergence, scale by N_t
                     div_loss = model.divergence(current_task_id=task_id, alpha=alpha)
                     div_loss_scaled = div_loss / n_task_samples if n_task_samples > 0 else 0
                     loss = ce_loss + div_loss_scaled
                 if loss.requires_grad:
                     if use_amp: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                     else: loss.backward(); optimizer.step()
                 epoch_loss += loss.item() if isinstance(loss, torch.Tensor) else loss * batch_size
                 total_data_count += batch_size

        avg_epoch_loss = epoch_loss / total_data_count if total_data_count > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
             print(f"Epoch {epoch+1}/{num_epochs}, Task {task_id} Combined Loss: {avg_epoch_loss:.6f}")


def refine_with_current_coreset(model_tilde_q, task_id, current_coreset_loader, device, alpha=1.0, num_refine_epochs=10, lr_refine=1e-4):
    """ Refines model using current coreset C_t (updates q_t). Uses specified alpha divergence. """
    if current_coreset_loader is None or len(current_coreset_loader.dataset) == 0:
        print("No current coreset provided or coreset is empty. Skipping refinement.")
        return copy.deepcopy(model_tilde_q)
    print(f"Refining posterior for task {task_id} using current coreset (size: {len(current_coreset_loader.dataset)})...")
    model_q = copy.deepcopy(model_tilde_q)
    model_q.train()
    optimizer_refine = optim.Adam(model_q.parameters(), lr=lr_refine)
    scaler = GradScaler() if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    model_q.update_priors(current_task_id=task_id) # Prior = \tilde{q}_t
    coreset_size = len(current_coreset_loader.dataset)
    if coreset_size == 0: coreset_size = 1

    for epoch in range(num_refine_epochs):
        for data, target in tqdm(current_coreset_loader, desc=f"Refine Epoch {epoch+1}/{num_refine_epochs}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer_refine.zero_grad()
            with autocast(enabled=use_amp):
                output = model_q(data, task_id=task_id, sample=True)
                ce_loss = F.cross_entropy(output, target)
                # Use specified alpha for divergence KL(q_t || \tilde{q}_t), scale by coreset size
                div_loss = model_q.divergence(current_task_id=task_id, alpha=alpha)
                div_loss_scaled = div_loss / coreset_size
                loss = ce_loss + div_loss_scaled
            if use_amp: scaler.scale(loss).backward(); scaler.step(optimizer_refine); scaler.update()
            else: loss.backward(); optimizer_refine.step()
    print("Refinement complete.")
    model_q.eval()
    return model_q


def evaluate(model, task_id, test_loader, device):
    # ... (evaluate code remains the same) ...
    model.eval()
    test_loss = 0; correct = 0; actual_samples_processed = 0
    try:
        if len(test_loader.dataset) == 0: return 0.0, 0.0
    except TypeError: pass
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Evaluating Task {task_id}", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data, task_id=task_id, sample=False)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            actual_samples_processed += data.size(0)
    if actual_samples_processed == 0: return 0.0, 0.0
    test_loss /= actual_samples_processed
    accuracy = correct / actual_samples_processed
    return test_loss, accuracy

# --- Save/Plot Functions (Unchanged) ---
def save_checkpoint(model, optimizer, task_id, path):
    # ... (save_checkpoint remains the same) ...
    torch.save({
        'task_id': task_id, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path); print(f"Checkpoint saved to {path}")

def plot_results(task_accuracies, plot_dir, experiment_name):
    # ... (plot_results remains the same) ...
    if not task_accuracies: print("No accuracies to plot."); return
    num_tasks_learned = len(task_accuracies)
    if num_tasks_learned == 0: print("No accuracies recorded."); return
    num_tasks_evaluated = max(len(accs_at_t) for accs_at_t in task_accuracies if accs_at_t) if any(task_accuracies) else 0
    if num_tasks_evaluated == 0: print("No evaluation results found."); return

    plt.figure(figsize=(15, 8))
    num_plots = num_tasks_evaluated + 1; cols = 3; rows = (num_plots + cols - 1) // cols
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.suptitle(f'{experiment_name} - VCL Performance', fontsize=16)
    plot_path_png = os.path.join(plot_dir, f"{experiment_name}_accuracies.png")
    plot_path_pdf = os.path.join(plot_dir, f"{experiment_name}_accuracies.pdf")
    try: plt.savefig(plot_path_png); plt.savefig(plot_path_pdf); print(f"Plot saved to {plot_path_png} and {plot_path_pdf}")
    except Exception as e: print(f"Error saving plot: {e}")
    plt.close()


# --- Experiment Runners (Pass alpha) ---
def run_permuted_mnist_experiment(
    num_tasks=10, hidden_size=100, alpha=1.0, use_coreset=False,
    coreset_method='random', coreset_size=200, batch_size=256,
    num_epochs=100, num_refine_epochs=10, lr=1e-3, lr_refine=1e-4,
    seed=None, device=None
):
    experiment_name = "permuted_mnist"
    base_dir, model_dir, plot_dir = setup_experiment_dirs(
        experiment_name, alpha, use_coreset, coreset_method, seed
    )
    model = BayesianMLP(input_size=784, hidden_size=hidden_size, num_tasks=1, num_classes_per_task=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"\n--- Running Permuted MNIST ---")
    print(f"Config: {num_tasks=}, {hidden_size=}, {alpha=}, {use_coreset=}, {coreset_method=}, {coreset_size=}, {batch_size=}, {num_epochs=}, {num_refine_epochs=}, {lr=}, {lr_refine=}, {seed=}")

    coreset_manager = None
    if use_coreset: coreset_manager = CoresetManager(selection_method=coreset_method, coreset_size=coreset_size)
    all_task_accuracies = []
    results = {"config": { # Log config ...
         "experiment": experiment_name, "num_tasks": num_tasks, "hidden_size": hidden_size, "alpha": alpha,
         "use_coreset": use_coreset, "coreset_method": coreset_method if use_coreset else None,
         "coreset_size": coreset_size if use_coreset else None, "batch_size": batch_size,
         "num_epochs": num_epochs, "num_refine_epochs": num_refine_epochs if use_coreset else 0,
         "learning_rate": lr, "refine_learning_rate": lr_refine if use_coreset else 0, "seed": seed
        }, "task_accuracies": [], "avg_accuracies": []
    }

    for task_id in range(num_tasks):
        print(f"\n{'='*20} Task {task_id} {'='*20}")
        train_dataset, _ = get_permuted_mnist(task_id)
        current_model_task_id = 0 # Single head
        worker_init = lambda worker_id: np.random.seed(seed + worker_id) if seed is not None else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=device.type=='cuda', num_workers=4 if device.type=='cuda' else 0, worker_init_fn=worker_init)
        past_coreset_loader = None
        if use_coreset and task_id > 0:
             combined_past_coreset = coreset_manager.get_combined_past_coreset_subset(current_task_id=task_id)
             if combined_past_coreset and len(combined_past_coreset) > 0:
                 past_coreset_loader = DataLoader(combined_past_coreset, batch_size=min(batch_size, len(combined_past_coreset)), shuffle=True, pin_memory=device.type=='cuda', num_workers=4 if device.type=='cuda' else 0, worker_init_fn=worker_init)
                 print(f"Using past coreset data (size: {len(combined_past_coreset)})")

        # Pass alpha to training function
        train_task_and_past_coresets(model, current_model_task_id, train_loader, optimizer, device, past_coreset_loader, alpha, num_epochs)

        current_coreset_subset = None; current_coreset_loader = None
        if use_coreset:
            current_coreset_subset = coreset_manager.add_coreset_for_task(train_dataset, task_id)
            if current_coreset_subset and len(current_coreset_subset) > 0:
                 current_coreset_loader = DataLoader(current_coreset_subset, batch_size=min(batch_size, len(current_coreset_subset)), shuffle=True, pin_memory=device.type=='cuda', num_workers=4 if device.type=='cuda' else 0, worker_init_fn=worker_init)

        if use_coreset and current_coreset_loader:
            # Pass alpha to refinement function
            eval_model = refine_with_current_coreset(model, current_model_task_id, current_coreset_loader, device, alpha, num_refine_epochs, lr_refine)
        else:
            eval_model = model

        accuracies_after_task_t = []
        print(f"Evaluating performance after task {task_id}...")
        for prev_task_id in range(task_id + 1):
            _, prev_test_dataset = get_permuted_mnist(prev_task_id)
            prev_test_loader = DataLoader(prev_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type=='cuda', num_workers=4 if device.type=='cuda' else 0)
            _, acc = evaluate(eval_model, current_model_task_id, prev_test_loader, device)
            accuracies_after_task_t.append(acc); print(f"  Task {prev_task_id} Test Accuracy: {acc:.4f}")

        all_task_accuracies.append(accuracies_after_task_t)
        avg_acc = np.mean(accuracies_after_task_t) if accuracies_after_task_t else 0
        results["task_accuracies"].append(accuracies_after_task_t); results["avg_accuracies"].append(avg_acc)
        with open(os.path.join(base_dir, "results.json"), "w") as f: json.dump(results, f, indent=2)

        model.update_priors(current_model_task_id)
        save_checkpoint(model, optimizer, task_id, os.path.join(model_dir, f"task_{task_id}_tilde_q_checkpoint.pt"))

    final_avg_acc = results["avg_accuracies"][-1] if results["avg_accuracies"] else 0
    print(f"\nFinal Average Accuracy (Permuted MNIST): {final_avg_acc:.4f}")
    plot_results(all_task_accuracies, plot_dir, experiment_name)
    return all_task_accuracies, results


def run_split_mnist_experiment(
    num_tasks=5, hidden_size=256, alpha=1.0, use_coreset=False,
    coreset_method='random', coreset_size=40, num_epochs=120,
    num_refine_epochs=20, lr=1e-3, lr_refine=1e-4, seed=None, device=None
):
    experiment_name = "split_mnist"
    base_dir, model_dir, plot_dir = setup_experiment_dirs(
        experiment_name, alpha, use_coreset, coreset_method, seed
    )
    model = BayesianMLP(input_size=784, hidden_size=hidden_size, num_tasks=num_tasks, num_classes_per_task=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"\n--- Running Split MNIST ---")
    print(f"Config: {num_tasks=}, {hidden_size=}, {alpha=}, {use_coreset=}, {coreset_method=}, {coreset_size=}, batch_size='Full', {num_epochs=}, {num_refine_epochs=}, {lr=}, {lr_refine=}, {seed=}")

    coreset_manager = None
    if use_coreset: coreset_manager = CoresetManager(selection_method=coreset_method, coreset_size=coreset_size)
    all_task_accuracies = []
    results = {"config": { # Log config ...
         "experiment": experiment_name, "num_tasks": num_tasks, "hidden_size": hidden_size, "alpha": alpha,
         "use_coreset": use_coreset, "coreset_method": coreset_method if use_coreset else None,
         "coreset_size": coreset_size if use_coreset else None, "batch_size": "Full",
         "num_epochs": num_epochs, "num_refine_epochs": num_refine_epochs if use_coreset else 0,
         "learning_rate": lr, "refine_learning_rate": lr_refine if use_coreset else 0, "seed": seed
        }, "task_accuracies": [], "avg_accuracies": []
    }

    for task_id in range(num_tasks):
        print(f"\n{'='*20} Task {task_id} (Digits {task_id*2}/{task_id*2+1}) {'='*20}")
        train_dataset, _ = get_split_mnist(task_id)
        current_batch_size = len(train_dataset)
        if current_batch_size == 0:
             print(f"Warning: Task {task_id} training set empty. Skipping.");
             if all_task_accuracies: all_task_accuracies.append(all_task_accuracies[-1])
             results["task_accuracies"].append([]); results["avg_accuracies"].append(results["avg_accuracies"][-1] if results["avg_accuracies"] else 0.0)
             continue

        worker_init = lambda worker_id: np.random.seed(seed + worker_id) if seed is not None else None
        train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, pin_memory=device.type=='cuda', num_workers=0)
        past_coreset_loader = None
        if use_coreset and task_id > 0:
             combined_past_coreset = coreset_manager.get_combined_past_coreset_subset(current_task_id=task_id)
             if combined_past_coreset and len(combined_past_coreset) > 0:
                 past_coreset_batch_size = min(128, len(combined_past_coreset))
                 past_coreset_loader = DataLoader(combined_past_coreset, batch_size=past_coreset_batch_size, shuffle=True, pin_memory=device.type=='cuda', num_workers=4 if device.type=='cuda' else 0, worker_init_fn=worker_init)
                 print(f"Using past coreset data (size: {len(combined_past_coreset)})")

        # Pass alpha to training function
        train_task_and_past_coresets(model, task_id, train_loader, optimizer, device, past_coreset_loader, alpha, num_epochs)

        current_coreset_subset_obj = None; current_coreset_loader = None
        if use_coreset:
            current_coreset_subset_obj = coreset_manager.add_coreset_for_task(train_dataset, task_id)
            if current_coreset_subset_obj and len(current_coreset_subset_obj) > 0:
                 current_coreset_loader = DataLoader(current_coreset_subset_obj, batch_size=min(current_batch_size, len(current_coreset_subset_obj)), shuffle=True, pin_memory=device.type=='cuda', num_workers=4 if device.type=='cuda' else 0, worker_init_fn=worker_init)

        if use_coreset and current_coreset_loader:
            # Pass alpha to refinement function
            eval_model = refine_with_current_coreset(model, task_id, current_coreset_loader, device, alpha, num_refine_epochs, lr_refine)
        else:
            eval_model = model

        accuracies_after_task_t = []
        print(f"Evaluating performance after task {task_id}...")
        for prev_task_id in range(task_id + 1):
            _, prev_test_dataset = get_split_mnist(prev_task_id)
            eval_batch_size = 256
            prev_test_loader = DataLoader(prev_test_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=device.type=='cuda', num_workers=4 if device.type=='cuda' else 0)
            _, acc = evaluate(eval_model, prev_task_id, prev_test_loader, device)
            accuracies_after_task_t.append(acc); print(f"  Task {prev_task_id} Test Accuracy: {acc:.4f}")

        all_task_accuracies.append(accuracies_after_task_t)
        avg_acc = np.mean(accuracies_after_task_t) if accuracies_after_task_t else 0
        results["task_accuracies"].append(accuracies_after_task_t); results["avg_accuracies"].append(avg_acc)
        with open(os.path.join(base_dir, "results.json"), "w") as f: json.dump(results, f, indent=2)

        model.update_priors(task_id)
        save_checkpoint(model, optimizer, task_id, os.path.join(model_dir, f"task_{task_id}_tilde_q_checkpoint.pt"))

    final_avg_acc = results["avg_accuracies"][-1] if results["avg_accuracies"] else 0
    print(f"\nFinal Average Accuracy (Split MNIST): {final_avg_acc:.4f}")
    plot_results(all_task_accuracies, plot_dir, experiment_name)
    return all_task_accuracies, results


# --- Main Execution Block (Simplified Args) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Continual Learning with Alpha-Divergence")
    parser.add_argument("--experiment", type=str, default="permuted",
                        choices=["permuted", "split"],
                        help="Which experiment to run (permuted or split MNIST)")
    # General arguments
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks (for Permuted MNIST, Split MNIST is fixed at 5)")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden layer size (default 100; SplitMNIST uses 256)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default 256; SplitMNIST uses Full)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs (default 100; SplitMNIST uses 120)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--coreset_size", type=int, default=200, help="Coreset size per task (default 200; SplitMNIST uses 40)")

    # Alpha divergence argument
    parser.add_argument("--alpha", type=float, default=1.0, help="Amari alpha-divergence parameter alpha (alpha=1 uses KL, alpha=-1 uses reverse KL)")

    # Common arguments
    parser.add_argument("--use_coreset", action="store_true", help="Whether to use coresets")
    parser.add_argument("--coreset_method", type=str, default="random", choices=["random", "k-center"], help="Coreset selection method")
    parser.add_argument("--num_refine_epochs", type=int, default=10, help="Number of epochs for coreset refinement step (tune)")
    parser.add_argument("--lr_refine", type=float, default=1e-4, help="Learning rate for coreset refinement step (tune)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        set_seed(args.seed)
    else:
        print("Running without fixed random seed.")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if use_cuda: print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else: print("Using CPU")

    print(f"Selected experiment: {args.experiment}")
    alpha_desc = "KL" if abs(args.alpha - 1.0) < 1e-4 else "Reverse KL" if abs(args.alpha + 1.0) < 1e-4 else f"Amari Alpha={args.alpha:.2f}"
    print(f"Using Divergence: {alpha_desc}")


    if args.experiment == "permuted":
        # Use general args, defaults match paper well
        run_permuted_mnist_experiment(
            num_tasks=args.num_tasks,
            hidden_size=args.hidden_size,
            alpha=args.alpha, # Pass alpha
            use_coreset=args.use_coreset,
            coreset_method=args.coreset_method,
            coreset_size=args.coreset_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_refine_epochs=args.num_refine_epochs,
            lr=args.lr,
            lr_refine=args.lr_refine,
            seed=args.seed,
            device=device
        )
    elif args.experiment == "split":
        # Override specific args with Split MNIST paper values
        split_hidden_size = 256
        split_coreset_size = 40
        split_epochs = 120
        split_lr = 1e-3
        num_split_tasks = 5

        print(f"NOTE: For Split MNIST, using {num_split_tasks=}, {split_hidden_size=}, {split_coreset_size=}, {split_epochs=}, {split_lr=}.")
        print(f"      Ignoring command-line --num_tasks, --hidden_size, --batch_size, --num_epochs, --lr, --coreset_size if provided.")

        run_split_mnist_experiment(
            num_tasks=num_split_tasks, # Fixed
            hidden_size=split_hidden_size,
            alpha=args.alpha, # Pass alpha
            use_coreset=args.use_coreset,
            coreset_method=args.coreset_method,
            coreset_size=split_coreset_size,
            num_epochs=split_epochs,
            num_refine_epochs=args.num_refine_epochs, # Keep tunable
            lr=split_lr,
            lr_refine=args.lr_refine, # Keep tunable
            seed=args.seed,
            device=device
        )
    else:
        print(f"Unknown experiment: {args.experiment}")