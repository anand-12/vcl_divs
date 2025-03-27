import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import argparse
import copy
from sklearn.metrics import pairwise_distances_argmin_min
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler  

def setup_experiment_dirs(experiment_name, alpha, use_coreset, coreset_method=None):
    """Create necessary directories for saving results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coreset_str = f"_coreset_{coreset_method}" if use_coreset else ""
    exp_name = f"{experiment_name}_alpha{alpha}{coreset_str}_{timestamp}"
    
    base_dir = os.path.join("results_combiend", exp_name)
    model_dir = os.path.join(base_dir, "models")
    plot_dir = os.path.join(base_dir, "plots")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    return base_dir, model_dir, plot_dir

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))
        
        # Prior parameters (initialized as N(0,1))
        self.register_buffer('weight_prior_mu', torch.zeros(out_features, in_features))
        self.register_buffer('weight_prior_log_var', torch.zeros(out_features, in_features))
        self.register_buffer('bias_prior_mu', torch.zeros(out_features))
        self.register_buffer('bias_prior_log_var', torch.zeros(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_log_var, -6)  # Small initial variance for stability
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_log_var, -6)
        
    def forward(self, x, sample=True):
        if self.training or sample:
            # Reparameterization trick for sampling
            weight = self.weight_mu + torch.exp(0.5 * self.weight_log_var) * torch.randn_like(self.weight_log_var)
            bias = self.bias_mu + torch.exp(0.5 * self.bias_log_var) * torch.randn_like(self.bias_log_var)
        else:
            # Use means for testing (no sampling)
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    
    def _kl_divergence(self, mu_q, log_var_q, mu_p, log_var_p):

        var_q = torch.exp(log_var_q)
        var_p = torch.exp(log_var_p)
        
        kl_div = 0.5 * torch.sum(
            log_var_p - log_var_q + 
            (var_q + (mu_q - mu_p)**2) / var_p - 1
        )
        return kl_div
    
    def _renyi_divergence(self, mu_q, log_var_q, mu_p, log_var_p, alpha=1.5):
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0")

        if abs(alpha - 1.0) < 1e-2:
            return self._kl_divergence(mu_q, log_var_q, mu_p, log_var_p)
        
        var_q = torch.exp(log_var_q)  
        var_p = torch.exp(log_var_p)  
        mean_diff_squared = (mu_q - mu_p) ** 2
        
        combined_var = alpha * var_p + (1 - alpha) * var_q
        combined_var = torch.clamp(combined_var, min=1e-8)  

        numerator = torch.sqrt(var_q)**alpha * torch.sqrt(var_p)**(1 - alpha)
        denominator = torch.sqrt(combined_var)

        exponent = -alpha * (1 - alpha) * mean_diff_squared / (2 * combined_var)
        exp_term = torch.exp(exponent)

        fraction = (numerator / denominator) * exp_term

        coefficient = 1.0 / (alpha * (1 - alpha))
        renyi_div = coefficient * (1 - torch.clamp(fraction, max=1.0 - 1e-8))
        
        return torch.sum(renyi_div)  
    
    def divergence(self, alpha=1.0):

        if alpha == 1:
            # KL divergence case
            return self._kl_divergence(
                self.weight_mu, self.weight_log_var, 
                self.weight_prior_mu, self.weight_prior_log_var
            ) + self._kl_divergence(
                self.bias_mu, self.bias_log_var, 
                self.bias_prior_mu, self.bias_prior_log_var
            )
        else:
            # Rényi divergence case
            return self._renyi_divergence(
                self.weight_mu, self.weight_log_var, 
                self.weight_prior_mu, self.weight_prior_log_var, alpha
            ) + self._renyi_divergence(
                self.bias_mu, self.bias_log_var, 
                self.bias_prior_mu, self.bias_prior_log_var, alpha
            )
    
    def update_prior(self):
        """Update the prior to the current variational posterior"""
        self.weight_prior_mu.data.copy_(self.weight_mu.data)
        self.weight_prior_log_var.data.copy_(self.weight_log_var.data)
        self.bias_prior_mu.data.copy_(self.bias_mu.data)
        self.bias_prior_log_var.data.copy_(self.bias_log_var.data)


class BayesianMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.fc1 = BayesianLinear(input_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, hidden_size)
        self.fc3 = BayesianLinear(hidden_size, output_size)
        
    def forward(self, x, sample=True):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = self.fc3(x, sample)
        return x
    
    def divergence(self, alpha=1.0):
        """Calculate total divergence across all layers"""
        return (self.fc1.divergence(alpha) + 
                self.fc2.divergence(alpha) + 
                self.fc3.divergence(alpha))
    
    def update_priors(self):
        """Update priors for all layers"""
        self.fc1.update_prior()
        self.fc2.update_prior()
        self.fc3.update_prior()

# doesn't work
class BayesianMultiHeadMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=2, num_tasks=5):
        super().__init__()

        self.fc1 = BayesianLinear(input_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, hidden_size)

        self.heads = nn.ModuleList([
            BayesianLinear(hidden_size, output_size) for _ in range(num_tasks)
        ])
        
        self.num_tasks = num_tasks
        self.current_task = 0
        
    def forward(self, x, task=None, sample=True):
        if task is None:
            task = self.current_task
            
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = self.heads[task](x, sample)
        return x
    
    def divergence(self, alpha=1.0):
        div = self.fc1.divergence(alpha) + self.fc2.divergence(alpha)
        div += self.heads[self.current_task].divergence(alpha)
        return div
    
    def update_priors(self):
        self.fc1.update_prior()
        self.fc2.update_prior()
        self.heads[self.current_task].update_prior()
        
    def set_task(self, task_id):
        assert 0 <= task_id < self.num_tasks
        self.current_task = task_id

class CoresetManager:
    def __init__(self, selection_method='random', coreset_size=200):
        self.selection_method = selection_method
        self.coreset_size = coreset_size
        self.coresets = []  
        
    def select_coreset(self, dataset, task_id):

        if self.selection_method == 'random':
            return self._random_selection(dataset)
        elif self.selection_method == 'k-center':
            return self._k_center_selection(dataset)
        else:
            raise ValueError(f"Unknown coreset selection method: {self.selection_method}")
    
    def _random_selection(self, dataset):

        indices = np.random.choice(
            len(dataset), size=min(self.coreset_size, len(dataset)), replace=False
        )
        return Subset(dataset, indices)
    
    def _k_center_selection(self, dataset):

        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for data, _ in data_loader:
            X = data.view(data.size(0), -1).cpu().numpy()
            break

        n_centers = min(self.coreset_size, len(dataset))
        center_indices = [np.random.randint(0, len(dataset))]
        centers = X[center_indices]

        for _ in range(1, n_centers):
            distances, _ = pairwise_distances_argmin_min(X, centers)
            new_center_idx = np.argmax(distances)
            center_indices.append(new_center_idx)
            centers = np.vstack([centers, X[new_center_idx]])
            
        return Subset(dataset, center_indices)
    
    def add_coreset(self, dataset, task_id):

        coreset = self.select_coreset(dataset, task_id)
        self.coresets.append(coreset)
        
    def get_coreset(self, task_id=None):
        if task_id is not None:
            return self.coresets[task_id]

        if not self.coresets:
            return None

        return torch.utils.data.ConcatDataset(self.coresets)


# Task Datasets
def get_permuted_mnist(task_id, permute_seed=42):
    """Create a permuted MNIST dataset for the given task ID"""
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

def get_split_mnist(task_id):

    digits = [2*task_id, 2*task_id+1]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Filter to only include the current pair of digits
    train_indices = [i for i, (_, y) in enumerate(train_dataset) if y in digits]
    test_indices = [i for i, (_, y) in enumerate(test_dataset) if y in digits]
    
    # Relabel to 0 and 1
    train_dataset.targets = torch.tensor(
        [0 if train_dataset.targets[i] == digits[0] else 1 for i in train_indices]
    )
    test_dataset.targets = torch.tensor(
        [0 if test_dataset.targets[i] == digits[0] else 1 for i in test_indices]
    )
    
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    
    return train_dataset, test_dataset


# Training and Evaluation Functions with Mixed Precision
def train_vcl(model, train_loader, optimizer, device, coreset_loader=None, alpha=1.0):
    """Train the VCL model on the current task with mixed precision"""
    model.train()
    total_loss = 0
    total_data = 0
    
    # Setup for mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    
    # Train on current task data
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        optimizer.zero_grad()
        
        # Mixed precision training context
        with autocast(enabled=use_amp):
            output = model(data, sample=True)
            ce_loss = F.cross_entropy(output, target)
            div_loss = model.divergence(alpha=alpha)
            
            # Scale the divergence loss by the dataset size as in the paper
            loss = ce_loss + div_loss / len(train_loader.dataset)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_data += batch_size

    if coreset_loader is not None:
        for data, target in tqdm(coreset_loader, desc="Coreset Training"):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                output = model(data, sample=True)
                ce_loss = F.cross_entropy(output, target)
                div_loss = model.divergence(alpha=alpha)

                loss = ce_loss + div_loss / len(coreset_loader.dataset)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_data += batch_size
    
    return total_loss / total_data

def evaluate(model, test_loader, device, task_id=None):
    """Evaluate the model on the test set"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)

            if task_id is not None and hasattr(model, 'set_task'):
                model.set_task(task_id)

            output = model(data, sample=False)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1) # vcl paper used mcmc, bu didn't mention #samples
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    return test_loss, accuracy

def save_checkpoint(model, optimizer, task_id, path):
    torch.save({
        'task_id': task_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def plot_results(task_accuracies, plot_dir, experiment_name):
    plt.figure(figsize=(12, 8))
    
    num_tasks = len(task_accuracies)

    for i in range(num_tasks):
        plt.subplot(2, (num_tasks + 1) // 2, i + 1)
        task_history = [acc[i] if i < len(acc) else None for acc in task_accuracies[:i+1]]
        valid_points = [(j+1, acc) for j, acc in enumerate(task_history) if acc is not None]
        if valid_points:
            x, y = zip(*valid_points)
            plt.plot(x, y, 'o-', label=f'Task {i}')
            plt.xlabel('Tasks Observed')
            plt.ylabel('Accuracy')
            plt.title(f'Task {i} Accuracy')
            plt.ylim(0, 1)
            plt.grid(True)

    plt.subplot(2, (num_tasks + 1) // 2, num_tasks + 1)
    avg_accuracies = []
    for i, accs in enumerate(task_accuracies):
        valid_accs = [acc for acc in accs if acc is not None]
        if valid_accs:
            avg_accuracies.append(sum(valid_accs) / len(valid_accs))
    
    plt.plot(range(1, len(avg_accuracies) + 1), avg_accuracies, 'o-', color='red')
    plt.xlabel('Tasks Observed')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy Across All Tasks')
    plt.ylim(0, 1)
    plt.grid(True)
    
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f"{experiment_name}_accuracies.png"))
    print(f"Plot saved to {os.path.join(plot_dir, f'{experiment_name}_accuracies.png')}")

    plt.savefig(os.path.join(plot_dir, f"{experiment_name}_accuracies.pdf"))

def run_permuted_mnist_experiment(
    num_tasks=5, 
    hidden_size=100, 
    alpha=1.0, 
    use_coreset=False,
    coreset_method='random',
    coreset_size=200,
    batch_size=256,
    num_epochs=100,
    lr=1e-3,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):

    base_dir, model_dir, plot_dir = setup_experiment_dirs(
        "permuted_mnist", alpha, use_coreset, coreset_method
    )
    

    model = BayesianMLP(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*50}")
    print(f"Starting Permuted MNIST experiment:")
    print(f"  - Alpha: {alpha} ({'KL divergence' if alpha == 1 else 'Rényi divergence'})")
    print(f"  - Coreset: {'Yes, ' + coreset_method if use_coreset else 'No'}")
    print(f"  - Device: {device}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"{'='*50}\n")
    
    # coreset manager if needed
    coreset_manager = None
    if use_coreset:
        coreset_manager = CoresetManager(
            selection_method=coreset_method, 
            coreset_size=coreset_size
        )

    task_accuracies = []
    results = {
        "config": {
            "experiment": "permuted_mnist",
            "num_tasks": num_tasks,
            "hidden_size": hidden_size,
            "alpha": alpha,
            "use_coreset": use_coreset,
            "coreset_method": coreset_method if use_coreset else None,
            "coreset_size": coreset_size if use_coreset else None,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": lr,
        },
        "task_accuracies": [],
        "avg_accuracies": []
    }
    
    for task_id in range(num_tasks):
        print(f"\n{'='*20} Training Task {task_id} {'='*20}")

        train_dataset, test_dataset = get_permuted_mnist(task_id)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True if device.type=='cuda' else False,
                                num_workers=4 if device.type=='cuda' else 0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                pin_memory=True if device.type=='cuda' else False,
                                num_workers=4 if device.type=='cuda' else 0)

        coreset_loader = None
        if use_coreset and task_id > 0:
            combined_coreset = coreset_manager.get_coreset()
            if combined_coreset:
                coreset_loader = DataLoader(
                    combined_coreset, 
                    batch_size=min(batch_size, len(combined_coreset)), 
                    shuffle=True,
                    pin_memory=True if device.type=='cuda' else False
                )

        print(f"Training on task {task_id}...")
        for epoch in range(num_epochs):
            loss = train_vcl(
                model, train_loader, optimizer, device, 
                coreset_loader=coreset_loader, alpha=alpha
            )
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")

        save_checkpoint(
            model, optimizer, task_id, 
            os.path.join(model_dir, f"task_{task_id}_checkpoint.pt")
        )

        model.update_priors()
        if use_coreset:
            print(f"Selecting coreset for task {task_id}...")
            coreset_manager.add_coreset(train_dataset, task_id)
        accuracies = []
        print(f"Evaluating performance on all tasks after learning task {task_id}...")
        for prev_task in range(task_id + 1):
            _, prev_test = get_permuted_mnist(prev_task)
            prev_test_loader = DataLoader(
                prev_test, 
                batch_size=batch_size,
                pin_memory=True if device.type=='cuda' else False,
                num_workers=4 if device.type=='cuda' else 0
            )
            _, acc = evaluate(model, prev_test_loader, device)
            accuracies.append(acc)
            print(f"Task {prev_task} Test Accuracy: {acc:.4f}")
        
        task_accuracies.append(accuracies)
        results["task_accuracies"].append(accuracies)
        results["avg_accuracies"].append(sum(accuracies) / len(accuracies))
        with open(os.path.join(base_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    avg_acc = np.mean([task_accuracies[-1][i] for i in range(num_tasks)])
    print(f"\nFinal Average Accuracy: {avg_acc:.4f}")

    plot_results(task_accuracies, plot_dir, "permuted_mnist")
    
    return task_accuracies, results

def run_split_mnist_experiment(
    num_tasks=5,
    hidden_size=256, 
    alpha=1.0, 
    use_coreset=False,
    coreset_method='random',
    coreset_size=40,
    batch_size=256,
    num_epochs=120, 
    lr=1e-3,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    base_dir, model_dir, plot_dir = setup_experiment_dirs(
        "split_mnist", alpha, use_coreset, coreset_method
    )
    

    model = BayesianMultiHeadMLP(
        hidden_size=hidden_size, output_size=2, num_tasks=num_tasks
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*50}")
    print(f"Starting Split MNIST experiment:")
    print(f"  - Alpha: {alpha} ({'KL divergence' if alpha == 1 else 'Rényi divergence'})")
    print(f"  - Coreset: {'Yes, ' + coreset_method if use_coreset else 'No'}")
    print(f"  - Device: {device}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"{'='*50}\n")
    
    coreset_manager = None
    if use_coreset:
        coreset_manager = CoresetManager(
            selection_method=coreset_method, 
            coreset_size=coreset_size
        )
    
    task_accuracies = []
    results = {
        "config": {
            "experiment": "split_mnist",
            "num_tasks": num_tasks,
            "hidden_size": hidden_size,
            "alpha": alpha,
            "use_coreset": use_coreset,
            "coreset_method": coreset_method if use_coreset else None,
            "coreset_size": coreset_size if use_coreset else None,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": lr,
        },
        "task_accuracies": [],
        "avg_accuracies": []
    }
    
    for task_id in range(num_tasks):
        print(f"\n{'='*20} Training Task {task_id} {'='*20}")
        
        model.set_task(task_id)
        
        train_dataset, test_dataset = get_split_mnist(task_id)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True if device.type=='cuda' else False,
                                num_workers=4 if device.type=='cuda' else 0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                pin_memory=True if device.type=='cuda' else False,
                                num_workers=4 if device.type=='cuda' else 0)
        
        coreset_loader = None
        if use_coreset and task_id > 0:
            combined_coreset = coreset_manager.get_coreset()
            if combined_coreset:
                coreset_loader = DataLoader(
                    combined_coreset, 
                    batch_size=min(batch_size, len(combined_coreset)), 
                    shuffle=True,
                    pin_memory=True if device.type=='cuda' else False
                )
        

        print(f"Training on task {task_id} (digits {2*task_id} vs {2*task_id+1})...")
        for epoch in range(num_epochs):
            loss = train_vcl(
                model, train_loader, optimizer, device, 
                coreset_loader=coreset_loader, alpha=alpha
            )
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")

        save_checkpoint(
            model, optimizer, task_id, 
            os.path.join(model_dir, f"task_{task_id}_checkpoint.pt")
        )

        model.update_priors()

        if use_coreset:
            coreset_manager.add_coreset(train_dataset, task_id)

        accuracies = []
        print(f"Evaluating after learning task {task_id}...")
        for prev_task in range(task_id + 1):
            _, prev_test = get_split_mnist(prev_task)
            prev_test_loader = DataLoader(
                prev_test, 
                batch_size=batch_size,
                pin_memory=True if device.type=='cuda' else False,
                num_workers=4 if device.type=='cuda' else 0
            )
            _, acc = evaluate(model, prev_test_loader, device, task_id=prev_task)
            accuracies.append(acc)
            print(f"Task {prev_task} (digits {2*prev_task} vs {2*prev_task+1}) Test Accuracy: {acc:.4f}")
        
        task_accuracies.append(accuracies)

        results["task_accuracies"].append(accuracies)
        results["avg_accuracies"].append(sum(accuracies) / len(accuracies))

        with open(os.path.join(base_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    avg_acc = np.mean([task_accuracies[-1][i] for i in range(num_tasks)])
    print(f"\nFinal Average Accuracy: {avg_acc:.4f}")

    plot_results(task_accuracies, plot_dir, "split_mnist")
    
    return task_accuracies, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Continual Learning")
    parser.add_argument("--experiment", type=str, default="permuted", 
                        choices=["permuted", "split"], 
                        help="Which experiment to run")
    parser.add_argument("--num_tasks", type=int, default=5, 
                        help="Number of tasks")
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="Divergence parameter alpha. Use alpha=1 for KL, alpha≠1 for Rényi")
    parser.add_argument("--use_coreset", action="store_true", 
                        help="Whether to use coresets")
    parser.add_argument("--coreset_method", type=str, default="random",
                        choices=["random", "k-center"], 
                        help="Coreset selection method")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size for training")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    if use_cuda:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s): {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        
    print(f"Running with alpha={args.alpha} " + 
          f"({'KL divergence' if args.alpha == 1 else 'Rényi divergence'})")
    
    if args.experiment == "permuted":
        run_permuted_mnist_experiment(
            num_tasks=args.num_tasks,
            alpha=args.alpha,
            use_coreset=args.use_coreset,
            coreset_method=args.coreset_method,
            batch_size=args.batch_size,
            device=device
        )
    else:
        run_split_mnist_experiment(
            num_tasks=args.num_tasks,
            alpha=args.alpha,
            use_coreset=args.use_coreset,
            coreset_method=args.coreset_method,
            batch_size=args.batch_size,
            device=device
        )