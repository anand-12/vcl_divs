# File: make_plots_learning_curves.py

import os
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re # For potentially extracting alpha from dirname if needed

# Define a tolerance for floating point comparisons (e.g., alpha=1.0)
ALPHA_TOLERANCE = 1e-4

def find_results_files(results_dir):
    """Finds all results.json files recursively within the results directory."""
    json_files = []
    for root, dirs, files in os.walk(results_dir):
        if 'results.json' in files:
            # Optional: Add more specific checks based on directory naming if needed
            json_files.append(os.path.join(root, 'results.json'))
    print(f"Found {len(json_files)} 'results.json' files.")
    return json_files

def parse_results(filepath):
    """Parses a single results.json file and extracts key information."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = data.get('config', {})
        avg_accuracies = data.get('avg_accuracies', [])
        task_accuracies = data.get('task_accuracies', [])

        alpha = config.get('alpha')
        if alpha is None:
             print(f"Warning: Could not find 'alpha' in config for {filepath}. Skipping.")
             return None

        experiment = config.get('experiment', 'unknown')
        use_coreset = config.get('use_coreset', False)
        seed = config.get('seed')

        if not avg_accuracies:
            print(f"Warning: 'avg_accuracies' list is empty in {filepath}. Skipping.")
            return None
        
        num_tasks_run = len(avg_accuracies)
        if num_tasks_run == 0:
             print(f"Warning: 'avg_accuracies' list has zero length in {filepath}. Skipping.")
             return None

        final_avg_accuracy = avg_accuracies[-1]

        return {
            'filepath': filepath,
            'alpha': float(alpha), # Ensure it's float
            'experiment': experiment,
            'use_coreset': use_coreset,
            'seed': seed,
            'final_avg_accuracy': final_avg_accuracy,
            'avg_accuracy_curve': avg_accuracies,
            'num_tasks_run': num_tasks_run,
            'task_accuracy_matrix': task_accuracies
        }

    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}. Skipping.")
        return None
    except Exception as e:
        print(f"Error processing file {filepath}: {e}. Skipping.")
        return None

def aggregate_results(parsed_data_list):
    """Aggregates results, calculating mean and std dev for each unique config."""
    # Key: (experiment, use_coreset, alpha)
    # Value: {'final_accuracies': [list], 'avg_curves': [list_of_lists], 'num_tasks': [int]}
    aggregated = defaultdict(lambda: {'final_accuracies': [], 'avg_curves': [], 'num_tasks': []})

    max_tasks_overall = 0

    for data in parsed_data_list:
        if data:
            key = (data['experiment'], data['use_coreset'], data['alpha'])
            aggregated[key]['final_accuracies'].append(data['final_avg_accuracy'])
            aggregated[key]['avg_curves'].append(data['avg_accuracy_curve'])
            aggregated[key]['num_tasks'].append(data['num_tasks_run'])
            if data['num_tasks_run'] > max_tasks_overall:
                max_tasks_overall = data['num_tasks_run']


    processed_results = {}
    for key, values in aggregated.items():
        # Check consistency of num_tasks for this group
        if len(set(values['num_tasks'])) > 1:
            print(f"Warning: Inconsistent number of tasks found for {key}. Runs: {values['num_tasks']}. Using minimum.")
            num_tasks = min(values['num_tasks'])
            # Trim longer curves
            curves_to_avg = [curve[:num_tasks] for curve in values['avg_curves']]
        elif not values['num_tasks']:
             print(f"Warning: No task counts found for {key}. Skipping.")
             continue
        else:
            num_tasks = values['num_tasks'][0]
            curves_to_avg = values['avg_curves']
            # Ensure all curves have the expected length (can happen if a run failed early)
            curves_to_avg = [c for c in curves_to_avg if len(c) == num_tasks]
            if len(curves_to_avg) != len(values['avg_curves']):
                 print(f"Warning: Some runs for {key} did not complete {num_tasks} tasks. Averaging over {len(curves_to_avg)} runs.")
            if not curves_to_avg:
                 print(f"Warning: No valid curves to average for {key} after length check. Skipping.")
                 continue


        final_acc_mean = np.mean(values['final_accuracies'])
        final_acc_std = np.std(values['final_accuracies'])

        try:
            avg_curve_mean = np.mean(np.array(curves_to_avg), axis=0).tolist()
            avg_curve_std = np.std(np.array(curves_to_avg), axis=0).tolist()
        except ValueError as e:
            print(f"Error averaging learning curves for {key}: {e}")
            avg_curve_mean = None
            avg_curve_std = None


        processed_results[key] = {
            'alpha': key[2],
            'final_acc_mean': final_acc_mean,
            'final_acc_std': final_acc_std,
            'num_runs': len(curves_to_avg), # Number of runs successfully averaged
            'num_tasks': num_tasks,
            'avg_curve_mean': avg_curve_mean,
            'avg_curve_std': avg_curve_std
        }

    return processed_results, max_tasks_overall


def plot_learning_curves_vs_alpha(aggregated_results, experiment, use_coreset, max_tasks_overall, output_dir, ylim_bottom=0.0):
    """Plots the average accuracy learning curves (mean) vs. task number for different alphas."""

    plot_data = []
    current_max_tasks = 0
    for key, data in aggregated_results.items():
        if key[0] == experiment and key[1] == use_coreset:
            # Only include if curve averaging was successful
            if data['avg_curve_mean'] is not None:
                plot_data.append((data['alpha'], data['avg_curve_mean'], data['num_runs'], data['num_tasks']))
                if data['num_tasks'] > current_max_tasks:
                    current_max_tasks = data['num_tasks']

    if not plot_data:
        print(f"No valid learning curve data found for {experiment} with use_coreset={use_coreset}. Skipping plot.")
        return
        
    # Use the maximum number of tasks observed for this specific plot, or overall max
    # Let's use current_max_tasks relevant to this specific plot configuration
    if current_max_tasks == 0 :
        print(f"No tasks found for {experiment} with use_coreset={use_coreset}. Skipping plot.")
        return


    # Sort by alpha value for consistent legend ordering and coloring
    plot_data.sort(key=lambda x: x[0])

    plt.figure(figsize=(8, 3)) # Make figure a bit larger for clarity

    # Define a colormap (e.g., 'viridis', 'plasma', 'coolwarm', 'rainbow')
    # 'rainbow' matches the example visually, but 'viridis' or 'plasma' are often preferred
    colormap = plt.cm.rainbow # Or plt.cm.viridis, plt.cm.plasma
    num_alphas = len(plot_data)
    colors = [colormap(i / num_alphas) for i in range(num_alphas)]

    for i, (alpha, curve, num_runs, num_tasks_in_curve) in enumerate(plot_data):
        # X-axis: Task number from 1 to num_tasks_in_curve
        task_numbers = np.arange(1, num_tasks_in_curve + 1)

        # Format label - remove trailing .0 for integer alphas if desired
        alpha_label = f"{alpha:.1f}" if alpha != int(alpha) else str(int(alpha))
        label = f'α = {alpha_label}'
        if num_runs > 1:
            label += f' (n={num_runs})' # Optionally show number of runs averaged

        plt.plot(task_numbers, curve, marker='o', linestyle='-', label=label, color=colors[i], linewidth=2.5)


    plt.xlabel("Number of Tasks Learned", fontsize=14)
    plt.ylabel("Average Accuracy", fontsize=14)
    coreset_str = "with Coreset" if use_coreset else "without Coreset"
    # plt.title(f"{experiment.replace('_', ' ').title()}: Avg Accuracy vs. Tasks for Different Alpha Values ({coreset_str})", fontsize=16)
    plt.title('Average Accuracy', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.6)
    plt.ylim(bottom=ylim_bottom, top=1.05) # Y-axis from 0 to 1 (accuracy)
    plt.xticks(np.arange(1, current_max_tasks + 1)) # Ensure integer ticks for tasks
    plt.xlim(left=0.5, right=current_max_tasks + 0.5)

    # Adjust legend - might need tweaking depending on the number of alphas
    plt.legend(title="Alpha Values", bbox_to_anchor=(1.04, 1), loc="upper left")
    # Alternatively, for fewer lines: plt.legend(title="Alpha Values")

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for external legend

    # Save plot
    filename = f"{experiment}_{'coreset' if use_coreset else 'no_coreset'}_learning_curves_{ylim_bottom}.png"
    output_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(output_path, bbox_inches='tight') # Use bbox_inches='tight' if legend outside was used
        print(f"Saved plot: {output_path}")
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot VCL Alpha-Divergence Learning Curves")
    parser.add_argument("--results_dir", type=str, default="./results_combined",
                        help="Directory containing experiment result subdirectories.")
    parser.add_argument("--output_dir", type=str, default="./plots_summary",
                        help="Directory to save the generated summary plots.")

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Find all result files
    json_files = find_results_files(args.results_dir)

    # 2. Parse each file
    parsed_data_list = [parse_results(f) for f in json_files]
    parsed_data_list = [data for data in parsed_data_list if data is not None]

    if not parsed_data_list:
        print("No valid results data found to plot.")
        exit(0)

    # 3. Aggregate results (calculate mean/std dev per config)
    aggregated_results, max_tasks_overall = aggregate_results(parsed_data_list)

    # 4. Identify unique experiment/coreset combinations found
    configs_found = set((key[0], key[1]) for key in aggregated_results.keys())

    # 5. Generate plots for each combination
    print("\nGenerating learning curve plots...")
    for experiment, use_coreset in sorted(list(configs_found)): # Sort for predictable order
        print(f"  Plotting for: Experiment='{experiment}', Coreset={use_coreset}")
        plot_learning_curves_vs_alpha(aggregated_results, experiment, use_coreset, max_tasks_overall, args.output_dir, ylim_bottom=0.0)
        plot_learning_curves_vs_alpha(aggregated_results, experiment, use_coreset, max_tasks_overall, args.output_dir, ylim_bottom=0.8)

    print("\nPlot generation complete.")
    print(f"Plots saved in: {args.output_dir}")