#!/usr/bin/env python3
"""
Analyze base64 encoding/decoding benchmark results.

This script performs comprehensive analysis of eval results including:
- Threshold sweep analysis (accuracy vs threshold)
- Similarity distribution analysis
- Performance by data type
- Model comparison across metrics
"""

import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any
import argparse
from inspect_ai.log import read_eval_log


def load_eval_results(logs_dir: str = "base64bench-logs/results") -> Dict[str, List]:
    """
    Load all evaluation results from the logs directory using Inspect AI's log reader.

    Returns:
        Dictionary mapping model names to lists of sample objects
    """
    results = {}

    # Find all .eval files
    pattern = os.path.join(logs_dir, "**", "*.eval")
    eval_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(eval_files)} eval files")

    for eval_file in eval_files:
        try:
            # Load the eval file using Inspect AI's reader
            log = read_eval_log(eval_file)

            # Extract model name from the eval log content
            model_name = "unknown"
            if hasattr(log, 'eval') and hasattr(log.eval, 'model') and log.eval.model:
                model_name = log.eval.model
            else:
                # Fallback to extracting from path if model info not in log
                path_parts = Path(eval_file).parts
                if len(path_parts) >= 3:
                    # Format: logs/provider/model/file.eval
                    provider = path_parts[-3]
                    model = path_parts[-2]
                    model_name = f"{provider}/{model}"

            if log.samples:
                if model_name not in results:
                    results[model_name] = []
                results[model_name].extend(log.samples)
                print(f"  {model_name}: {len(log.samples)} samples")

        except Exception as e:
            print(f"Error loading {eval_file}: {e}")

    return results


def extract_similarity_scores(results: Dict[str, List]) -> pd.DataFrame:
    """
    Extract similarity scores and metadata into a pandas DataFrame.

    Returns:
        DataFrame with columns: model, task, data_type, similarity, passed_at_1_0, input_length
    """
    rows = []

    for model_name, samples in results.items():
        for sample in samples:
            # sample.scores is a dict mapping scorer names to Score objects
            if hasattr(sample, 'scores') and sample.scores:
                # Look through all scorers (usually just one)
                for scorer_name, score_obj in sample.scores.items():
                    # Get task and data type from sample metadata
                    task_type = sample.metadata.get('task', 'unknown') if sample.metadata else 'unknown'
                    data_type = sample.metadata.get('type', 'unknown') if sample.metadata else 'unknown'

                    if hasattr(score_obj, 'metadata') and score_obj.metadata:
                        metadata = score_obj.metadata
                        # Extract key metrics from successful samples
                        similarity = metadata.get('similarity')
                        input_length = metadata.get('input_length', 0)
                        distance = metadata.get('distance', 0)
                    else:
                        # Failed samples (no metadata) - treat as 0.0 similarity to match Inspect's calculation
                        similarity = 0.0
                        input_length = 0
                        distance = float('inf')  # Large distance for failed attempts

                    if similarity is not None:
                        rows.append({
                            'model': model_name,
                            'task': task_type,
                            'data_type': data_type,
                            'similarity': similarity,
                            'passed_at_1_0': similarity >= 1.0,
                            'passed_at_95': similarity >= 0.95,
                            'passed_at_90': similarity >= 0.90,
                            'input_length': input_length,
                            'distance': distance,
                            'sample_id': getattr(sample, 'id', 'unknown'),
                            'score_value': getattr(score_obj, 'value', None),
                            'scorer': scorer_name,
                            'failed_decode': not (hasattr(score_obj, 'metadata') and score_obj.metadata)
                        })

    return pd.DataFrame(rows)


def plot_threshold_sweep(df: pd.DataFrame, save_path: str = None):
    """Plot accuracy vs threshold for each model with zoom-in subplot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    thresholds = np.arange(0.0, 1.01, 0.05)
    # Create zoom thresholds for high-performance range
    thresholds_zoom = np.concatenate([
        np.arange(0.95, 1.0, 0.005),  # 0.95 to 0.995 with high resolution
        [1.0]  # Exactly 1.0
    ])

    # Calculate performance at threshold=1.0 for ordering
    model_performance_at_1_0 = []

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        accuracy_at_1_0 = (model_data['similarity'] >= 1.0).mean()
        model_performance_at_1_0.append((model, accuracy_at_1_0))

    # Sort models by performance at threshold=1.0 (descending)
    model_performance_at_1_0.sort(key=lambda x: x[1], reverse=True)

    # Function to clean model names (remove provider prefix)
    def clean_model_name(full_name: str) -> str:
        return full_name.split('/')[-1]  # Take everything after the last slash

    # Plot 1: Full range (0.0 - 1.0)
    for model, _ in model_performance_at_1_0:
        model_data = df[df['model'] == model]
        accuracies = []

        for thresh in thresholds:
            accuracy = (model_data['similarity'] >= thresh).mean()
            accuracies.append(accuracy)

        clean_name = clean_model_name(model)
        ax1.plot(thresholds, accuracies, marker='o', markersize=3, label=clean_name, alpha=0.7, linewidth=1)

    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Similarity Threshold (Full Range)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed range (0.8 - 1.0)
    for model, _ in model_performance_at_1_0:
        model_data = df[df['model'] == model]
        accuracies_zoom = []

        for thresh in thresholds_zoom:
            accuracy = (model_data['similarity'] >= thresh).mean()
            accuracies_zoom.append(accuracy)

        clean_name = clean_model_name(model)
        ax2.plot(thresholds_zoom, accuracies_zoom, marker='o', markersize=3, label=clean_name, alpha=0.7, linewidth=1)

    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Similarity Threshold (Zoomed: 0.95-1.0)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.95, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_similarity_distributions(df: pd.DataFrame, save_path: str = None):
    """Plot similarity score distributions for each model."""
    plt.figure(figsize=(15, 10))

    models = df['model'].unique()
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    for i, model in enumerate(models, 1):
        plt.subplot(rows, cols, i)
        model_data = df[df['model'] == model]

        plt.hist(model_data['similarity'], bins=50, alpha=0.7, density=True)
        plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
        plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='100% threshold')

        plt.title(f'{model}\n(n={len(model_data)})')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.xlim(0, 1)

        if i == 1:  # Add legend to first plot
            plt.legend()

    plt.suptitle('Similarity Score Distributions by Model', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_by_data_type(df: pd.DataFrame, threshold: float = 0.95, save_path: str = None):
    """Plot model performance broken down by data type."""
    # Calculate accuracy by model and data type
    accuracy_by_type = df.groupby(['model', 'data_type'])['similarity'].apply(
        lambda x: (x >= threshold).mean()
    ).reset_index()
    accuracy_by_type.columns = ['model', 'data_type', 'accuracy']

    # Pivot for heatmap
    heatmap_data = accuracy_by_type.pivot(index='data_type', columns='model', values='accuracy')

    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f',
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    plt.title(f'Accuracy by Data Type and Model (Threshold: {threshold})')
    plt.ylabel('Data Type')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_by_task(df: pd.DataFrame, threshold: float = 0.95, save_path: str = None):
    """Plot model performance by encoding vs decoding task."""
    accuracy_by_task = df.groupby(['model', 'task'])['similarity'].apply(
        lambda x: (x >= threshold).mean()
    ).reset_index()
    accuracy_by_task.columns = ['model', 'task', 'accuracy']

    # Calculate overall performance for ordering
    model_performance = df.groupby('model')['similarity'].apply(
        lambda x: (x >= 1.0).mean()
    ).reset_index()
    model_performance.columns = ['model', 'overall_accuracy']
    model_performance = model_performance.sort_values('overall_accuracy', ascending=False)

    # Function to clean model names
    def clean_model_name(full_name: str) -> str:
        return full_name.split('/')[-1]

    plt.figure(figsize=(15, 8))

    # Filter and order data
    encode_data = accuracy_by_task[accuracy_by_task['task'] == 'encode']
    decode_data = accuracy_by_task[accuracy_by_task['task'] == 'decode']

    # Merge with performance ordering and clean names
    encode_data = encode_data.merge(model_performance[['model']], on='model', how='inner')
    decode_data = decode_data.merge(model_performance[['model']], on='model', how='inner')

    # Clean model names
    encode_data['clean_model'] = encode_data['model'].apply(clean_model_name)
    decode_data['clean_model'] = decode_data['model'].apply(clean_model_name)

    x = np.arange(len(encode_data))
    width = 0.35

    plt.bar(x - width/2, encode_data['accuracy'], width, label='Encode', alpha=0.7)
    plt.bar(x + width/2, decode_data['accuracy'], width, label='Decode', alpha=0.7)

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(f'Encoding vs Decoding Performance (Threshold: {threshold})')
    plt.xticks(x, encode_data['clean_model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_table(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Generate a summary table of model performance."""
    summary = df.groupby('model').agg({
        'similarity': ['count', 'mean', 'std', 'min', 'max'],
        'passed_at_95': 'mean',
        'passed_at_90': 'mean',
        'input_length': 'mean',
        'distance': 'mean'
    }).round(4)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.rename(columns={
        'similarity_count': 'total_samples',
        'similarity_mean': 'avg_similarity',
        'similarity_std': 'std_similarity',
        'similarity_min': 'min_similarity',
        'similarity_max': 'max_similarity',
        'passed_at_95_mean': 'accuracy_95',
        'passed_at_90_mean': 'accuracy_90',
        'input_length_mean': 'avg_input_length',
        'distance_mean': 'avg_levenshtein_distance'
    })

    # Sort by accuracy at 95% threshold
    summary = summary.sort_values('accuracy_95', ascending=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze base64 benchmark results')
    parser.add_argument('--logs-dir', default='base64bench-logs/results',
                       help='Directory containing eval logs')
    parser.add_argument('--output-dir', default='analysis_plots',
                       help='Directory to save plots')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Primary threshold for analysis')

    args = parser.parse_args()

    print("Loading evaluation results...")
    results = load_eval_results(args.logs_dir)

    print(f"Found results for {len(results)} models:")
    for model, samples in results.items():
        print(f"  {model}: {len(samples)} samples")

    print("\nExtracting similarity scores...")
    df = extract_similarity_scores(results)

    if df.empty:
        print("No data found! Check that eval files contain similarity metadata.")
        return

    print(f"Extracted {len(df)} samples across {df['model'].nunique()} models")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate analyses
    print("\n1. Generating threshold sweep analysis...")
    plot_threshold_sweep(df, os.path.join(args.output_dir, "threshold_sweep.png"))

    print("2. Generating similarity distributions...")
    plot_similarity_distributions(df, os.path.join(args.output_dir, "similarity_distributions.png"))

    print("3. Analyzing performance by data type...")
    plot_performance_by_data_type(df, args.threshold, os.path.join(args.output_dir, "performance_by_type.png"))

    print("4. Analyzing encoding vs decoding performance...")
    plot_performance_by_task(df, args.threshold, os.path.join(args.output_dir, "encode_vs_decode.png"))

    print("5. Generating summary table...")
    summary = generate_summary_table(df, args.threshold)
    summary_path = os.path.join(args.output_dir, "model_summary.csv")
    summary.to_csv(summary_path)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(summary[['total_samples', 'avg_similarity', 'accuracy_95', 'accuracy_90']].to_string())

    print(f"\nAll plots and summary saved to: {args.output_dir}/")
    print("\nFiles generated:")
    print("  - threshold_sweep.png: Accuracy vs threshold curves")
    print("  - similarity_distributions.png: Histogram of similarity scores")
    print("  - performance_by_type.png: Heatmap of accuracy by data type")
    print("  - encode_vs_decode.png: Encoding vs decoding performance")
    print("  - model_summary.csv: Detailed performance statistics")


if __name__ == '__main__':
    # Add required imports to requirements if not already present
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Install with: pip install matplotlib seaborn pandas")
        exit(1)

    main()
