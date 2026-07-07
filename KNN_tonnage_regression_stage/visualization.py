# file: visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distributions_by_class(df, categorical_feature, target_variable, class_mapping, output_path=None):
    """
    Generates and optionally saves a plot of the target distributions by class.
    """
    plt.figure(figsize=(12, 8))
    
    inverse_class_mapping = {v: k for k, v in class_mapping.items()}
    
    sns.kdeplot(data=df, x=target_variable, hue=categorical_feature, 
                palette='viridis', fill=True, common_norm=False)

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [inverse_class_mapping.get(int(label), label) for label in labels]
    plt.legend(handles, new_labels, title='Class Name')

    plt.title(f'Distribution of "{target_variable}" by Class', fontsize=16)
    plt.xlabel(target_variable, fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    
    plt.show()
    
def create_heatmap(matrix, class_mapping, metric, output_dir='results/heatmaps'):
    inverse_mapping = {v: k for k, v in class_mapping.items()}
    labels = [inverse_mapping[i] for i in sorted(inverse_mapping)]
    plt.figure(figsize=(12, 10))
    
    heatmap = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap='viridis',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Heatmap: ' + metric, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_dir + '/' + metric + ".png", dpi=300)

