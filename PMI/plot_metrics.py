import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Define the x-axis labels
x_labels = ['PMI', 'Post Loss', 'Base Loss', 'Smooth', 'Post Acc', 'Base Acc']

def plot_metrics(data_lines):
    # Create figure with subplots
    # plt.style.use('seaborn')
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Performance Metrics Across Different Methods', fontsize=16, y=0.95)

    # Helper function to plot with different line styles for different parameters
    def plot_metric(ax, data, title, ylabel):
        markers = ['o-', 's-', '^-']
        params = ['1', '5', '9']
        epochs = ['30', '300', '3000']
        
        for i, param in enumerate(params):
            for j, epoch in enumerate(epochs):
                # Find rows with matching parameters
                rows = [row for row in data if f'${param}, {epoch}, 400$' in row]
                if rows:
                    # Extract values from the row (skip first two columns)
                    parts = rows[0].split('&')
                    values = [float(v.strip(' \\')) for v in parts[2:8]]
                    
                    label = f'param={param}, epoch={epoch}'
                    ax.plot(range(6), values, markers[i], alpha=0.6, label=label, markersize=4)
        
        ax.set_title(title)
        ax.set_xlabel('Method')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(6))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Extract data for each metric
    metrics = {
        'Accuracy': [row for row in data_lines if row.startswith('Acc &')],
        'Average Accuracy': [row for row in data_lines if row.startswith('Aver. Acc &')],
        'Loss': [row for row in data_lines if row.startswith('Loss &')],
        'Average Loss': [row for row in data_lines if row.startswith('Aver. Loss &')],
        'Noise Ratio': [row for row in data_lines if row.startswith('Noise Ratio &')],
        'Label MAE': [row for row in data_lines if row.startswith('Label MAE &')],
        'Bias MAE': [row for row in data_lines if row.startswith('Bias MAE &')]
    }

    # Plot each metric
    plot_metric(axs[0,0], metrics['Accuracy'], 'Accuracy', 'Accuracy')
    plot_metric(axs[0,1], metrics['Average Accuracy'], 'Average Accuracy', 'Average Accuracy')
    plot_metric(axs[1,0], metrics['Loss'], 'Loss', 'Loss')
    plot_metric(axs[1,1], metrics['Average Loss'], 'Average Loss', 'Average Loss')
    plot_metric(axs[2,0], metrics['Noise Ratio'], 'Noise Ratio', 'Ratio')
    plot_metric(axs[2,1], metrics['Label MAE'], 'Label MAE', 'MAE')
    plot_metric(axs[3,0], metrics['Bias MAE'], 'Bias MAE', 'MAE')
    axs[3,1].remove()  # Remove the empty subplot

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Read the data file
    data_lines = read_data('results.txt')
    # Create the plots
    plot_metrics(data_lines)