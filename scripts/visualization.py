import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_signal(signal, sampling_rate, title='Filtered PPG Signal', save_path=None):
    """
    Plot PPG signal.

    Parameters:
    signal (numpy array): Preprocessed PPG signal.
    sampling_rate (int): Sampling rate of the PPG signal.
    title (str, optional): Title of the plot. Defaults to 'Filtered PPG Signal'.
    save_path (str, optional): Path to save the plot image. Defaults to None.
    """
    time = np.arange(0, len(signal)) / sampling_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label='Filtered PPG Signal')
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)  # Save the figure if save_path is provided
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()  # Display the plot if no save_path is provided

def plot_heart_rate(heart_rates, save_path=None):
    """
    Plot heart rate over time.

    Parameters:
    heart_rates (numpy array): Heart rates over time.
    save_path (str, optional): Path to save the plot image. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(heart_rates, label='Heart Rate (BPM)')
    plt.title('Heart Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)  # Save the figure if save_path is provided
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()  # Display the plot if no save_path is provided

def plot_feature_distribution(features, save_path=None):
    """
    Plot distribution of extracted features.

    Parameters:
    features (dict): Extracted features.
    save_path (str, optional): Path to save the plot image. Defaults to None.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.bar(features.keys(), features.values())
    plt.title('Distribution of Extracted Features')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    
    if save_path:
        plt.savefig(save_path)  # Save the figure if save_path is provided
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()  # Display the plot if no save_path is provided
