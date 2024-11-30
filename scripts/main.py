import os
import numpy as np
import pandas as pd
from preprocessing import load_data, preprocess_signal, correct_ppg_signal
from feature_extraction import extract_features, calculate_respiratory_rate, calculate_heart_rate, calculate_systolic_amplitude
from arrhythmia_detection import detect_arrhythmia
from visualization import plot_signal, plot_feature_distribution

# Path to the data folder
data_folder = '../data/'

# List all files in the data folder (excluding PPG-5 dataset)
dataset_files = [f for f in os.listdir(data_folder) if f.startswith('PPG-') and f.endswith('.csv') and f != 'PPG-5.csv']

# Create a results directory if it doesn't exist
results_folder = '../results/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Process each dataset
for dataset_file in dataset_files:
    # Construct full file path
    file_path = os.path.join(data_folder, dataset_file)
    
    # Load data
    print(f"Processing {dataset_file}...")
    data, sampling_rate, duration = load_data(file_path)
    
    # Preprocess PPG signal
    ppg_signal = preprocess_signal(data, sampling_rate)
    
    # Correct and normalize PPG signal
    corrected_ppg_signal = correct_ppg_signal(ppg_signal)
    
    # Extract features
    features = extract_features(corrected_ppg_signal, sampling_rate)
    
    # Detect arrhythmia
    irregular_heart_rate, abnormal_waveform, start_time, end_time = detect_arrhythmia(corrected_ppg_signal, sampling_rate)
    
    # Calculate mean heart rate, respiratory rate, and systolic amplitude
    mean_heart_rate = calculate_heart_rate(corrected_ppg_signal, sampling_rate)
    mean_respiratory_rate = calculate_respiratory_rate(corrected_ppg_signal, sampling_rate)
    mean_systolic_amplitude = calculate_systolic_amplitude(corrected_ppg_signal)
    
    # Print results
    print(f"Dataset Name: {dataset_file}")
    print(f"Mean Heart Rate: {mean_heart_rate} BPM")
    print(f"Mean Respiratory Rate: {mean_respiratory_rate} /minute")
    print(f"Mean Systolic Amplitude: {mean_systolic_amplitude} units")
    print(f"Arrhythmia Detected: {'Yes' if irregular_heart_rate else 'No'}")
    if irregular_heart_rate:
        print(f"  - Segment Time: {start_time} - {end_time}")
    
    # Prepare result data (ensure all values are aligned in lists)
    result_data = {
        'Feature': list(features.keys()) + ['Irregular Heart Rate', 'Abnormal Waveform', 'Mean Heart Rate', 'Mean Respiratory Rate', 'Mean Systolic Amplitude', 'Arrhythmia Detected', 'Segment Time Start', 'Segment Time End'],
        'Value': list(features.values()) + [irregular_heart_rate, abnormal_waveform, mean_heart_rate, mean_respiratory_rate, mean_systolic_amplitude, 'Yes' if irregular_heart_rate else 'No', start_time if irregular_heart_rate else None, end_time if irregular_heart_rate else None]
    }
    
    # Check if lengths of 'Feature' and 'Value' match
    if len(result_data['Feature']) != len(result_data['Value']):
        print(f"Error: The number of features does not match the number of values.")
    else:
        # Create DataFrame and save to CSV
        result_df = pd.DataFrame(result_data)
        result_file = os.path.join(results_folder, f'{dataset_file}_results.csv')
        result_df.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")
    
    # Save the plots (signal and feature distribution)
    signal_plot_path = os.path.join(results_folder, f'{dataset_file}_signal_plot.png')
    plot_signal(corrected_ppg_signal, sampling_rate, save_path=signal_plot_path)
    
    feature_dist_plot_path = os.path.join(results_folder, f'{dataset_file}_feature_distribution.png')
    plot_feature_distribution(features, save_path=feature_dist_plot_path)
    
    print(f"Finished processing {dataset_file}\n")
