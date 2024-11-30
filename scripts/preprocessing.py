import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def load_data(file_path):
    """
    Load data from file.

    Parameters:
    file_path (str): Path to the data file.

    Returns:
    tuple: (data, sampling_rate, duration)
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        
        # Extract Sampling Rate and Duration
        try:
            sampling_rate = int(first_line.split('Sampling Rate : ')[1].split('Hz')[0].strip())
        except IndexError:
            raise ValueError("Unable to extract Sampling Rate from the file.")
        
        try:
            duration = float(first_line.split('Duration : ')[1].split()[0].strip())
        except IndexError:
            raise ValueError("Unable to extract Duration from the file.")
        
        # Load the data into DataFrame
        data = pd.read_csv(f, header=None, names=['Time', 'PPG'])
        data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
        data['PPG'] = pd.to_numeric(data['PPG'], errors='coerce')
        
        # Drop rows with NaN values
        data = data.dropna(subset=['Time', 'PPG'])
    
    return data, sampling_rate, duration

def preprocess_signal(df, sampling_rate, cutoff=0.5):
    """
    Preprocess PPG signal.

    Parameters:
    df (pandas DataFrame): Data containing PPG signal.
    sampling_rate (int): Sampling rate of the PPG signal.
    cutoff (float, optional): Cutoff frequency for low-pass filter. Defaults to 0.5.

    Returns:
    numpy array: Preprocessed PPG signal.
    """
    ppg_signal = df['PPG'].values
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    
    # Apply low-pass filter
    b, a = butter(5, normal_cutoff, btype='low')
    filtered_signal = filtfilt(b, a, ppg_signal)
    
    return filtered_signal

def correct_ppg_signal(ppg_signal):
    """
    Correct and normalize PPG signal.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.

    Returns:
    numpy array: Corrected and normalized PPG signal.
    """
    min_ppg = ppg_signal.min()
    
    # Correct signal if the minimum value is less than 0
    if min_ppg < 0:
        ppg_signal = ppg_signal - min_ppg
    
    # Normalize signal and handle division by zero
    signal_range = ppg_signal.max() - ppg_signal.min()
    if signal_range == 0:
        normalized_ppg = np.zeros_like(ppg_signal)  # If range is zero, return an array of zeros
    else:
        normalized_ppg = (ppg_signal - ppg_signal.min()) / signal_range
    
    return normalized_ppg
