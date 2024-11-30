import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import kurtosis, skew

# Constants for respiratory rate calculation
LOWCUT = 0.1
HIGHCUT = 0.5

def calculate_heart_rate(ppg_signal, sampling_rate):
    """
    Calculate heart rate from PPG signal.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.
    sampling_rate (int): Sampling rate of the PPG signal.

    Returns:
    dict: Dictionary containing mean heart rate and individual heart rates.
    """
    peaks, _ = find_peaks(ppg_signal, distance=sampling_rate * 0.6)
    if len(peaks) < 2:
        return {'mean_heart_rate': None, 'heart_rates': []}  # Not enough peaks
    rr_intervals = np.diff(peaks) / sampling_rate
    heart_rates = 60 / rr_intervals
    mean_heart_rate = np.mean(heart_rates)
    return {'mean_heart_rate': mean_heart_rate, 'heart_rates': heart_rates}

def calculate_respiratory_rate(ppg_signal, sampling_rate, lowcut=LOWCUT, highcut=HIGHCUT):
    """
    Calculate respiratory rate from PPG signal.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.
    sampling_rate (int): Sampling rate of the PPG signal.
    lowcut (float, optional): Lower cutoff frequency for bandpass filter. Defaults to 0.1.
    highcut (float, optional): Upper cutoff frequency for bandpass filter. Defaults to 0.5.

    Returns:
    float: Respiratory rate in breaths per minute.
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    respiratory_signal = filtfilt(b, a, ppg_signal)
    peaks, _ = find_peaks(respiratory_signal, distance=sampling_rate * 2)
    if len(peaks) < 2:
        return None  # Not enough peaks to calculate respiratory rate
    rr_intervals = np.diff(peaks) / sampling_rate
    respiratory_rate = 60 / np.mean(rr_intervals)
    return respiratory_rate

def calculate_systolic_amplitude(ppg_signal):
    """
    Calculate systolic amplitude from PPG signal.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.

    Returns:
    float: Systolic amplitude.
    """
    systolic_peak = np.max(ppg_signal)
    baseline = np.min(ppg_signal)
    systolic_amplitude = systolic_peak - baseline
    return systolic_amplitude

def calculate_signal_quality_metrics(ppg_signal):
    """
    Calculate signal quality metrics (SNR, kurtosis, skewness).

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.

    Returns:
    tuple: (snr, kurtosis, skewness)
    """
    # Calculate SNR as the ratio of the signal's power to the noise's power
    noise_signal = ppg_signal - np.mean(ppg_signal)  # Simple noise estimation (mean removal)
    snr = np.mean(ppg_signal**2) / np.mean(noise_signal**2) if np.mean(noise_signal**2) > 0 else 0
    kurt = kurtosis(ppg_signal)
    skewness = skew(ppg_signal)
    return snr, kurt, skewness

def extract_features(ppg_signal, sampling_rate):
    """
    Extract features from PPG signal.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.
    sampling_rate (int): Sampling rate of the PPG signal.

    Returns:
    dict: Dictionary containing extracted features.
    """
    # Extract heart rate
    heart_rate_data = calculate_heart_rate(ppg_signal, sampling_rate)
    
    # Extract respiratory rate
    respiratory_rate = calculate_respiratory_rate(ppg_signal, sampling_rate)
    
    # Extract systolic amplitude
    systolic_amplitude = calculate_systolic_amplitude(ppg_signal)
    
    # Extract signal quality metrics
    snr, kurt, skewness = calculate_signal_quality_metrics(ppg_signal)
    
    # Return features as a dictionary
    return {
        'Heart Rate (BPM)': heart_rate_data['mean_heart_rate'],
        'Respiratory Rate (breaths/min)': respiratory_rate,
        'Systolic Amplitude': systolic_amplitude,
        'SNR': snr,
        'Kurtosis': kurt,
        'Skewness': skewness
    }
