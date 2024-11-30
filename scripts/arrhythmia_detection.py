import numpy as np
from scipy.signal import find_peaks
# from feature_extraction import calculate_respiratory_rate, calculate_systolic_amplitude

def calculate_rr_intervals(ppg_signal, sampling_rate, height_threshold=0.5, min_distance_factor=0.4):
    """
    Calculate RR intervals from PPG signal.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.
    sampling_rate (int): Sampling rate of the PPG signal.
    height_threshold (float, optional): Minimum peak height threshold. Defaults to 0.5.
    min_distance_factor (float, optional): Factor to determine minimum distance between peaks. Defaults to 0.4.

    Returns:
    tuple: (rr_intervals (numpy array), peaks (numpy array))
    """
    # Find peaks with height threshold and distance between consecutive peaks
    peaks, _ = find_peaks(ppg_signal, height=height_threshold, distance=int(sampling_rate * min_distance_factor))
    
    # Calculate RR intervals (in seconds), only if more than 1 peak exists
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / sampling_rate
    else:
        rr_intervals = np.array([])  # Return an empty array if not enough peaks
    
    return rr_intervals, peaks

def calculate_heart_rate(rr_intervals):
    """
    Calculate the average heart rate (in beats per minute) from RR intervals.

    Parameters:
    rr_intervals (numpy array): RR intervals in seconds.

    Returns:
    float: Heart rate in BPM.
    """
    # Avoid division by zero if no RR intervals are provided
    if len(rr_intervals) == 0:
        return 0
    
    # Calculate the mean RR interval and convert it to BPM
    mean_rr_interval = np.mean(rr_intervals)
    heart_rate = 60 / mean_rr_interval  # Heart rate = 60 / average RR interval (in seconds)
    
    return heart_rate

def detect_irregular_heart_rate(rr_intervals, threshold=0.15):
    """
    Detect irregular heart rate based on RR intervals.

    Parameters:
    rr_intervals (numpy array): RR intervals in seconds.
    threshold (float, optional): Threshold for irregularity detection. Defaults to 0.15.

    Returns:
    bool: True if irregular heart rate is detected, False otherwise.
    """
    # Calculate the standard deviation of RR intervals (Heart Rate Variability)
    if len(rr_intervals) == 0:
        return False  # Return False if no RR intervals available
    
    hrv = np.std(rr_intervals)
    
    # If HRV exceeds the threshold, consider it irregular
    return hrv > threshold

def detect_abnormal_waveform(ppg_signal, threshold=0.2):
    """
    Detect abnormal waveform in PPG signal based on peak-to-valley differences.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.
    threshold (float, optional): Threshold for abnormality detection. Defaults to 0.2.

    Returns:
    bool: True if abnormal waveform is detected, False otherwise.
    """
    peaks, _ = find_peaks(ppg_signal)
    valleys, _ = find_peaks(-ppg_signal)
    
    # Ensure there's a matching number of peaks and valleys
    min_len = min(len(peaks), len(valleys))
    if min_len == 0:
        return False
    
    # Calculate peak-to-valley differences
    peak_to_valley_diff = [ppg_signal[peaks[i]] - ppg_signal[valleys[i]] for i in range(min_len)]
    
    # Check if any peak-to-valley difference exceeds the threshold
    return any(diff > threshold for diff in peak_to_valley_diff)

def detect_arrhythmia(ppg_signal, sampling_rate, threshold_heart_rate=0.15, threshold_waveform=0.2):
    """
    Detect arrhythmia in PPG signal.

    Parameters:
    ppg_signal (numpy array): Preprocessed PPG signal.
    sampling_rate (int): Sampling rate of the PPG signal.
    threshold_heart_rate (float, optional): Threshold for irregular heart rate detection. Defaults to 0.15.
    threshold_waveform (float, optional): Threshold for abnormal waveform detection. Defaults to 0.2.

    Returns:
    tuple: (irregular_heart_rate (bool), abnormal_waveform (bool), start_time (float), end_time (float))
    """
    # Calculate RR intervals and peaks from the PPG signal
    rr_intervals, peaks = calculate_rr_intervals(ppg_signal, sampling_rate)
    
    # If there are no RR intervals or peaks, return defaults
    if len(rr_intervals) == 0 or len(peaks) == 0:
        return False, False, None, None
    
    # Calculate heart rate from RR intervals
    heart_rate = calculate_heart_rate(rr_intervals)
    
    # Detect irregular heart rate and abnormal waveform
    irregular_heart_rate = detect_irregular_heart_rate(rr_intervals, threshold_heart_rate)
    abnormal_waveform = detect_abnormal_waveform(ppg_signal, threshold_waveform)
    
    # If irregular heart rate is detected, calculate the start and end time of the segment
    start_time, end_time = None, None
    if irregular_heart_rate and len(peaks) > 1:
        start_time = peaks[0] / sampling_rate  # Time of the first peak
        end_time = peaks[-1] / sampling_rate  # Time of the last peak

    return irregular_heart_rate, abnormal_waveform, start_time, end_time
