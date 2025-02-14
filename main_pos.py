import numpy as np
import cv2

def pos_algorithm(video_path, fps=20):
    """
    Implementation of Plane-Orthogonal-to-Skin (POS) algorithm for remote PPG.
    
    Args:
        video_path (str): Path to the input video file
        fps (int): Frames per second of the video (default: 20)
    
    Returns:
        numpy.ndarray: The extracted pulse signal
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize parameters
    l = 32  # Window length for 20 fps camera
    H = np.zeros(N)  # Output pulse signal
    
    # Initialize buffer for RGB values
    rgb_buffer = []
    
    # Process each frame
    for n in range(N):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Step 1: Spatial averaging
        rgb_mean = np.mean(frame, axis=(0,1))  # Average RGB values
        C_n = rgb_mean[::-1]  # Convert to BGR (OpenCV) to RGB
        rgb_buffer.append(C_n)
        
        # Check if we have enough frames for temporal processing
        m = n - l + 1
        if m >= 0:
            # Get the window of frames
            C_window = np.array(rgb_buffer[-l:])
            
            # Step 2: Temporal normalization
            C_norm = np.zeros_like(C_window, dtype=float)
            for i in range(3):  # For each color channel
                segment = C_window[:, i]
                C_norm[:, i] = segment / np.mean(segment)
            
            # Step 3: Projection
            P = np.array([[0, 1, -1], [-2, 1, 1]])
            C_n_normalized = C_norm[-1]  # Use the latest normalized frame
            S = np.dot(P, C_n_normalized)
            
            # Step 4: Tuning
            h = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
            
            # Step 5: Overlap-adding
            h_normalized = h - np.mean(h)
            H[m:n+1] += h_normalized
    
    cap.release()
    return H

def process_signal(H, fps=20):
    """
    Post-process the extracted pulse signal.
    
    Args:
        H (numpy.ndarray): Raw pulse signal
        fps (int): Frames per second
    
    Returns:
        tuple: (processed_signal, heart_rate)
    """
    # Apply bandpass filter (0.7-4.0 Hz, typical heart rate range)
    from scipy.signal import butter, filtfilt
    
    nyquist = fps / 2
    low = 0.7 / nyquist
    high = 4.0 / nyquist
    b, a = butter(3, [low, high], btype='band')
    
    # Filter the signal
    processed_signal = filtfilt(b, a, H)
    
    # Calculate heart rate using FFT
    from scipy.fft import fft, fftfreq
    
    Y = fft(processed_signal)
    freq = fftfreq(len(processed_signal), 1/fps)
    
    # Get positive frequencies in HR range
    mask = (freq > 0.7) & (freq < 4.0)
    peaks = np.abs(Y[mask])
    peak_freq = freq[mask][np.argmax(peaks)]
    
    heart_rate = peak_freq * 60  # Convert to BPM
    
    return processed_signal, heart_rate

def extract_heart_rate(video_path, fps=20):
    """
    Extract heart rate from video using POS algorithm.
    
    Args:
        video_path (str): Path to input video
        fps (int): Frames per second
    
    Returns:
        tuple: (heart_rate, pulse_signal)
    """
    # Extract raw pulse signal
    raw_signal = pos_algorithm(video_path, fps)
    
    # Process signal and get heart rate
    processed_signal, heart_rate = process_signal(raw_signal, fps)
    
    return heart_rate, processed_signal

# Example usage
if __name__ == "__main__":
    video_path = "sample_video.mp4"
    heart_rate, pulse_signal = extract_heart_rate(video_path)
    print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
    
    # Optional: Plot the results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.plot(pulse_signal)
    plt.title("Extracted Pulse Signal")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.show()
