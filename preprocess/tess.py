import librosa
import numpy as np
import pandas as pd

def extract_feature(file_path, emotion=None, actor=None):
    """
    Trích xuất đặc trưng âm thanh từ file audio
    
    Args:
        file_path: Đường dẫn đến file audio
        emotion: Nhãn cảm xúc (optional)
        actor: Diễn viên (optional)
    
    Returns:
        Dictionary chứa các đặc trưng đã trích xuất
    """
    try:
        # Load file audio
        y, sr = librosa.load(file_path, duration=3, sr=22050)
        
        # --- 1. Spectral Features (Đặc trưng phổ) ---
        
        # Chroma
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Mel Spectrogram & MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # RMS (Root Mean Square - Năng lượng)
        rms = librosa.feature.rms(y=y)
        
        # Spectral Descriptors
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_flatness = librosa.feature.spectral_flatness(y=y)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # --- 2. Rhythm Features (Đặc trưng nhịp điệu) ---
        
        # Tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        
        # Tạo dictionary chứa các features
        data = {
            'filename': file_path,
            'tempo': tempo[0] if len(tempo) > 0 else 0,
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'spec_centroid_mean': np.mean(spec_centroid),
            'spec_centroid_std': np.std(spec_centroid),
            'spec_bandwidth_mean': np.mean(spec_bandwidth),
            'spec_bandwidth_std': np.std(spec_bandwidth),
            'spec_rolloff_mean': np.mean(spec_rolloff),
            'spec_rolloff_std': np.std(spec_rolloff),
            'spec_flatness_mean': np.mean(spec_flatness),
            'spec_flatness_std': np.std(spec_flatness),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'tonnetz_mean': np.mean(tonnetz),
            'tonnetz_std': np.std(tonnetz),
        }
        
        # Thêm MFCC features (mean và std cho mỗi coefficient)
        for i in range(mfcc.shape[0]):
            data[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
            data[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        
        # Thêm Chroma features (mean và std)
        for i in range(chroma_stft.shape[0]):
            data[f'chroma_{i+1}_mean'] = np.mean(chroma_stft[i])
            data[f'chroma_{i+1}_std'] = np.std(chroma_stft[i])
        
        # Thêm Spectral Contrast features (mean và std)
        for i in range(spec_contrast.shape[0]):
            data[f'contrast_{i+1}_mean'] = np.mean(spec_contrast[i])
            data[f'contrast_{i+1}_std'] = np.std(spec_contrast[i])
        
        # Thêm thông tin về emotion và actor nếu có
        if emotion is not None:
            data['emotion'] = emotion
        if actor is not None:
            data['actor'] = actor
        
        return data
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        return None