"""
FAST Feature Extraction with Multiprocessing - FULL FEATURES
=============================================================
ALL features + ALL 9 statistics (mean, std, min, max, median, skew, kurtosis, q25, q75)
Includes: chroma_stft, chroma_cqt, chroma_cens, tonnetz

Usage:
    python extract_features_fast.py --workers 8 --format csv
    python extract_features_fast.py --split train
    python extract_features_fast.py  # Extract all splits
"""

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')


# ===================== CONFIG =====================
AUGMENTED_DIR = Path("d:/AI/EmoSpeech/dataset/augmented")
FEATURES_DIR = Path("d:/AI/EmoSpeech/data/features")
SR = 16000
DURATION = 2.0
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160


# ===================== FEATURE EXTRACTOR =====================
class AudioFeatureExtractor:
    """Extract ALL audio features with 9 statistics per feature"""
    
    def __init__(self, sr=16000, duration=2.0, n_mfcc=13, n_mels=40, n_fft=512, hop_length=160):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(self, path):
        y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
        return y
    
    @staticmethod
    def _stats(arr, name):
        """9 statistics: mean, std, min, max, median, skew, kurtosis, q25, q75"""
        x = arr.flatten()
        return {
            f'{name}_mean': np.mean(x), f'{name}_std': np.std(x),
            f'{name}_min': np.min(x), f'{name}_max': np.max(x),
            f'{name}_median': np.median(x),
            f'{name}_skew': float(np.nan_to_num(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-10))),
            f'{name}_kurtosis': float(np.nan_to_num(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-10) - 3)),
            f'{name}_q25': np.percentile(x, 25), f'{name}_q75': np.percentile(x, 75),
        }
    
    def time_domain(self, y):
        f = {}
        f.update(self._stats(librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length), 'zcr'))
        f.update(self._stats(librosa.feature.rms(y=y, hop_length=self.hop_length), 'rms'))
        frames = librosa.util.frame(y, frame_length=self.n_fft, hop_length=self.hop_length)
        f.update(self._stats(np.max(np.abs(frames), axis=0), 'amplitude_envelope'))
        return f
    
    def spectral(self, y):
        f = {}
        kw = {'y': y, 'sr': self.sr, 'n_fft': self.n_fft, 'hop_length': self.hop_length}
        f.update(self._stats(librosa.feature.spectral_centroid(**kw), 'spectral_centroid'))
        f.update(self._stats(librosa.feature.spectral_bandwidth(**kw), 'spectral_bandwidth'))
        for p in [0.85, 0.95]:
            f.update(self._stats(librosa.feature.spectral_rolloff(**kw, roll_percent=p), f'spectral_rolloff_{int(p*100)}'))
        f.update(self._stats(librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length), 'spectral_flatness'))
        contrast = librosa.feature.spectral_contrast(**kw)
        for i in range(contrast.shape[0]):
            f.update(self._stats(contrast[i], f'spectral_contrast_{i+1}'))
        f.update(self._stats(librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length), 'spectral_flux'))
        return f
    
    def mfcc(self, y, delta=True):
        f = {}
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        for i in range(mfcc.shape[0]):
            f.update(self._stats(mfcc[i], f'mfcc_{i+1}'))
        if delta:
            d1 = librosa.feature.delta(mfcc)
            d2 = librosa.feature.delta(mfcc, order=2)
            for i in range(d1.shape[0]):
                f.update(self._stats(d1[i], f'mfcc_delta_{i+1}'))
                f.update(self._stats(d2[i], f'mfcc_delta2_{i+1}'))
        return f
    
    def chroma(self, y):
        """ALL chroma: STFT, CQT, CENS + Tonnetz"""
        f = {}
        c_stft = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        for i in range(c_stft.shape[0]):
            f.update(self._stats(c_stft[i], f'chroma_stft_{i+1}'))
        
        c_cqt = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)
        for i in range(c_cqt.shape[0]):
            f.update(self._stats(c_cqt[i], f'chroma_cqt_{i+1}'))
        
        c_cens = librosa.feature.chroma_cens(y=y, sr=self.sr, hop_length=self.hop_length)
        for i in range(c_cens.shape[0]):
            f.update(self._stats(c_cens[i], f'chroma_cens_{i+1}'))
        
        tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
        for i in range(tonnetz.shape[0]):
            f.update(self._stats(tonnetz[i], f'tonnetz_{i+1}'))
        
        return f
    
    def rhythm(self, y):
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sr)
        f = {'tempo': float(tempo[0]) if len(tempo) > 0 else 0.0}
        try:
            _, beats = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
            intervals = np.diff(beats) if len(beats) > 1 else [0]
            f.update({'beat_count': len(beats), 'beat_interval_mean': np.mean(intervals), 'beat_interval_std': np.std(intervals)})
        except:
            f.update({'beat_count': 0, 'beat_interval_mean': 0.0, 'beat_interval_std': 0.0})
        f.update(self._stats(librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length), 'tempogram'))
        return f
    
    def pitch(self, y, use_pyin=False):
        try:
            fmin, fmax = librosa.note_to_hz('C2'), librosa.note_to_hz('C7')
            if use_pyin:
                f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=self.sr, hop_length=self.hop_length)
                f0_voiced = f0[~np.isnan(f0)]
            else:
                f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=self.sr, hop_length=self.hop_length)
                f0_voiced = f0[(f0 >= fmin) & (f0 <= fmax)]
            
            if len(f0_voiced) > 0:
                return {
                    'f0_mean': np.mean(f0_voiced), 'f0_std': np.std(f0_voiced),
                    'f0_min': np.min(f0_voiced), 'f0_max': np.max(f0_voiced),
                    'f0_range': np.ptp(f0_voiced), 'f0_median': np.median(f0_voiced),
                    'voiced_fraction': len(f0_voiced) / len(f0) if len(f0) > 0 else 0
                }
        except:
            pass
        return {'f0_mean': 0, 'f0_std': 0, 'f0_min': 0, 'f0_max': 0, 'f0_range': 0, 'f0_median': 0, 'voiced_fraction': 0}
    
    def mel(self, y, n_bands=40):
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length), ref=np.max)
        f = {}
        for i in range(min(self.n_mels, n_bands)):
            f[f'mel_{i+1}_mean'], f[f'mel_{i+1}_std'] = np.mean(mel_spec[i]), np.std(mel_spec[i])
        f.update(self._stats(mel_spec, 'mel_spec'))
        return f
    
    def voice_quality(self, y):
        try:
            h, p = librosa.effects.hpss(y)
            h_e, p_e = np.sum(h**2), np.sum(p**2)
            f = {
                'hnr_approx': 10 * np.log10(h_e / (p_e + 1e-10)) if p_e > 0 else 0.0,
                'harmonic_ratio': h_e / (h_e + p_e + 1e-10)
            }
            f.update(self._stats(librosa.feature.rms(y=h), 'harmonic_rms'))
            f.update(self._stats(librosa.feature.rms(y=p), 'percussive_rms'))
            return f
        except:
            return {'hnr_approx': 0, 'harmonic_ratio': 0}
    
    def prosodic(self, y):
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_norm = rms / (np.sum(rms) + 1e-10)
        return {
            'duration': len(y) / self.sr,
            'energy_range': np.ptp(rms),
            'energy_dynamic_range': 20 * np.log10((np.max(rms) + 1e-10) / (np.min(rms) + 1e-10)),
            'silence_ratio': np.sum(rms < 0.01 * np.max(rms)) / len(rms),
            'energy_entropy': -np.sum(rms_norm * np.log2(rms_norm + 1e-10))
        }
    
    def extract(self, path, metadata=None, use_pyin=False):
        try:
            y = self.load_audio(path)
            f = {'filename': path}
            f.update(self.time_domain(y))
            f.update(self.spectral(y))
            f.update(self.mfcc(y, delta=True))
            f.update(self.chroma(y))
            f.update(self.rhythm(y))
            f.update(self.pitch(y, use_pyin=use_pyin))
            f.update(self.mel(y))
            f.update(self.voice_quality(y))
            f.update(self.prosodic(y))
            if metadata:
                f.update(metadata)
            return f
        except Exception as e:
            return None


# ===================== WORKER FUNCTION =====================
def extract_single(args):
    """Worker function for multiprocessing"""
    path, metadata, config = args
    extractor = AudioFeatureExtractor(**config)
    return extractor.extract(path, metadata=metadata, use_pyin=False)


def extract_split(split_name, workers, output_format='csv'):
    """Extract features for a single split (train/val/test)"""
    input_dir = AUGMENTED_DIR / split_name
    config = {'sr': SR, 'duration': DURATION, 'n_mfcc': N_MFCC, 'n_mels': N_MELS, 'n_fft': N_FFT, 'hop_length': HOP_LENGTH}
    
    meta_path = input_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"  [SKIP] {meta_path} not found!")
        return None
    
    df_meta = pd.read_csv(meta_path)
    print(f"\n  {split_name.upper()}: {len(df_meta)} samples")
    
    # Prepare tasks
    tasks = [
        (row['path'], {'emotion': row['emotion'], 'source': row['source'], 'augmentation': row['augmentation']}, config)
        for _, row in df_meta.iterrows()
    ]
    
    # Extract with multiprocessing
    all_features = []
    errors = 0
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_single, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {split_name}"):
            result = future.result()
            if result:
                all_features.append(result)
            else:
                errors += 1
    
    elapsed = time.time() - start
    
    # Create DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Reorder columns
    meta_cols = ['filename', 'emotion', 'source', 'augmentation']
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    df_features = df_features[[c for c in meta_cols if c in df_features.columns] + feature_cols]
    
    # Save to features folder
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_format in ['csv', 'both']:
        csv_path = FEATURES_DIR / f"{split_name}.csv"
        df_features.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    print(f"  Done in {elapsed/60:.1f} min | Errors: {errors}")
    print(f"  By emotion:\n{df_features['emotion'].value_counts().to_string()}")
    
    return len(df_features)


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser(description='Fast Feature Extraction with Multiprocessing')
    parser.add_argument('--workers', '-w', type=int, default=mp.cpu_count(), 
                        help=f'Number of workers (default: {mp.cpu_count()})')
    parser.add_argument('--format', '-f', type=str, default='csv', choices=['csv', 'parquet', 'both'])
    parser.add_argument('--split', '-s', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Which split to extract (default: all)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FAST FEATURE EXTRACTION (Full Features + Multiprocessing)")
    print("=" * 60)
    print(f"  Input: {AUGMENTED_DIR}")
    print(f"  Output: {FEATURES_DIR}")
    print(f"  Workers: {args.workers}")
    print("=" * 60)
    
    # Determine which splits to process
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    total_samples = 0
    for split in splits:
        count = extract_split(split, args.workers, args.format)
        if count:
            total_samples += count
    
    print("\n" + "=" * 60)
    print(f"  ALL DONE! Total: {total_samples} samples")
    print(f"  Output files: {FEATURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
