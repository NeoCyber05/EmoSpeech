"""Audio Feature Extractor - Base module for all dataset configs"""

import librosa
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract audio features for emotion recognition"""
    
    def __init__(
        self,
        sr: int = 22050,          # Sample rate
        duration: float = None,    # Max duration (None = full)
        n_mfcc: int = 40,         # Number of MFCCs
        n_mels: int = 128,        # Number of mel bands
        n_fft: int = 2048,        # FFT window size
        hop_length: int = 512,    # Hop length
    ):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
        return y
    
    @staticmethod
    def _stats(arr: np.ndarray, name: str) -> Dict[str, float]:
        """Compute statistics for a feature array"""
        x = arr.flatten()
        return {
            f'{name}_mean': np.mean(x), f'{name}_std': np.std(x),
            f'{name}_min': np.min(x), f'{name}_max': np.max(x),
            f'{name}_median': np.median(x),
            f'{name}_skew': float(np.nan_to_num(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-10))),
            f'{name}_kurtosis': float(np.nan_to_num(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-10) - 3)),
            f'{name}_q25': np.percentile(x, 25), f'{name}_q75': np.percentile(x, 75),
        }
    
    # === TIME-DOMAIN ===
    def time_domain(self, y: np.ndarray) -> Dict[str, float]:
        f = {}
        f.update(self._stats(librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length), 'zcr'))
        f.update(self._stats(librosa.feature.rms(y=y, hop_length=self.hop_length), 'rms'))
        frames = librosa.util.frame(y, frame_length=self.n_fft, hop_length=self.hop_length)
        f.update(self._stats(np.max(np.abs(frames), axis=0), 'amplitude_envelope'))
        return f
    
    # === SPECTRAL ===
    def spectral(self, y: np.ndarray) -> Dict[str, float]:
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
    
    # === MFCC ===
    def mfcc(self, y: np.ndarray, delta: bool = True) -> Dict[str, float]:
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
    
    # === CHROMA ===
    def chroma(self, y: np.ndarray) -> Dict[str, float]:
        f = {}
        for name, func in [('chroma_stft', librosa.feature.chroma_stft),
                           ('chroma_cqt', librosa.feature.chroma_cqt),
                           ('chroma_cens', librosa.feature.chroma_cens)]:
            c = func(y=y, sr=self.sr, hop_length=self.hop_length) if 'stft' not in name else \
                func(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            for i in range(c.shape[0]):
                f.update(self._stats(c[i], f'{name}_{i+1}'))
        tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
        for i in range(tonnetz.shape[0]):
            f.update(self._stats(tonnetz[i], f'tonnetz_{i+1}'))
        return f
    
    # === RHYTHM ===
    def rhythm(self, y: np.ndarray) -> Dict[str, float]:
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
    
    # === PITCH/F0 ===
    def pitch(self, y: np.ndarray, use_pyin: bool = False) -> Dict[str, float]:
        """Extract pitch features. use_pyin=False (default) uses yin (faster), True uses pyin (slower but more accurate)"""
        try:
            fmin, fmax = librosa.note_to_hz('C2'), librosa.note_to_hz('C7')
            if use_pyin:
                # pyin: probabilistic, more accurate but SLOW
                f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=self.sr, hop_length=self.hop_length)
                f0_voiced = f0[~np.isnan(f0)]
            else:
                # yin: faster, good for large datasets
                f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=self.sr, hop_length=self.hop_length)
                f0_voiced = f0[(f0 >= fmin) & (f0 <= fmax)]  # Filter valid range
            
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
    
    # === MEL SPECTROGRAM ===
    def mel(self, y: np.ndarray, n_bands: int = 40) -> Dict[str, float]:
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length), ref=np.max)
        f = {}
        for i in range(min(self.n_mels, n_bands)):
            f[f'mel_{i+1}_mean'], f[f'mel_{i+1}_std'] = np.mean(mel_spec[i]), np.std(mel_spec[i])
        f.update(self._stats(mel_spec, 'mel_spec'))
        return f
    
    # === VOICE QUALITY ===
    def voice_quality(self, y: np.ndarray) -> Dict[str, float]:
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
    
    # === PROSODIC ===
    def prosodic(self, y: np.ndarray) -> Dict[str, float]:
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_norm = rms / (np.sum(rms) + 1e-10)
        return {
            'duration': len(y) / self.sr,
            'energy_range': np.ptp(rms),
            'energy_dynamic_range': 20 * np.log10((np.max(rms) + 1e-10) / (np.min(rms) + 1e-10)),
            'silence_ratio': np.sum(rms < 0.01 * np.max(rms)) / len(rms),
            'energy_entropy': -np.sum(rms_norm * np.log2(rms_norm + 1e-10))
        }
    
    # === MAIN EXTRACTION ===
    def extract(
        self, path: str,
        time_domain: bool = True, spectral: bool = True,
        mfcc: bool = True, mfcc_delta: bool = True,
        chroma: bool = True, rhythm: bool = True,
        pitch: bool = True, mel: bool = True,
        voice_quality: bool = True, prosodic: bool = True,
        use_pyin: bool = False,  # False=yin (fast), True=pyin (slow but accurate)
        metadata: Dict = None
    ) -> Optional[Dict]:
        try:
            y = self.load_audio(path)
            f = {'filename': path}
            if time_domain: f.update(self.time_domain(y))
            if spectral: f.update(self.spectral(y))
            if mfcc: f.update(self.mfcc(y, delta=mfcc_delta))
            if chroma: f.update(self.chroma(y))
            if rhythm: f.update(self.rhythm(y))
            if pitch: f.update(self.pitch(y, use_pyin=use_pyin))
            if mel: f.update(self.mel(y))
            if voice_quality: f.update(self.voice_quality(y))
            if prosodic: f.update(self.prosodic(y))
            if metadata: f.update(metadata)
            return f
        except Exception as e:
            print(f"Error: {path}: {e}")
            return None
    
    def count_features(self, **kwargs) -> int:
        """Count total features based on config"""
        stats = 9  # mean, std, min, max, median, skew, kurtosis, q25, q75
        count = 0
        if kwargs.get('time_domain', True): count += 3 * stats  # zcr, rms, amplitude
        if kwargs.get('spectral', True): count += 6 * stats + 7 * stats  # 6 features + 7 contrast bands
        if kwargs.get('mfcc', True):
            count += self.n_mfcc * stats
            if kwargs.get('mfcc_delta', True): count += self.n_mfcc * 2 * stats
        if kwargs.get('chroma', True): count += 3 * 12 * stats + 6 * stats  # 3 chroma types + tonnetz
        if kwargs.get('rhythm', True): count += 4 + stats  # tempo, beat stats + tempogram
        if kwargs.get('pitch', True): count += 7  # f0 stats
        if kwargs.get('mel', True): count += min(self.n_mels, 40) * 2 + stats  # mel bands + overall
        if kwargs.get('voice_quality', True): count += 2 + 2 * stats  # hnr, ratio + harmonic/percussive
        if kwargs.get('prosodic', True): count += 5  # duration, energy stats
        return count


if __name__ == "__main__":
    ext = AudioFeatureExtractor(n_mfcc=40)
    print(f"Total features: {ext.count_features()}")
    print("\nBy group:")
    base = {k: False for k in ['time_domain', 'spectral', 'mfcc', 'mfcc_delta', 'chroma', 'rhythm', 'pitch', 'mel', 'voice_quality', 'prosodic']}
    for name in base.keys():
        cfg = {**base, name: True}
        if name == 'mfcc': cfg['mfcc_delta'] = True
        print(f"  {name}: {ext.count_features(**cfg)}")
