import librosa
import numpy as np
import os

def extract_supervised_features(path):
    """
    Extrai features tradicionais para classificação/regressão supervisionada.
    Inclui: RMS, ZCR, centroid, rolloff, bandwidth e 13 MFCCs.
    """
    try:
        if not os.path.exists(path):           
            return None

        y, sr = librosa.load(path, sr=None, mono=True)

        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)

        return [rms, zcr, centroid, rolloff, bandwidth] + list(mfcc_mean)

    except Exception as e:
        print(f"[ERROR] Problema ao processar {path}: {e}")
        return None
