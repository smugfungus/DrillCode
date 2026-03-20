import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile

# Evita problemas de cache corrompido do Matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig_")

# ========================================
# CONFIGURAÇÕES
# ========================================
FILEPATH = "data/standardized/03/03_common_Tr3_ext.wav"  
ENERGY_THRESHOLD = 0.02
MIN_HOLE_DURATION = 0.5  # segundos
SILENCE_GAP = 0.2        # segundos

# ========================================
# FUNÇÃO DE SEGMENTAÇÃO
# ========================================
def segment_holes(y, sr, energy_threshold=ENERGY_THRESHOLD, min_hole_duration=MIN_HOLE_DURATION):
    hop_length = 512
    frame_length = 1024

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_norm = rms / max_rms

    mask = rms_norm > energy_threshold
    holes = []
    start = None

    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            end = i
            start_sample = int(start * hop_length)
            end_sample = int(end * hop_length)
            if (end_sample - start_sample) / sr >= min_hole_duration:
                holes.append((start_sample, end_sample))
            start = None

    if start is not None:
        holes.append((int(start * hop_length), len(y)))

    return holes, rms_norm, hop_length

# ========================================
# MAIN
# ========================================
def main():
    print(f"🔍 Analisando: {FILEPATH}")

    y, sr = librosa.load(FILEPATH, sr=None, mono=True)
    holes, rms_norm, hop_length = segment_holes(y, sr)

    print(f"📈 {len(holes)} furos detectados")
    for i, (start, end) in enumerate(holes, 1):
        print(f"   Furo {i}: {start/sr:.2f}s → {end/sr:.2f}s ({(end-start)/sr:.2f}s)")

    # ========================================
    # PLOT: ESPECTROGRAMA + FUROS DETECTADOS
    # ========================================
    plt.figure(figsize=(15, 6))

    # Espectrograma
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title("Espectrograma com furos detectados")

    # Adiciona retângulos e marca início/fim de cada furo
    for i, (start, end) in enumerate(holes, 1):
        plt.axvspan(start / sr, end / sr, color='red', alpha=0.3)
        plt.axvline(start / sr, color='lime', linestyle='--', linewidth=1.5)  # início
        plt.axvline(end / sr, color='cyan', linestyle='--', linewidth=1.5)   # fim
        # Numeração no meio do furo
        plt.text((start + end) / (2 * sr), sr/2, f'{i}', color='white',
                 fontsize=12, fontweight='bold', ha='center', va='center', alpha=0.8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
