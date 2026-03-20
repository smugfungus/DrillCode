import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import math

# Evita problemas de cache corrompido do Matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig_")

# ========================================
# CONFIGURAÇÕES
# ========================================
FOLDERPATH = "data/standardized/01"
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

    return holes

# ========================================
# MAIN: PLOT ÚNICO COM TODOS ARQUIVOS
# ========================================
def main():
    files = [f for f in os.listdir(FOLDERPATH) if "common" in f and "int" in f and f.endswith(".wav")]
    n_files = len(files)
    if n_files == 0:
        print("Nenhum arquivo encontrado.")
        return

    # Define grid de subplots
    n_cols = 2  # você pode ajustar
    n_rows = math.ceil(n_files / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # para iterar facilmente

    for ax, filename in zip(axes, files):
        full_path = os.path.join(FOLDERPATH, filename)
        y, sr = librosa.load(full_path, sr=None, mono=True)
        holes = segment_holes(y, sr)

        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='magma', ax=ax)
        ax.set_title(filename, fontsize=10)

        # Marcar furos
        for start, end in holes:
            ax.axvspan(start / sr, end / sr, color='red', alpha=0.3)

    # Remove eixos extras se houver mais subplots que arquivos
    for i in range(n_files, len(axes)):
        fig.delaxes(axes[i])

    fig.colorbar(img, ax=axes, format='%+2.0f dB', location='right')
    plt.tight_layout()
    plt.show()  # ou plt.savefig("todos_spectrogramas.png", dpi=200)

if __name__ == "__main__":
    main()
