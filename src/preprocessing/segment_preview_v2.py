import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal

# =========================================================
# PARÂMETROS AJUSTÁVEIS
# =========================================================
MIN_HOLE_DURATION = 2.0       # duração mínima de um furo (s)
SMOOTH_WINDOW = 9             # suavização RMS (frames)
VALLEY_WINDOW_SEC = 1.0       # janela para buscar máximos locais
DEPTH_THRESH = 0.15           # profundidade mínima do vale (0..1)
MIN_PROMINENCE_VALLEY = 0.01  # mínima proeminência bruta do vale
HOP_LENGTH = 512
FRAME_LENGTH = 1024
GROUP_GAP_SEC = 3.0           # distância máxima entre vales para o mesmo furo
MERGE_GAP_SEC = 2.0           # distância máxima entre furos consecutivos para mesclar

# =========================================================
# DETECÇÃO PRINCIPAL
# =========================================================
def detect_holes_by_deep_valleys(y, sr):
    hop_length = HOP_LENGTH
    frame_length = FRAME_LENGTH

    # --- Pré-processamento ---
    y = librosa.effects.preemphasis(y)
    y = y / np.max(np.abs(y))

    # --- Cálculo do RMS e suavização ---
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = np.convolve(rms, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode='same')
    rms_norm = rms_smooth / np.max(rms_smooth)

    # Derivada para diagnóstico
    diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))

    # --- Detecção de vales (mínimos locais) ---
    valleys_all, props = signal.find_peaks(-rms_norm, prominence=MIN_PROMINENCE_VALLEY)
    window_frames = max(1, int((VALLEY_WINDOW_SEC * sr) / hop_length))

    # --- Filtragem por profundidade ---
    valleys_kept = []
    depths = []
    for v in valleys_all:
        left_idx = max(0, v - window_frames)
        right_idx = min(len(rms_norm) - 1, v + window_frames)

        max_before = np.max(rms_norm[left_idx:v+1]) if v - left_idx > 0 else rms_norm[v]
        max_after = np.max(rms_norm[v:right_idx+1]) if right_idx - v > 0 else rms_norm[v]
        local_peak = max(max_before, max_after)

        depth = local_peak - rms_norm[v]
        depths.append(depth)

        if depth >= DEPTH_THRESH:
            valleys_kept.append(v)

    # fallback se poucos vales detectados
    if len(valleys_kept) < 2 and len(valleys_all) >= 2:
        N = min(10, len(valleys_all))
        sorted_idx = np.argsort(depths)[-N:]
        valleys_kept = list(np.array(valleys_all)[sorted_idx])
        valleys_kept.sort()

    # =========================================================
    # AGRUPAMENTO DE VALES EM FUROS (corrigido)
    # =========================================================
    holes = []
    if len(valleys_kept) > 0:
        group_start = valleys_kept[0]
        group_end = valleys_kept[0]
        max_gap = int((GROUP_GAP_SEC * sr) / hop_length)

        for i in range(1, len(valleys_kept)):
            if valleys_kept[i] - valleys_kept[i - 1] > max_gap:
                dur = (group_end - group_start) * hop_length / sr
                if dur >= MIN_HOLE_DURATION:
                    holes.append((group_start * hop_length, group_end * hop_length))
                group_start = valleys_kept[i]
            group_end = valleys_kept[i]

        # adiciona o último grupo
        dur = (group_end - group_start) * hop_length / sr
        if dur >= MIN_HOLE_DURATION:
            holes.append((group_start * hop_length, group_end * hop_length))

    # =========================================================
    # PÓS-PROCESSAMENTO: MESCLAR FUROS MUITO PRÓXIMOS
    # =========================================================
    merged_holes = []
    if len(holes) > 0:
        merged_holes.append(holes[0])
        for s, e in holes[1:]:
            last_s, last_e = merged_holes[-1]
            if (s / sr) - (last_e / sr) < MERGE_GAP_SEC:
                merged_holes[-1] = (last_s, e)
            else:
                merged_holes.append((s, e))
    holes = merged_holes

    return holes, rms_norm, diff, valleys_all, valleys_kept, depths, hop_length


# =========================================================
# PLOTAGENS
# =========================================================
def plot_diagnostics_with_valleys(rms_norm, diff, valleys_all, valleys_kept, depths, hop_length, sr, holes):
    times = np.arange(len(rms_norm)) * hop_length / sr
    plt.figure(figsize=(14, 5))
    plt.plot(times, rms_norm, label='RMS Normalizado', color='royalblue')
    plt.plot(times, diff, label='Variação (|ΔRMS|)', color='orange', alpha=0.8)

    # todos os vales
    if len(valleys_all) > 0:
        t_all = np.array(valleys_all) * hop_length / sr
        plt.vlines(t_all, ymin=0, ymax=1.05, colors='gray', alpha=0.25, linewidth=0.8, label='vales (todos)')

    # vales filtrados
    if len(valleys_kept) > 0:
        t_kept = np.array(valleys_kept) * hop_length / sr
        plt.vlines(t_kept, ymin=0, ymax=1.05, colors='green', alpha=0.9, linewidth=1.2, label='vales profundos (filtrados)')

    # áreas dos furos detectados
    for s, e in holes:
        plt.axvspan(s/sr, e/sr, color='lime', alpha=0.25)

    plt.ylim(-0.02, 1.05)
    plt.xlabel('Tempo (s)')
    plt.title('Diagnóstico: RMS, variação e vales (filtrados)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_holes(y, sr, holes):
    plt.figure(figsize=(14, 6))
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma com furos detectados')

    for i, (s, e) in enumerate(holes, 1):
        plt.axvspan(s/sr, e/sr, color='lime', alpha=0.3)
        plt.axvline(s/sr, color='white', linestyle='--', linewidth=0.7)
        plt.axvline(e/sr, color='white', linestyle='--', linewidth=0.7)
        mid = (s + e) / 2 / sr
        plt.text(mid, sr/4, f'{i}', color='white', ha='center', va='center',
                 fontsize=10, fontweight='bold', alpha=0.9)

    plt.tight_layout()
    plt.show()


# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================
if __name__ == "__main__":
    FILEPATH = "data/standardized/01/01_common_5_int.wav"  # substitua pelo seu arquivo
    # FILEPATH = "data/standardized/03/03_common_Tr3_ext.wav"  # substitua pelo seu arquivo
    y, sr = librosa.load(FILEPATH, sr=None, mono=True)

    holes, rms_norm, diff, valleys_all, valleys_kept, depths, hop_length = detect_holes_by_deep_valleys(y, sr)

    print(f"→ {len(holes)} furos detectados:")
    for i, (s, e) in enumerate(holes, 1):
        print(f"   Furo {i}: {s/sr:.2f}s → {e/sr:.2f}s  (duração {(e-s)/sr:.2f}s)")

    plot_diagnostics_with_valleys(rms_norm, diff, valleys_all, valleys_kept, depths, hop_length, sr, holes)
    plot_spectrogram_with_holes(y, sr, holes)
