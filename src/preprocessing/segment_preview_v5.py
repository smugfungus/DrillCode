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

# parâmetros para detectar início
DROP_PROMINENCE = 0.02        # proeminência mínima da queda (para detectar início com diff)
DROP_SEARCH_SEC = 1.5         # janela antes do vale para buscar início (s)
START_FACTOR = 0.10           # fração do pico local que marca início (ex.: 0.25 = 25%)
START_ABS_THRESH = 0.02       # limite absoluto alternativo (rms normalizado) para início
START_SEARCH_SEC = 20.0        # janela maior para buscar threshold de início (s)


# =========================================================
# DETECÇÃO PRINCIPAL
# =========================================================
def detect_holes_by_deep_valleys(y, sr):
    hop_length = HOP_LENGTH
    frame_length = FRAME_LENGTH
    END_MARGIN_SEC = 4.0  # margem extra no final do furo (s)

    # --- Pré-processamento ---
    y = librosa.effects.preemphasis(y)
    y = y / np.max(np.abs(y))

    # --- Cálculo do RMS e suavização ---
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = np.convolve(rms, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode='same')
    rms_norm = rms_smooth / np.max(rms_smooth)

    # --- Derivada para diagnóstico ---
    diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    if np.max(diff) > 0:
        diff = diff / np.max(diff)

    # --- Detecção de vales (mínimos locais) ---
    valleys_all, props = signal.find_peaks(-rms_norm, prominence=MIN_PROMINENCE_VALLEY)
    window_frames = max(1, int((VALLEY_WINDOW_SEC * sr) / hop_length))

    # --- Filtragem por profundidade ---
    valleys_kept = []
    depths = []
    local_peaks = {}  # map valley_idx -> local_peak (valor RMS do pico usado para depth)
    for v in valleys_all:
        left_idx = max(0, v - window_frames)
        right_idx = min(len(rms_norm) - 1, v + window_frames)

        max_before = np.max(rms_norm[left_idx:v+1]) if v - left_idx > 0 else rms_norm[v]
        max_after = np.max(rms_norm[v:right_idx+1]) if right_idx - v > 0 else rms_norm[v]
        local_peak = max(max_before, max_after)

        depth = local_peak - rms_norm[v]
        depths.append(depth)
        local_peaks[v] = local_peak

        if depth >= DEPTH_THRESH:
            valleys_kept.append(v)

    # fallback se poucos vales detectados
    if len(valleys_kept) < 2 and len(valleys_all) >= 2:
        N = min(10, len(valleys_all))
        sorted_idx = np.argsort(depths)[-N:]
        valleys_kept = list(np.array(valleys_all)[sorted_idx])
        valleys_kept.sort()

    # --- Detecção de quedas abruptas (para ajustar inícios como fallback) ---
    drop_peaks, _ = signal.find_peaks(-diff, prominence=DROP_PROMINENCE)

    # =========================================================
    # AGRUPAMENTO DE VALES EM FUROS
    # =========================================================
    holes = []
    valley_to_localpeak = local_peaks
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

    # =========================================================
    # AJUSTE DOS INÍCIOS E FINS
    # =========================================================
    refined_holes = []
    search_window_drop = int(DROP_SEARCH_SEC * sr / hop_length)
    search_window_start = int(START_SEARCH_SEC * sr / hop_length)
    end_margin_frames = int(END_MARGIN_SEC * sr / hop_length)

    for idx, (s, e) in enumerate(holes):
        s_frame = int(s / hop_length)
        e_frame = int(e / hop_length)

        # valley de referência
        ref_valley = None
        if len(valleys_kept) > 0:
            candidates = [v for v in valleys_kept if v <= s_frame + 2]
            if len(candidates) == 0:
                ref_valley = valleys_kept[np.argmin(np.abs(np.array(valleys_kept) - s_frame))]
            else:
                ref_valley = candidates[-1]

        local_peak = valley_to_localpeak.get(ref_valley, None)
        s_new_frame = None

        # 1) busca threshold relativo
        if local_peak is not None:
            start_threshold = max(START_ABS_THRESH, local_peak * START_FACTOR)
            start_search_left = max(0, s_frame - search_window_start)
            segment = rms_norm[start_search_left:s_frame]
            hits = np.where(segment >= start_threshold)[0]
            if hits.size > 0:
                s_new_frame = start_search_left + hits[0]

        # 2) fallback por quedas abruptas
        if s_new_frame is None:
            candidates_drop = [p for p in drop_peaks if s_frame - search_window_drop <= p < s_frame]
            if candidates_drop:
                s_new_frame = candidates_drop[-1]

        # 3) primeiro furo no início do arquivo → começa do início
        if idx == 0:
            s_new_frame = 0

        if s_new_frame is not None:
            s = int(s_new_frame * hop_length)

        # 4) garantir início após o final do furo anterior
        if len(refined_holes) > 0:
            s = max(s, refined_holes[-1][1])

        # 5) margem de segurança no final
        e_frame_extended = e_frame + end_margin_frames
        e = min(len(y), int(e_frame_extended * hop_length))

        # 6) último furo vai até o final do arquivo
        if idx == len(holes) - 1:
            e = len(y)

        refined_holes.append((s, e))

    holes = refined_holes

    return holes, rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length




# =========================================================
# PLOTAGENS
# =========================================================
def plot_diagnostics_with_valleys(rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length, sr, holes):
    times = np.arange(len(rms_norm)) * hop_length / sr
    plt.figure(figsize=(14, 6))
    plt.plot(times, rms_norm, label='RMS Normalizado')
    plt.plot(times, diff, label='|ΔRMS| (variação)', alpha=0.8)

    # todos os vales
    if len(valleys_all) > 0:
        t_all = np.array(valleys_all) * hop_length / sr
        plt.vlines(t_all, ymin=0, ymax=1.05, colors='gray', alpha=0.25, linewidth=0.8, label='vales (todos)')

    # vales filtrados
    if len(valleys_kept) > 0:
        t_kept = np.array(valleys_kept) * hop_length / sr
        plt.vlines(t_kept, ymin=0, ymax=1.05, colors='green', alpha=0.9, linewidth=1.2, label='vales profundos (filtrados)')

    # quedas detectadas (fallback)
    if len(drop_peaks) > 0:
        t_drops = np.array(drop_peaks) * hop_length / sr
        plt.vlines(t_drops, ymin=0, ymax=1.05, colors='red', alpha=0.4, linewidth=1.0, label='quedas (fallback)')

    # áreas dos furos detectados
    for s, e in holes:
        plt.axvspan(s/sr, e/sr, color='lime', alpha=0.25)

    plt.ylim(-0.02, 1.05)
    plt.xlabel('Tempo (s)')
    plt.title('Diagnóstico: RMS, variação, vales e quedas (ajuste de inícios)')
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

def detect_holes_by_deep_valleys_custom(
    y, sr,
    DEPTH_THRESH=0.15,
    GROUP_GAP_SEC=3.0,
    MERGE_GAP_SEC=2.0
):
    # reaproveita quase tudo da versão original
    global HOP_LENGTH, FRAME_LENGTH, MIN_PROMINENCE_VALLEY, MIN_HOLE_DURATION
    global SMOOTH_WINDOW, VALLEY_WINDOW_SEC, DROP_PROMINENCE, DROP_SEARCH_SEC
    global START_FACTOR, START_ABS_THRESH, START_SEARCH_SEC

    # apenas muda os valores substituídos:
    orig_DEPTH_THRESH = globals().get('DEPTH_THRESH', 0.15)
    orig_GROUP_GAP_SEC = globals().get('GROUP_GAP_SEC', 3.0)
    orig_MERGE_GAP_SEC = globals().get('MERGE_GAP_SEC', 2.0)

    # substitui temporariamente
    globals()['DEPTH_THRESH'] = DEPTH_THRESH
    globals()['GROUP_GAP_SEC'] = GROUP_GAP_SEC
    globals()['MERGE_GAP_SEC'] = MERGE_GAP_SEC

    # executa
    results = detect_holes_by_deep_valleys(y, sr)

    # restaura originais
    globals()['DEPTH_THRESH'] = orig_DEPTH_THRESH
    globals()['GROUP_GAP_SEC'] = orig_GROUP_GAP_SEC
    globals()['MERGE_GAP_SEC'] = orig_MERGE_GAP_SEC

    return results


def refine_long_holes(y, sr, holes, duration_factor=1.5, aggressive_factor=2.5):
    """
    Refina furos anormalmente longos com detecção recursiva mais sensível.
    duration_factor: limite (x mediana) acima do qual o furo é refinado
    aggressive_factor: se muito acima (ex: >2.5× mediana), aplica parâmetros ainda mais finos
    """
    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations)
    refined_holes = []

    for (s, e), dur in zip(holes, durations):
        if dur <= duration_factor * median_dur:
            refined_holes.append((s, e))
            continue

        print(f"→ Refinando furo longo ({dur:.2f}s)...")

        y_seg = y[s:e]
        params = {}

        # Definir sensibilidade de acordo com o tamanho
        if dur > aggressive_factor * median_dur:
            params = dict(
                DEPTH_THRESH=0.05,
                GROUP_GAP_SEC=0.5,
                MERGE_GAP_SEC=0.25
            )
        else:
            params = dict(
                DEPTH_THRESH=0.08,
                GROUP_GAP_SEC=1.0,
                MERGE_GAP_SEC=0.5
            )

        sub_holes, *_ = detect_holes_by_deep_valleys_custom(y_seg, sr, **params)
        sub_holes_global = [(s + s2, s + e2) for (s2, e2) in sub_holes]

        if len(sub_holes_global) == 0:
            refined_holes.append((s, e))
        else:
            refined_holes.extend(sub_holes_global)

    return refined_holes




# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================
if __name__ == "__main__":
    # FILEPATH = "data/standardized/01/01_ultrasonic_250130_013_Tr6_ch1_int.wav"  # substitua pelo seu arquivo
    # FILEPATH = "data/standardized/01/01_ultrasonic_250130_011_Tr2_ch1_int.wav"  # substitua pelo seu arquivo
    # FILEPATH = "data/standardized/07/07_ultrasonic_190514_005_Tr3_ch1_int.wav"  # substitua pelo seu arquivo
    FILEPATH = "data/standardized/07/07_common_Tr5_int.wav"  # substitua pelo seu arquivo
    # FILEPATH = "data/standardized/02/02_common_190512_009_190512_009_int.wav"  # substitua pelo seu arquivo
    y, sr = librosa.load(FILEPATH, sr=None, mono=True)

    holes, rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length = detect_holes_by_deep_valleys(y, sr)

    holes_refined = refine_long_holes(y, sr, holes, duration_factor=1.5, aggressive_factor=2.5)

    print(f"\n→ Após refinamento adaptativo: {len(holes_refined)} furos detectados:")
    for i, (s, e) in enumerate(holes_refined, 1):
        print(f"   Furo {i}: {s/sr:.2f}s → {e/sr:.2f}s  (duração {(e-s)/sr:.2f}s)")

    plot_diagnostics_with_valleys(rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length, sr, holes_refined)
    plot_spectrogram_with_holes(y, sr, holes_refined)

