import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal

# =========================================================
# PARÂMETROS AJUSTÁVEIS
# =========================================================

MIN_HOLE_DURATION = 2.0            # duração mínima de um furo (s)
SMOOTH_WINDOW = 9                  # suavização RMS (frames)
VALLEY_WINDOW_SEC = 1.0            # janela para buscar máximos locais
DEPTH_THRESH = 0.15                # profundidade mínima do vale (0..1)
MIN_PROMINENCE_VALLEY = 0.01       # mínima proeminência bruta do vale
HOP_LENGTH = 512
FRAME_LENGTH = 1024
GROUP_GAP_SEC = 3.0                # distância máxima entre vales para o mesmo furo
MERGE_GAP_SEC = 2.0                # distância máxima entre furos consecutivos para mesclar

# parâmetros para detectar início
DROP_PROMINENCE = 0.02             # proeminência mínima da queda
DROP_SEARCH_SEC = 1.5              # janela antes do vale para buscar início
START_FACTOR = 0.10                # fração do pico local para marcar início
START_ABS_THRESH = 0.02            # limite absoluto alternativo
START_SEARCH_SEC = 20.0            # janela grande para buscar início


# =========================================================
# DETECÇÃO PRINCIPAL
# =========================================================

def detect_holes_by_deep_valleys(y, sr):
    hop_length = HOP_LENGTH
    frame_length = FRAME_LENGTH
    END_MARGIN_SEC = 4.0

    # --- Pré-processamento ---
    if np.max(np.abs(y)) == 0:
        return [], np.array([]), np.array([]), [], [], [], [], hop_length

    y = librosa.effects.preemphasis(y)
    y = y / np.max(np.abs(y))

    # --- Cálculo do RMS ---
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    if np.max(rms) == 0:
        rms_smooth = rms.copy()
    else:
        rms_smooth = np.convolve(
            rms,
            np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
            mode='same'
        )

    if np.max(rms_smooth) > 0:
        rms_norm = rms_smooth / np.max(rms_smooth)
    else:
        rms_norm = rms_smooth

    # --- Derivada ---
    if len(rms_norm) > 0:
        diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    else:
        diff = np.array([])

    if np.max(diff) > 0:
        diff = diff / np.max(diff)

    # --- Detecção de vales ---
    if len(rms_norm) == 0:
        valleys_all = np.array([], dtype=int)
        props = {}
    else:
        valleys_all, props = signal.find_peaks(
            -rms_norm,
            prominence=MIN_PROMINENCE_VALLEY
        )

    window_frames = max(1, int((VALLEY_WINDOW_SEC * sr) / hop_length))

    valleys_kept = []
    depths = []
    local_peaks = {}

    for v in valleys_all:
        left_idx = max(0, v - window_frames)
        right_idx = min(len(rms_norm) - 1, v + window_frames)

        max_before = (
            np.max(rms_norm[left_idx:v + 1])
            if v - left_idx > 0 else rms_norm[v]
        )
        max_after = (
            np.max(rms_norm[v:right_idx + 1])
            if right_idx - v > 0 else rms_norm[v]
        )

        local_peak = max(max_before, max_after)
        depth = local_peak - rms_norm[v]

        depths.append(depth)
        local_peaks[v] = local_peak

        if depth >= DEPTH_THRESH:
            valleys_kept.append(v)

    # fallback se poucos vales
    if len(valleys_kept) < 2 and len(valleys_all) >= 2:
        N = min(10, len(valleys_all))
        sorted_idx = np.argsort(depths)[-N:]
        valleys_kept = list(np.array(valleys_all)[sorted_idx])
        valleys_kept.sort()

    # --- Quedas abruptas (fallback) ---
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
                    holes.append(
                        (group_start * hop_length, group_end * hop_length)
                    )
                group_start = valleys_kept[i]

            group_end = valleys_kept[i]

        dur = (group_end - group_start) * hop_length / sr
        if dur >= MIN_HOLE_DURATION:
            holes.append((group_start * hop_length, group_end * hop_length))

    # =========================================================
    # MESCLAR FUROS PRÓXIMOS
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
    # AJUSTE DE INÍCIOS E FINS
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
                ref_valley = valleys_kept[
                    np.argmin(np.abs(np.array(valleys_kept) - s_frame))
                ]
            else:
                ref_valley = candidates[-1]

        local_peak = valley_to_localpeak.get(ref_valley, None)
        s_new_frame = None

        # 1) threshold relativo
        if local_peak is not None:
            start_threshold = max(
                START_ABS_THRESH,
                local_peak * START_FACTOR
            )

            start_left = max(0, s_frame - search_window_start)
            segment = rms_norm[start_left:s_frame]

            hits = np.where(segment >= start_threshold)[0]
            if hits.size > 0:
                s_new_frame = start_left + hits[0]

        # 2) fallback via quedas
        if s_new_frame is None:
            candidates_drop = [
                p for p in drop_peaks
                if s_frame - search_window_drop <= p < s_frame
            ]
            if candidates_drop:
                s_new_frame = candidates_drop[-1]

        # 3) início do arquivo
        if idx == 0:
            s_new_frame = 0

        if s_new_frame is not None:
            s = int(s_new_frame * hop_length)

        # 4) evitar sobreposição
        if len(refined_holes) > 0:
            s = max(s, refined_holes[-1][1])

        # 5) margem final
        e_frame_ext = e_frame + end_margin_frames
        e = min(len(y), int(e_frame_ext * hop_length))

        # 6) último furo → até o final
        if idx == len(holes) - 1:
            e = len(y)

        refined_holes.append((s, e))

    holes = refined_holes

    return (
        holes,
        rms_norm,
        diff,
        valleys_all,
        valleys_kept,
        depths,
        drop_peaks,
        hop_length
    )


# =========================================================
# PLOTAGENS
# =========================================================

def plot_diagnostics_with_valleys(
    rms_norm,
    diff,
    valleys_all,
    valleys_kept,
    depths,
    drop_peaks,
    hop_length,
    sr,
    holes
):
    times = (
        np.arange(len(rms_norm)) * hop_length / sr
        if len(rms_norm) > 0 else np.array([])
    )

    plt.figure(figsize=(14, 6))

    if len(times) > 0:
        plt.plot(times, rms_norm, label='RMS Normalizado')
        plt.plot(times, diff, label='|ΔRMS|', alpha=0.8)

    else:
        plt.plot([], [], label='RMS Normalizado')

    # vales
    if len(valleys_all) > 0:
        t_all = np.array(valleys_all) * hop_length / sr
        plt.vlines(
            t_all, ymin=0, ymax=1.05,
            colors='gray', alpha=0.25,
            linewidth=0.8, label='vales (todos)'
        )

    # vales filtrados
    if len(valleys_kept) > 0:
        t_kept = np.array(valleys_kept) * hop_length / sr
        plt.vlines(
            t_kept, ymin=0, ymax=1.05,
            colors='green', alpha=0.9,
            linewidth=1.2, label='vales profundos'
        )

    # quedas
    if len(drop_peaks) > 0:
        t_drops = np.array(drop_peaks) * hop_length / sr
        plt.vlines(
            t_drops, ymin=0, ymax=1.05,
            colors='red', alpha=0.4, linewidth=1.0,
            label='quedas (fallback)'
        )

    # furos
    for s, e in holes:
        plt.axvspan(s / sr, e / sr, color='lime', alpha=0.25)

    plt.ylim(-0.02, 1.05)
    plt.xlabel('Tempo (s)')
    plt.title('Diagnóstico de Furos e Vales')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_holes(y, sr, holes, broken_hole=None):
    plt.figure(figsize=(14, 6))

    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(y)),
        ref=np.max
    )

    librosa.display.specshow(
        S, sr=sr, x_axis='time', y_axis='log', cmap='magma'
    )

    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma com Furos')

    for i, (s, e) in enumerate(holes, 1):
        plt.axvspan(s / sr, e / sr, color='lime', alpha=0.3)
        plt.axvline(s / sr, color='white', linestyle='--', linewidth=0.7)
        plt.axvline(e / sr, color='white', linestyle='--', linewidth=0.7)

        mid = (s + e) / 2 / sr
        plt.text(
            mid, sr / 4, f'{i}',
            color='white', ha='center',
            va='center', fontsize=10,
            fontweight='bold', alpha=0.9
        )

    if broken_hole is not None:
        s_b, e_b = broken_hole
        s_b = max(0, int(s_b))
        e_b = min(len(y), int(e_b))

        plt.axvspan(s_b / sr, e_b / sr, color='red', alpha=0.4)
        plt.text(
            (s_b + e_b) / (2 * sr),
            sr / 3,
            "BROCA QUEBRADA",
            color='white',
            fontsize=11,
            fontweight='bold',
            ha='center'
        )

    plt.tight_layout()
    plt.show()


# =========================================================
# CUSTOM / PARAMETRIZADO
# =========================================================

def detect_holes_by_deep_valleys_custom(
    y,
    sr,
    DEPTH_THRESH=0.15,
    GROUP_GAP_SEC=3.0,
    MERGE_GAP_SEC=2.0
):
    global HOP_LENGTH, FRAME_LENGTH
    global MIN_PROMINENCE_VALLEY, MIN_HOLE_DURATION
    global SMOOTH_WINDOW, VALLEY_WINDOW_SEC
    global DROP_PROMINENCE, DROP_SEARCH_SEC
    global START_FACTOR, START_ABS_THRESH, START_SEARCH_SEC

    orig_DEPTH = globals().get('DEPTH_THRESH', 0.15)
    orig_GROUP = globals().get('GROUP_GAP_SEC', 3.0)
    orig_MERGE = globals().get('MERGE_GAP_SEC', 2.0)

    globals()['DEPTH_THRESH'] = DEPTH_THRESH
    globals()['GROUP_GAP_SEC'] = GROUP_GAP_SEC
    globals()['MERGE_GAP_SEC'] = MERGE_GAP_SEC

    results = detect_holes_by_deep_valleys(y, sr)

    globals()['DEPTH_THRESH'] = orig_DEPTH
    globals()['GROUP_GAP_SEC'] = orig_GROUP
    globals()['MERGE_GAP_SEC'] = orig_MERGE

    return results


# =========================================================
# REFINAMENTO DE FUROS LONGOS
# =========================================================

def refine_long_holes(
    y,
    sr,
    holes,
    duration_factor=1.5,
    aggressive_factor=2.5
):
    if len(holes) == 0:
        return []

    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations) if len(durations) > 0 else 0.0

    refined_holes = []

    for (s, e), dur in zip(holes, durations):
        if median_dur == 0 or dur <= duration_factor * median_dur:
            refined_holes.append((s, e))
            continue

        print(f"→ Refinando furo longo ({dur:.2f}s)...")

        y_seg = y[s:e]

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

        sub_holes, *_ = detect_holes_by_deep_valleys_custom(
            y_seg, sr, **params
        )

        sub_global = [(s + s2, s + e2) for (s2, e2) in sub_holes]

        if len(sub_global) == 0:
            refined_holes.append((s, e))
        else:
            refined_holes.extend(sub_global)

    return refined_holes


# =========================================================
# DETECÇÃO DE BROCA QUEBRADA NO ÚLTIMO FURO
# =========================================================

def detect_broken_drill_in_last_hole(
    y,
    sr,
    holes,
    rms_window=5,
    diff_thresh=0.15,
    anticip_sec=0.20,
):
    """
    Detector híbrido para identificar quebra de broca dentro do último furo.
    Usa:
      1) RMS diff (quedas)
      2) Crescimento explosivo de RMS (picos)
      3) Distância espectral entre frames
    """
    if not holes:
        return holes, None

    s_last, e_last = holes[-1]
    y_last = y[s_last:e_last]

    hop = 256
    frame = 1024

    # ---------- 1) RMS ----------
    rms = librosa.feature.rms(y=y_last, frame_length=frame, hop_length=hop)[0]

    # Menos smooth para não apagar eventos curtos
    rms_smooth = np.convolve(rms, np.ones(3)/3, mode='same')
    rms_norm = rms_smooth / np.max(rms_smooth)

    diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    diff = diff / np.max(diff) if np.max(diff)>0 else diff

    # ---------- 2) Picos de aumento (quebra por explosão) ----------
    rise_peaks, _ = signal.find_peaks(
        diff,
        prominence=0.15,
        distance=3
    )

    # ---------- 3) Mudança espectral ----------
    S = np.abs(librosa.stft(y_last, n_fft=1024, hop_length=hop))
    S = S / (np.max(S) + 1e-9)
    S_diff = np.mean(np.abs(np.diff(S, axis=1)), axis=0)

    S_diff_norm = S_diff / np.max(S_diff) if np.max(S_diff)>0 else S_diff

    spectral_peaks, _ = signal.find_peaks(
        S_diff_norm,
        prominence=0.25
    )

    # Combina todos os candidatos
    candidates = set(rise_peaks.tolist() + np.where(diff > 0.35)[0].tolist() + spectral_peaks.tolist())

    if not candidates:
        return holes, None

    # Escolhe o evento mais forte (maior intensidade espectral)
    best = max(candidates, key=lambda i: S_diff_norm[i])

    cut_time = best * hop / sr
    cut_time_adj = max(cut_time - anticip_sec, 0)
    cut_sample = s_last + int(cut_time_adj * sr)

    # Não permitir cortar dentro do furo anterior
    if len(holes) > 1:
        prev_end = holes[-2][1]
        cut_sample = max(cut_sample, prev_end + int(0.2 * sr))

    hole_normal = (s_last, cut_sample)
    hole_broken = (cut_sample, e_last)

    holes_adj = holes[:-1] + [hole_normal, hole_broken]

    print(f"⚠️ Broca quebrada detectada em {cut_sample/sr:.2f}s → {e_last/sr:.2f}s")

    return holes_adj, hole_broken


def fix_last_hole_by_duration(holes, sr, min_ratio=0.6):
    """
    Se o(s) último(s) furos tiverem duração menor que um percentual da média,
    funde o último furo com o penúltimo até atingir uma duração aceitável.
    """

    if len(holes) < 3:
        return holes

    # Cálculo da média dos FUROS NORMAIS (tirando os 2 últimos)
    durations = [(e - s) / sr for (s, e) in holes[:-2]]
    mean_dur = np.mean(durations)

    acceptable_min = mean_dur * min_ratio

    holes_fixed = holes.copy()

    # Enquanto o último furo for muito curto, combine com o penúltimo
    while len(holes_fixed) >= 2:
        s_last, e_last = holes_fixed[-1]
        dur_last = (e_last - s_last) / sr

        if dur_last >= acceptable_min:
            break  # já OK

        # Funde com o anterior
        s_prev, e_prev = holes_fixed[-2]
        holes_fixed[-2] = (s_prev, e_last)
        holes_fixed.pop()

    return holes_fixed



# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================

if __name__ == "__main__":
    # FILEPATH = "data/standardized/01/01_ultrasonic_250130_013_Tr6_ch1_int.wav"
    # FILEPATH = "data/standardized/01/01_ultrasonic_250130_011_Tr2_ch1_int.wav"
    # FILEPATH = "data/standardized/07/07_ultrasonic_190514_005_Tr3_ch1_int.wav"
    # FILEPATH = "data/standardized/02/02_common_190512_009_190512_009_int.wav"
    FILEPATH = "data/standardized/07/07_common_Tr5_int.wav" 

    # =========================================================
    # Carrega áudio
    # =========================================================
    y, sr = librosa.load(FILEPATH, sr=None, mono=True)

    # =========================================================
    # 1) Detecção inicial de furos (deep valleys)
    # =========================================================
    holes, rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length = (
        detect_holes_by_deep_valleys(y, sr)
    )

    # =========================================================
    # 2) Refinamento de furos muito longos
    # =========================================================
    holes_refined = refine_long_holes(
        y, sr, holes, duration_factor=1.5, aggressive_factor=2.5
    )

    print(f"\n→ Após refinamento adaptativo: {len(holes_refined)} furos detectados.\n")

    # =========================================================
    # 3) Detecta possível furo quebrado via RMS (pode ou não encontrar)
    # =========================================================
    holes_after_rms, broken_hole = detect_broken_drill_in_last_hole(
        y, sr, holes_refined
    )

    # =========================================================
    # 4) Corrige usando duração média (une furos curtos)
    # =========================================================
    holes_final = fix_last_hole_by_duration(holes_after_rms, sr)

    # Se o furo quebrado detectado anteriormente foi englobado na junção por duração,
    # ele deixa de ser válido.
    if broken_hole and holes_final[-1] != broken_hole:
        broken_hole = None

    # =========================================================
    # 5) Prints finais
    # =========================================================
    if broken_hole:
        print(
            f"→ Furo quebrado detectado antes da correção por duração: "
            f"{broken_hole[0] / sr:.2f}s → {broken_hole[1] / sr:.2f}s\n"
        )
    else:
        print("→ Nenhum furo quebrado detectado (após todas as correções).\n")

    print("→ Furos finais:\n")
    for i, (s, e) in enumerate(holes_final, start=1):
        dur = (e - s) / sr
        print(f" Furo {i}: {s / sr:.2f}s → {e / sr:.2f}s (duração {dur:.2f}s)")

    # =========================================================
    # 6) Gráficos
    # =========================================================
    plot_diagnostics_with_valleys(
        rms_norm,
        diff,
        valleys_all,
        valleys_kept,
        depths,
        drop_peaks,
        hop_length,
        sr,
        holes_final,
    )

    plot_spectrogram_with_holes(
        y, sr, holes_final, broken_hole=broken_hole
    )
