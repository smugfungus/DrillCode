#!/usr/bin/env python3
"""
segment.py (versão aprimorada)

Segmenta áudios padronizados em furos individuais, com detecção automática de furos quebrados ("jam").
Código: Lucas Araújo (UFRPE) - Novembro 2025

Requisitos:
    pip install librosa soundfile pandas matplotlib tqdm
"""

import os
import re
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# DIRETÓRIOS
# =========================================================
STANDARDIZED_DIR = "data/standardized"
RAW_DIR = "data/raw"
SEGMENTED_DIR = "data/segmented"
DOCS_IMG_DIR = "docs/img"
METADATA_DIR = "data/metadata"
METADATA_CSV = os.path.join(METADATA_DIR, "segmented_metadata.csv")

# =========================================================
# PARÂMETROS DETECTOR DE FUROS
# =========================================================
MIN_HOLE_DURATION = 2.0
SMOOTH_WINDOW = 9
VALLEY_WINDOW_SEC = 1.0
DEPTH_THRESH = 0.15
MIN_PROMINENCE_VALLEY = 0.01
HOP_LENGTH = 512
FRAME_LENGTH = 1024
GROUP_GAP_SEC = 3.0
MERGE_GAP_SEC = 2.0
DROP_PROMINENCE = 0.02
DROP_SEARCH_SEC = 1.5
START_FACTOR = 0.10
START_ABS_THRESH = 0.02
START_SEARCH_SEC = 20.0

# =========================================================
# PARÂMETROS DE DETECÇÃO AUTOMÁTICA DE JAM
# =========================================================
AUTO_JAM_DETECTION = True
JAM_RMS_DROP_RATIO = 0.5  # se RMS final cair mais que 50% do valor médio -> possível quebra (ajustável)

# =========================================================
# UTILITÁRIOS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_first_int(s):
    m = re.search(r'\d+', s)
    return int(m.group()) if m else 0

def load_jams_for_drill(drill_str, drill_folder_path):
    jams = set()
    m = re.search(r'(\d+)', drill_str)
    drill_num = int(m.group(1)) if m else None
    candidates = [
        os.path.join(METADATA_DIR, f"drill{drill_num:02d}_jams.txt") if drill_num else None,
        os.path.join(METADATA_DIR, "jams.txt"),
        os.path.join(METADATA_DIR, drill_str + "_jams.txt"),
        os.path.join(drill_folder_path, "jams.txt"),
        os.path.join(drill_folder_path, f"{drill_str}_jams.txt")
    ]
    candidates = [p for p in candidates if p]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read().replace(",", "\n")
                    for line in content.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        m2 = re.search(r'(\d+)', line)
                        if m2:
                            jams.add(int(m2.group(1)))
                print(f"   ▶ Arquivo de jams carregado: {p} -> {sorted(list(jams))}")
                return sorted(list(jams))
            except Exception as e:
                print(f"   ⚠️ Falha ao ler {p}: {e}")
    return []

# =========================================================
# DETECTOR DE FUROS (deep valleys) - substitui versão antiga
# =========================================================
def detect_holes_by_deep_valleys(y, sr):
    hop_length = HOP_LENGTH
    frame_length = FRAME_LENGTH
    END_MARGIN_SEC = 4.0

    if np.max(np.abs(y)) == 0:
        return [], None, None, [], [], [], [], hop_length

    y = librosa.effects.preemphasis(y)
    y = y / (np.max(np.abs(y)) + 1e-12)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return [], rms, np.array([]), [], [], [], [], hop_length

    rms_smooth = np.convolve(rms, np.ones(SMOOTH_WINDOW)/SMOOTH_WINDOW, mode='same')
    rms_norm = rms_smooth / (np.max(rms_smooth) + 1e-12)

    valleys_all, _ = signal.find_peaks(-rms_norm, prominence=MIN_PROMINENCE_VALLEY)
    window_frames = max(1, int((VALLEY_WINDOW_SEC * sr) / hop_length))

    valleys_kept = []
    depths = []
    local_peaks = {}
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

    # fallback se poucos vales úteis
    if len(valleys_kept) < 2 and len(valleys_all) >= 2:
        N = min(10, len(valleys_all))
        sorted_idx = np.argsort(depths)[-N:]
        valleys_kept = list(np.array(valleys_all)[sorted_idx])
        valleys_kept.sort()

    # agrupa vales em furos
    holes = []
    if valleys_kept:
        group_start = valleys_kept[0]
        group_end = valleys_kept[0]
        max_gap = int((GROUP_GAP_SEC * sr) / hop_length)
        for i in range(1, len(valleys_kept)):
            if valleys_kept[i] - valleys_kept[i-1] > max_gap:
                dur = (group_end - group_start) * hop_length / sr
                if dur >= MIN_HOLE_DURATION:
                    holes.append((group_start * hop_length, group_end * hop_length))
                group_start = valleys_kept[i]
            group_end = valleys_kept[i]
        dur = (group_end - group_start) * hop_length / sr
        if dur >= MIN_HOLE_DURATION:
            holes.append((group_start * hop_length, group_end * hop_length))

    # merge furos próximos
    merged_holes = []
    if holes:
        merged_holes.append(holes[0])
        for s, e in holes[1:]:
            last_s, last_e = merged_holes[-1]
            if (s/sr) - (last_e/sr) < MERGE_GAP_SEC:
                merged_holes[-1] = (last_s, e)
            else:
                merged_holes.append((s, e))
    holes = merged_holes

    # ajuste de margens - final extendido
    refined_holes = []
    end_margin_frames = int(END_MARGIN_SEC * sr / hop_length)
    for (s, e) in holes:
        s_new_frame = int(s / hop_length)
        e_frame_extended = int(e / hop_length) + end_margin_frames
        e = min(len(y), int(e_frame_extended * hop_length))
        refined_holes.append((s_new_frame * hop_length, e))
    return refined_holes, rms, rms_norm, valleys_all, valleys_kept, depths, [], hop_length

# =========================================================
# REFINAMENTO DE FUROS LONGOS
# =========================================================
def refine_long_holes(y, sr, holes, duration_factor=1.5, aggressive_factor=2.5):
    if len(holes) == 0:
        return []

    durations = np.array([(e - s) / sr for s, e in holes])
    median_dur = np.median(durations) if len(durations) > 0 else 0.0

    refined_holes = []
    for (s, e), dur in zip(holes, durations):
        if median_dur == 0 or dur <= duration_factor * median_dur:
            refined_holes.append((s, e))
            continue

        y_seg = y[s:e]
        if dur > aggressive_factor * median_dur:
            params = dict(DEPTH_THRESH=0.05, GROUP_GAP_SEC=0.5, MERGE_GAP_SEC=0.25)
        else:
            params = dict(DEPTH_THRESH=0.08, GROUP_GAP_SEC=1.0, MERGE_GAP_SEC=0.5)

        sub_holes, *_ = detect_holes_by_deep_valleys_custom(y_seg, sr, **params)
        sub_global = [(s + s2, s + e2) for (s2, e2) in sub_holes]
        if len(sub_global) == 0:
            refined_holes.append((s, e))
        else:
            refined_holes.extend(sub_global)
    return refined_holes

# =========================================================
# detect_holes_by_deep_valleys_custom (temporarily adjusts globals)
# =========================================================
def detect_holes_by_deep_valleys_custom(y, sr, DEPTH_THRESH=0.15, GROUP_GAP_SEC=3.0, MERGE_GAP_SEC=2.0):
    global HOP_LENGTH, FRAME_LENGTH
    global MIN_PROMINENCE_VALLEY, MIN_HOLE_DURATION
    global SMOOTH_WINDOW, VALLEY_WINDOW_SEC
    global DROP_PROMINENCE, DROP_SEARCH_SEC
    global START_FACTOR, START_ABS_THRESH, START_SEARCH_SEC

    # backup
    orig_DEPTH = globals().get('DEPTH_THRESH', 0.15)
    orig_GROUP = globals().get('GROUP_GAP_SEC', 3.0)
    orig_MERGE = globals().get('MERGE_GAP_SEC', 2.0)

    globals()['DEPTH_THRESH'] = DEPTH_THRESH
    globals()['GROUP_GAP_SEC'] = GROUP_GAP_SEC
    globals()['MERGE_GAP_SEC'] = MERGE_GAP_SEC

    results = detect_holes_by_deep_valleys(y, sr)

    # restore
    globals()['DEPTH_THRESH'] = orig_DEPTH
    globals()['GROUP_GAP_SEC'] = orig_GROUP
    globals()['MERGE_GAP_SEC'] = orig_MERGE

    return results

# =========================================================
# DETECÇÃO HÍBRIDA DE BROCA QUEBRADA NO ÚLTIMO FURO
# =========================================================
def detect_broken_drill_in_last_hole(y, sr, holes, rms_window=5, diff_thresh=0.15, anticip_sec=0.20):
    """
    Detector híbrido para identificar quebra de broca dentro do último furo.
    Retorna (holes_adj, hole_broken) onde hole_broken é (cut_sample, end_sample) or None
    """
    if not holes:
        return holes, None

    s_last, e_last = holes[-1]
    y_last = y[s_last:e_last]

    hop = 256
    frame = 1024

    # ---------- 1) RMS ----------
    rms = librosa.feature.rms(y=y_last, frame_length=frame, hop_length=hop)[0]
    if len(rms) == 0:
        return holes, None

    rms_smooth = np.convolve(rms, np.ones(3)/3, mode='same')
    rms_norm = rms_smooth / (np.max(rms_smooth) + 1e-12)

    diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    diff = diff / (np.max(diff) + 1e-12)

    # ---------- 2) Picos de aumento (explosão) ----------
    rise_peaks, _ = signal.find_peaks(diff, prominence=0.15, distance=3)

    # ---------- 3) Mudança espectral ----------
    S = np.abs(librosa.stft(y_last, n_fft=1024, hop_length=hop))
    S = S / (np.max(S) + 1e-12)
    S_diff = np.mean(np.abs(np.diff(S, axis=1)), axis=0)
    S_diff_norm = S_diff / (np.max(S_diff) + 1e-12)
    spectral_peaks, _ = signal.find_peaks(S_diff_norm, prominence=0.25)

    # Combina candidatos
    candidates = set(rise_peaks.tolist() + np.where(diff > 0.35)[0].tolist() + spectral_peaks.tolist())

    # Additional rule: if the tail RMS is very low compared to head -> consider jam at 80% position
    tail_mean = np.mean(rms_norm[int(len(rms_norm)*0.8):]) if len(rms_norm) > 0 else 0.0
    head_mean = np.mean(rms_norm[:max(1,int(len(rms_norm)*0.1))])
    if head_mean > 0 and (tail_mean / head_mean) < 0.25:
        # mark a candidate near 80% if none
        cand = int(len(rms_norm)*0.8)
        candidates.add(cand)

    if not candidates:
        return holes, None

    # choose best by spectral change strength
    best = max(candidates, key=lambda i: S_diff_norm[i] if i < len(S_diff_norm) else 0.0)
    cut_time = best * hop / sr
    cut_time_adj = max(cut_time - anticip_sec, 0)
    cut_sample = s_last + int(cut_time_adj * sr)

    # do not cut into previous hole
    if len(holes) > 1:
        prev_end = holes[-2][1]
        cut_sample = max(cut_sample, prev_end + int(0.2 * sr))

    hole_normal = (s_last, cut_sample)
    hole_broken = (cut_sample, e_last)
    holes_adj = holes[:-1] + [hole_normal, hole_broken]

    print(f"⚠️ Broca quebrada detectada (RMS/spectral) em {cut_sample/sr:.2f}s → {e_last/sr:.2f}s")
    return holes_adj, hole_broken

# =========================================================
# CORREÇÃO POR DURAÇÃO (une últimos furos curtos até ficar aceitável)
# =========================================================
def fix_last_hole_by_duration(holes, sr, min_ratio=0.6):
    """
    Se o(s) último(s) furos tiverem duração menor que um percentual da média,
    funde o último furo com o penúltimo até atingir uma duração aceitável.
    """
    if len(holes) < 3:
        return holes

    durations = [(e - s) / sr for (s, e) in holes[:-2]]
    mean_dur = np.mean(durations) if len(durations) > 0 else 0.0
    acceptable_min = mean_dur * min_ratio if mean_dur > 0 else 0.0

    holes_fixed = holes.copy()
    while len(holes_fixed) >= 2:
        s_last, e_last = holes_fixed[-1]
        dur_last = (e_last - s_last) / sr
        if dur_last >= acceptable_min:
            break
        # fuse with previous
        s_prev, e_prev = holes_fixed[-2]
        holes_fixed[-2] = (s_prev, e_last)
        holes_fixed.pop()
    return holes_fixed

# =========================================================
# DETECÇÃO AUTOMÁTICA DE JAM (por RMS - simples)
# =========================================================
def detect_jam_automatically(y_segment, sr):
    rms = librosa.feature.rms(y=y_segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_norm = rms / (np.max(rms) + 1e-12) if np.max(rms) > 0 else rms
    mean_rms = np.mean(rms_norm) if len(rms_norm) > 0 else 0.0
    last_10pct = rms_norm[int(len(rms_norm)*0.9):] if len(rms_norm) > 0 else np.array([0.0])
    mean_last = np.mean(last_10pct) if len(last_10pct) > 0 else 0.0
    drop_ratio = (mean_rms - mean_last) / mean_rms if mean_rms > 0 else 0.0
    if drop_ratio > JAM_RMS_DROP_RATIO:
        return True, drop_ratio
    return False, drop_ratio

# =========================================================
# SALVAMENTO DE FIGURAS
# =========================================================
def save_rms_histogram(y_segment, sr, out_path):
    try:
        rms = librosa.feature.rms(y=y_segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        plt.figure(figsize=(6,4))
        plt.hist(rms, bins=30, edgecolor='black', alpha=0.8)
        plt.xlabel("RMS"); plt.ylabel("Contagem de frames")
        plt.title("Histograma RMS do segmento")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Falha ao salvar histograma RMS: {e}")

def save_spectrogram(y_segment, sr, out_path, jam=False):
    try:
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y_segment, n_fft=2048, hop_length=HOP_LENGTH)), ref=np.max)
        plt.figure(figsize=(8,3))
        if jam:
            plt.gca().set_facecolor((1, 0.9, 0.9))  # leve tom avermelhado
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Espectrograma do segmento" + (" [JAM]" if jam else ""))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Falha ao salvar espectrograma: {e}")

def save_diagnostic_plot_full(y, sr, rms_norm, diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length, holes, out_path):
    try:
        times = np.arange(len(rms_norm)) * hop_length / sr if len(rms_norm) > 0 else np.array([])
        plt.figure(figsize=(14, 6))
        if len(times) > 0:
            plt.plot(times, rms_norm, label='RMS Normalizado')
            plt.plot(times, diff, label='|ΔRMS|', alpha=0.8)
        if len(valleys_all) > 0:
            t_all = np.array(valleys_all) * hop_length / sr
            plt.vlines(t_all, ymin=0, ymax=1.05, colors='gray', alpha=0.25, linewidth=0.8, label='vales (todos)')
        if len(valleys_kept) > 0:
            t_kept = np.array(valleys_kept) * hop_length / sr
            plt.vlines(t_kept, ymin=0, ymax=1.05, colors='green', alpha=0.9, linewidth=1.2, label='vales profundos')
        if len(drop_peaks) > 0:
            t_drops = np.array(drop_peaks) * hop_length / sr
            plt.vlines(t_drops, ymin=0, ymax=1.05, colors='red', alpha=0.4, linewidth=1.0, label='quedas (fallback)')
        for s, e in holes:
            plt.axvspan(s / sr, e / sr, color='lime', alpha=0.25)
        plt.ylim(-0.02, 1.05); plt.xlabel('Tempo (s)'); plt.title('Diagnóstico de Furos e Vales'); plt.legend(loc='upper right')
        plt.tight_layout(); plt.savefig(out_path); plt.close()
    except Exception as e:
        print(f"   ⚠️ Falha ao salvar diagnóstico geral: {e}")

# =========================================================
# OVERVIEW POR DRILL (timeline)
# =========================================================
def gerar_overview_drill(drill_str, records_for_drill):
    if not records_for_drill:
        return
    plt.figure(figsize=(12,2))
    for rec in records_for_drill:
        try:
            sr = rec.get("sr", 44100)
            start_sec = rec.get("start_sample", 0) / sr
            end_sec = rec.get("end_sample", rec.get("duration_sec", 0) * sr) / sr
            color = 'red' if rec.get("jam") else 'blue'
            plt.barh(0, width=end_sec-start_sec, left=start_sec, height=0.5, color=color, edgecolor='black', alpha=0.8)
        except Exception:
            continue
    plt.yticks([]); plt.xlabel("Tempo (s)")
    plt.title(f"Overview {drill_str}")
    out_dir = os.path.join(DOCS_IMG_DIR, drill_str)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{drill_str}_overview.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"   ▶ Overview salvo: {out_path}")

# =========================================================
# LOOP PRINCIPAL
# =========================================================
def main():
    ensure_dir(SEGMENTED_DIR)
    ensure_dir(METADATA_DIR)
    ensure_dir(DOCS_IMG_DIR)

    drills = [d for d in os.listdir(STANDARDIZED_DIR) if os.path.isdir(os.path.join(STANDARDIZED_DIR, d))]
    print(f"\n🔍 Iniciando segmentação de {len(drills)} drills encontrados em '{STANDARDIZED_DIR}'...\n")

    metadata_records = []
    hole_counters = {}

    for drill_folder_name in tqdm(drills, desc="Processando drills", unit="drill"):
        drill_folder = os.path.join(STANDARDIZED_DIR, drill_folder_name)
        drill_num = extract_first_int(drill_folder_name) or 0
        drill_str = f"drill{drill_num:02d}"
        jam_list = load_jams_for_drill(drill_str, drill_folder)

        files = [f for f in os.listdir(drill_folder) if f.lower().endswith(".wav") and not f.startswith("._")]
        print(f"\n🎯 {drill_str}: {len(files)} arquivos WAV encontrados.")

        records_for_overview = []
        hole_counters.setdefault(drill_str, 0)

        for file in tqdm(files, desc=f"Áudios de {drill_str}", unit="arquivo", leave=False):
            filepath = os.path.join(drill_folder, file)
            mic_id = extract_first_int(file)
            mic_type = "ult" if ("ult" in file.lower() or "ultrasonic" in file.lower()) else "com"
            position = "ext" if mic_id in [1,2,3] else "int"

            print(f"   ▶ Processando: {file}  (mic_type={mic_type}, mic_id={mic_id}, position={position})")
            try:
                y, sr = librosa.load(filepath, sr=None, mono=True)
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar {filepath}: {e}")
                continue

            holes, rms, rms_norm, valleys_all, valleys_kept, depths, drop_peaks, hop_length = detect_holes_by_deep_valleys(y, sr)
            if not holes:
                print(f"   → Nenhum furo detectado em {file}.")
                continue

            holes_refined = refine_long_holes(y, sr, holes, duration_factor=1.5, aggressive_factor=2.5)
            print(f"   → {file}: {len(holes_refined)} furos detectados (após refinamento).")

            # detect broken inside last hole (may split last hole)
            holes_after_rms, broken_hole = detect_broken_drill_in_last_hole(y, sr, holes_refined)
            # then fix durations (merge short tail pieces)
            holes_final = fix_last_hole_by_duration(holes_after_rms, sr)

            # diagnostic plot for whole audio (saved per-audio)
            diag_out_dir = os.path.join(DOCS_IMG_DIR, drill_str, os.path.splitext(file)[0])
            ensure_dir(diag_out_dir)
            try:
                diff = np.abs(np.diff((rms / (np.max(rms) + 1e-12))))
            except Exception:
                diff = np.array([])
            save_diagnostic_plot_full(y, sr, (rms / (np.max(rms) + 1e-12)), diff, valleys_all, valleys_kept, depths, drop_peaks, hop_length, holes_final, os.path.join(diag_out_dir, os.path.splitext(file)[0] + "_diagnostic.png"))

            # Export each hole
            for i, (start_sample, end_sample) in enumerate(holes_final):
                hole_counters[drill_str] += 1
                hole_idx = hole_counters[drill_str]

                # Determine jam flag:
                jam_flag = False
                # 1) explicit jams.txt
                if hole_idx in jam_list:
                    jam_flag = True
                # 2) automatic RMS-based detection (only for last hole in the file)
                if AUTO_JAM_DETECTION and (i == len(holes_final)-1) and not jam_flag:
                    auto_jam, drop_ratio = detect_jam_automatically(y[start_sample:end_sample], sr)
                    if auto_jam:
                        jam_flag = True
                        print(f"      ⚙️ Jam detectado automaticamente (queda RMS={drop_ratio:.2f})")
                # 3) broken_hole detection: if there was a broken range and it lies inside this exported hole
                if broken_hole is not None:
                    b_start, b_end = broken_hole
                    # if break cut point is inside this hole (or hole spans it), mark jam
                    if (b_start >= start_sample and b_start < end_sample) or (start_sample <= b_start < end_sample):
                        jam_flag = True

                # naming: Pasta A (nome da pasta = nome do arquivo sem ext)
                filename_base = f"{os.path.splitext(file)[0]}_hole{hole_idx:02d}_{mic_type}_{mic_id}_{position}"
                if jam_flag:
                    filename_base += "_jam"

                subfolder = f"{mic_type}_{position}"
                out_dir = os.path.join(SEGMENTED_DIR, os.path.splitext(file)[0], subfolder)  # Pasta A: usa nome do arquivo
                ensure_dir(out_dir)
                out_path = os.path.join(out_dir, filename_base + ".wav")
                try:
                    sf.write(out_path, y[start_sample:end_sample], sr)
                except Exception as e:
                    print(f"      ⚠️ Falha ao salvar WAV {out_path}: {e}")
                    continue

                # salvar imagens
                img_dir = os.path.join(DOCS_IMG_DIR, drill_str, os.path.splitext(file)[0], subfolder)
                ensure_dir(img_dir)
                save_rms_histogram(y[start_sample:end_sample], sr, os.path.join(img_dir, filename_base + "_rms_hist.png"))
                save_spectrogram(y[start_sample:end_sample], sr, os.path.join(img_dir, filename_base + "_spectrogram.png"), jam=jam_flag)

                duration_sec = (end_sample - start_sample) / sr if sr else 0
                rec = dict(drill_id=drill_num, hole_idx=hole_idx, mic_type=mic_type,
                           mic_id=mic_id, position=position, jam=bool(jam_flag),
                           filepath=out_path, duration_sec=round(duration_sec,3),
                           start_sample=int(start_sample), end_sample=int(end_sample), sr=int(sr))
                metadata_records.append(rec)
                records_for_overview.append(rec)
                print(f"      • Exportado: {os.path.basename(out_path)}  (d={duration_sec:.2f}s){' [JAM]' if jam_flag else ''}")

        # gerar overview do drill
        gerar_overview_drill(drill_str, records_for_overview)
        print(f"✅ {drill_str} concluído.\n")

    # salvar CSV metadata
    if metadata_records:
        df = pd.DataFrame(metadata_records)
        # garantir coluna jam_flag como TRUE/FALSE
        if 'jam' in df.columns:
            df.rename(columns={'jam': 'jam_flag'}, inplace=True)
        if 'jam_flag' not in df.columns:
            df['jam_flag'] = False
        df['jam_flag'] = df['jam_flag'].astype(bool)
        ensure_dir(os.path.dirname(METADATA_CSV))
        df.to_csv(METADATA_CSV, index=False)
        print(f"\n📁 Metadata salva em: {METADATA_CSV}")
    else:
        print("\n⚠️ Nenhum metadado gerado (nenhuma segmentação salva).")

    print("\n🏁 Segmentação concluída com sucesso!\n")


if __name__ == "__main__":
    main()
