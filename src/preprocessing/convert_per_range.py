# src/preprocessing/convert.py

import os
import pandas as pd
import soundfile as sf
import librosa

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/standardized"
METADATA_CSV = "data/metadata/initial_metadata.csv"

# Mapeamento de mics comuns
MIC_MAPPING = {
    "Tr1": ("common", "ext"),
    "Tr2": ("common", "ext"),
    "Tr3": ("common", "ext"),
    "Tr4": ("common", "int"),
    "Tr5": ("common", "int"),
    "Tr6": ("common", "int"),
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_channel(y_channel, sr, drill_id, mic_type, position, file_prefix, ch_index, metadata_list):
    """Salva cada canal de mic ultrassônico separadamente"""
    mic_id = f"ch{ch_index}"
    out_name = f"{drill_id}_{mic_type}_{file_prefix}_{mic_id}_{position}.wav"
    out_dir = os.path.join(OUTPUT_DIR, drill_id)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, out_name)
    sf.write(out_path, y_channel, sr)

    metadata_list.append({
        "drill_id": drill_id,
        "mic_type": mic_type,
        "mic_id": mic_id,
        "position": position,
        "sr": sr,
        "filepath_wav": out_path
    })
    print(f"✅ Canal {ch_index} salvo: {out_path}")

def process_multichannel(filepath, drill_id, mic_type, position, metadata_list):
    """Processa mic ultrassônico multicanal"""
    y, sr = sf.read(filepath, always_2d=True)
    file_prefix = os.path.splitext(os.path.basename(filepath))[0]
    for ch_index in range(y.shape[1]):
        y_ch = y[:, ch_index]
        process_channel(y_ch, sr, drill_id, mic_type, position, file_prefix, ch_index+1, metadata_list)

def process_wav(filepath, drill_id, mic_type, position, mic_id, metadata_list):
    """Processa mic comum mono"""
    y, sr = librosa.load(filepath, sr=None, mono=True)
    file_prefix = os.path.splitext(os.path.basename(filepath))[0]
    out_name = f"{drill_id}_{mic_type}_{mic_id}_{file_prefix}_{position}.wav"
    out_dir = os.path.join(OUTPUT_DIR, drill_id)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, out_name)
    sf.write(out_path, y, sr)

    metadata_list.append({
        "drill_id": drill_id,
        "mic_type": mic_type,
        "mic_id": mic_id,
        "position": position,
        "sr": sr,
        "filepath_wav": out_path
    })
    print(f"✅ Processado: {out_path}")

def main():
    metadata_list = []

    for drill_folder in os.listdir(RAW_DIR):
        drill_path = os.path.join(RAW_DIR, drill_folder)
        if not os.path.isdir(drill_path):
            continue

        # exemplo: drill_4mm_01_batch_00_collet_1_29-01-2025 → drill_id = 01
        drill_id = drill_folder.split("_")[2]

        for root, _, files in os.walk(drill_path):
            for file in files:
                if not file.lower().endswith(".wav") or file.startswith("._"):
                    continue

                filepath = os.path.join(root, file)

                # pega taxa de amostragem
                try:
                    sr = sf.info(filepath).samplerate
                except:
                    print(f"⚠️ Não foi possível ler: {file}")
                    continue

                # determina tipo de mic
                mic_type = "common" if sr <= 48000 else "ultrasonic"

                # determina posição
                position = "ext" if "ext" in file.lower() or "ext" in root.lower() else "int"

                if mic_type == "common":
                    # mantém mic_id original se possível
                    mic_name = [k for k in MIC_MAPPING.keys() if k in file]
                    mic_id = mic_name[0] if mic_name else os.path.splitext(file)[0]
                    process_wav(filepath, drill_id, mic_type, position, mic_id, metadata_list)
                else:
                    process_multichannel(filepath, drill_id, mic_type, position, metadata_list)

    # salva metadata
    ensure_dir(os.path.dirname(METADATA_CSV))
    pd.DataFrame(metadata_list).to_csv(METADATA_CSV, index=False)
    print(f"\n📊 Metadata inicial salva em: {METADATA_CSV}")

if __name__ == "__main__":
    main()
