# src/preprocessing/convert.py

import os
import pandas as pd
import soundfile as sf
import numpy as np

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

# Mapeamento ultrassônicos (ajuste conforme necessário)
ULTRASONIC_MAPPING = {
    "ultrasonic_ext": ("ultrasonic", "ext"),
    "ultrasonic_int": ("ultrasonic", "int"),
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_channel(y_channel, sr, out_path):
    sf.write(out_path, y_channel, sr)

def process_multichannel(filepath, drill_id, mic_type, position, metadata_list):
    """Separa canais multicanais e salva individualmente"""
    y, sr = sf.read(filepath, always_2d=True)
    num_channels = y.shape[1]

    for ch in range(num_channels):
        y_ch = y[:, ch]
        out_name = f"{drill_id}_{mic_type}_ch{ch+1}_{position}.wav"
        out_dir = os.path.join(OUTPUT_DIR, drill_id)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, out_name)
        process_channel(y_ch, sr, out_path)

        metadata_list.append({
            "drill_id": drill_id,
            "mic_type": mic_type,
            "mic_id": f"ch{ch+1}",
            "position": position,
            "sr": sr,
            "filepath_wav": out_path
        })
        print(f"✅ Canal {ch+1}/{num_channels} salvo: {out_path}")


def process_wav(filepath, drill_id, mic_name, mic_type, position, mic_id, metadata_list):
    # Load audio with soundfile; convert to mono if necessary
    y, sr = sf.read(filepath, always_2d=False, dtype='float32')
    # If data is 2D (multi-channel), average channels to mono
    if hasattr(y, "ndim") and y.ndim > 1:
        y = np.mean(y, axis=1)
    out_name = f"{drill_id}_{mic_type}_{mic_id}_{position}.wav"
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

                # Verifica se é mic comum
                mic_name = [k for k in MIC_MAPPING.keys() if k in file]
                if mic_name:
                    mic_name = mic_name[0]
                    mic_type, position = MIC_MAPPING[mic_name]
                    mic_id = mic_name
                    process_wav(filepath, drill_id, mic_name, mic_type, position, mic_id, metadata_list)
                    continue

                # Verifica se é mic ultrassônico
                if "ultrasonic" in root.lower() or "ultrasonic" in file.lower():
                    # Tenta inferir posição pelo nome do arquivo ou pasta
                    position = "ext" if "ext" in file.lower() else "int"
                    mic_type = "ultrasonic"
                    mic_id = os.path.splitext(file)[0]
                    process_multichannel(filepath, drill_id, mic_type, position, metadata_list)
                    continue

                print(f"⚠️ Mic não mapeado: {file}, pulando...")

    ensure_dir(os.path.dirname(METADATA_CSV))
    pd.DataFrame(metadata_list).to_csv(METADATA_CSV, index=False)
    print(f"\n📊 Metadata inicial salva em: {METADATA_CSV}")

if __name__ == "__main__":
    main()
