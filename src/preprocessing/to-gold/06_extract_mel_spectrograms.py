import os
import argparse
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

TARGET_SR = 16000
TARGET_SAMPLES = int(3.0 * TARGET_SR)


def extract_mel_spectrogram(filepath, n_mels=128, n_fft=1024, hop_length=512):
    try:
        y, sr = librosa.load(filepath, sr=TARGET_SR)

        if len(y) > TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
        elif len(y) < TARGET_SAMPLES:
            pad_len = TARGET_SAMPLES - len(y)
            y = np.pad(y, (0, pad_len), mode="constant")

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=TARGET_SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        flat_mel = log_mel_spec.flatten()

        features = {f"mel_{i}": val for i, val in enumerate(flat_mel)}
        return features

    except Exception as e:
        print(f"\nПомилка вилучення Mel Spectrogram для {filepath}: {e}")
        return None


def process_mel_spectrograms(
    metadata_path: str,
    gold_dir: str,
    use_ravdess: bool,
    use_iemocap: bool,
    use_savee: bool,
    output_filename: str,
):
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        print(f"Файл метаданих {meta_path} не знайдено!")
        return

    df_meta = pd.read_parquet(meta_path)

    datasets = []
    if use_ravdess:
        datasets.append("ravdess")
    if use_iemocap:
        datasets.append("iemocap")
    if use_savee:
        datasets.append("savee")

    if not datasets:
        print("Вкажіть датасети: --ravdess, --iemocap, --savee.")
        return

    df_filtered = df_meta[df_meta["dataset"].isin(datasets)].copy()

    if df_filtered.empty:
        print("Не знайдено записів для вибраних датасетів.")
        return

    print(f"До обробки підготовлено {len(df_filtered)} аудіо-чанків.")

    all_features = []

    for idx, row in tqdm(
        df_filtered.iterrows(),
        total=len(df_filtered),
        desc="Обчислення Log-Mel Spectrograms",
    ):
        filepath = row["filepath"]

        if not os.path.exists(filepath):
            print(f"\nФайл {filepath} не знайдено.")
            all_features.append({})
            continue

        feats = extract_mel_spectrogram(
            filepath, n_mels=128, n_fft=1024, hop_length=512
        )

        if feats is None:
            feats = {}

        all_features.append(feats)

    features_df = pd.DataFrame(all_features)

    df_filtered.reset_index(drop=True, inplace=True)
    final_df = pd.concat([df_filtered, features_df], axis=1)

    final_df = final_df.dropna(subset=["mel_0"])

    gold_path = Path(gold_dir)
    gold_path.mkdir(parents=True, exist_ok=True)
    output_path = gold_path / output_filename

    final_df.to_parquet(output_path, engine="pyarrow", index=False)

    print(
        f"Готово! Успішно збережено {len(final_df)} рядків та {len(final_df.columns)} колонок. Шлях файлу {output_path}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Попереднє обчислення Mel-спектрограм для Gold шару"
    )

    parser.add_argument("--ravdess", action="store_true", help="Обробити RAVDESS")
    parser.add_argument("--iemocap", action="store_true", help="Обробити IEMOCAP")
    parser.add_argument("--savee", action="store_true", help="Обробити SAVEE")

    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/gold/basic_features/metadata.parquet",
        help="Шлях до вхідного файлу метаданих",
    )
    parser.add_argument(
        "--gold-dir", type=str, default="data/gold", help="Шлях до вихідної директорії"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mel_features.parquet",
        help="Назва вихідного parquet файлу",
    )

    args = parser.parse_args()

    process_mel_spectrograms(
        args.metadata_path,
        args.gold_dir,
        args.ravdess,
        args.iemocap,
        args.savee,
        args.output,
    )


if __name__ == "__main__":
    main()
