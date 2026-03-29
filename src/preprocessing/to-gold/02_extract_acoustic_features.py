import os
import argparse
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


def extract_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=16000)

        features = {}

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
            features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
            features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta_mfcc[i])
            features[f"delta_mfcc_{i+1}_std"] = np.std(delta_mfcc[i])
            features[f"delta2_mfcc_{i+1}_mean"] = np.mean(delta2_mfcc[i])
            features[f"delta2_mfcc_{i+1}_std"] = np.std(delta2_mfcc[i])

        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

        features["spectral_centroid_mean"] = np.mean(cent)
        features["spectral_centroid_std"] = np.std(cent)
        features["spectral_rolloff_mean"] = np.mean(rolloff)
        features["spectral_rolloff_std"] = np.std(rolloff)
        features["spectral_bandwidth_mean"] = np.mean(bw)
        features["spectral_bandwidth_std"] = np.std(bw)

        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]

        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)
        features["rms_max"] = np.max(rms)

        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)

        f0 = librosa.yin(y, fmin=50, fmax=500, frame_length=2048)
        f0 = f0[~np.isnan(f0)]

        if len(f0) > 0:
            features["pitch_mean"] = np.mean(f0)
            features["pitch_std"] = np.std(f0)
            features["pitch_min"] = np.min(f0)
            features["pitch_max"] = np.max(f0)
            features["pitch_range"] = np.max(f0) - np.min(f0)
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_min"] = 0.0
            features["pitch_max"] = 0.0
            features["pitch_range"] = 0.0

        return features

    except Exception as e:
        print(f"\nПомилка вилучення ознак для {filepath}: {e}")
        return None


def process_features(
    metadata_path: str,
    gold_dir: str,
    use_ravdess: bool,
    use_iemocap: bool,
    use_savee: bool,
    output_filename: str,
):
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        print(
            f"Файл метаданих {meta_path} не знайдено! Спочатку запустіть скрипт вилучення метаданих."
        )
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
        print(
            "Не вказано жодного датасету для обробки. Використовуйте --ravdess, --iemocap, --savee."
        )
        return

    df_filtered = df_meta[df_meta["dataset"].isin(datasets)].copy()

    if df_filtered.empty:
        print("Не знайдено записів для вибраних датасетів у файлі метаданих.")
        return

    print(f"До обробки підготовлено {len(df_filtered)} аудіо-чанків.")

    all_features = []

    for idx, row in tqdm(
        df_filtered.iterrows(),
        total=len(df_filtered),
        desc="Обчислення акустичних ознак",
    ):
        filepath = row["filepath"]

        if not os.path.exists(filepath):
            print(f"\nФайл {filepath} не знайдено на диску.")
            all_features.append({})
            continue

        feats = extract_features(filepath)
        if feats is None:
            feats = {}

        all_features.append(feats)

    features_df = pd.DataFrame(all_features)

    df_filtered.reset_index(drop=True, inplace=True)
    final_df = pd.concat([df_filtered, features_df], axis=1)

    final_df = final_df.dropna(subset=["mfcc_1_mean"])

    gold_path = Path(gold_dir)
    gold_path.mkdir(parents=True, exist_ok=True)
    output_path = gold_path / output_filename

    final_df.to_parquet(output_path, engine="pyarrow", index=False)

    print(
        f"Успішно збережено {len(final_df)} рядків з {len(final_df.columns)} колонками. Шлях до файлу {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Додавання акустичних ознак до метаданих"
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
        default="features.parquet",
        help="Назва фінального parquet файлу",
    )

    args = parser.parse_args()

    process_features(
        args.metadata_path,
        args.gold_dir,
        args.ravdess,
        args.iemocap,
        args.savee,
        args.output,
    )


if __name__ == "__main__":
    main()
