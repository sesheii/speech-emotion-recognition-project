import os
import argparse
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def compute_file_hash(filepath, algo="md5"):
    hash_func = hashlib.new(algo)
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def process_metadata(
    silver_dir: str,
    gold_dir: str,
    use_ravdess: bool,
    use_iemocap: bool,
    use_savee: bool,
    output_filename: str,
):
    silver_path = Path(silver_dir)
    gold_path = Path(gold_dir) / "basic_features"
    gold_path.mkdir(parents=True, exist_ok=True)

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

    records = []

    for dataset in datasets:
        dataset_dir = silver_path / dataset
        if not dataset_dir.exists():
            print(f"Директорія {dataset_dir} не існує. Пропускаємо.")
            continue

        wav_files = list(dataset_dir.rglob("*.wav"))
        if not wav_files:
            print(f"Не знайдено файлів у {dataset_dir}.")
            continue

        for wav_path in tqdm(wav_files, desc=f"Збір метаданих {dataset.upper()}"):
            filename = wav_path.name

            parts = filename.replace(".wav", "").split("_")

            if len(parts) < 6:
                print(f"\nФайл {filename} не відповідає конвенції. Пропускаємо.")
                continue

            emotion = parts[0]
            gender = parts[1]
            actor_id = parts[2].replace("actor", "")
            dataset_name = parts[3]

            original_filename = "_".join(parts[4:-1])
            chunk_id = parts[-1].replace("chunk", "")

            file_hash = compute_file_hash(wav_path)

            record = {
                "chunk_name": filename,
                "original_filename": original_filename,
                "chunk_id": int(chunk_id),
                "filepath": str(wav_path),
                "file_hash": file_hash,
                "dataset": dataset_name,
                "actor_id": actor_id,
                "emotion": emotion,
                "gender": gender,
            }

            records.append(record)

    if not records:
        print("\nНе вдалося сформувати жодного запису.")
        return

    df = pd.DataFrame(records)

    df = df.sort_values(by=["original_filename", "chunk_id"]).reset_index(drop=True)

    output_path = gold_path / output_filename
    df.to_parquet(output_path, engine="pyarrow", index=False)

    print(f"Успішно збережено {len(df)} рядків. Шлях до файлу {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Створення метаданих")

    parser.add_argument(
        "--ravdess", action="store_true", help="Включити метадані з RAVDESS"
    )
    parser.add_argument(
        "--iemocap", action="store_true", help="Включити метадані з IEMOCAP"
    )
    parser.add_argument(
        "--savee", action="store_true", help="Включити метадані з SAVEE"
    )

    parser.add_argument(
        "--silver-dir",
        "--silver_dir",
        type=str,
        default="data/silver/chunked",
        help="Шлях до вхідних даних",
    )
    parser.add_argument(
        "--gold-dir",
        "--gold_dir",
        type=str,
        default="data/gold",
        help="Базовий шлях до gold даних",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata.parquet",
        help="Назва вихідного parquet файлу",
    )

    args = parser.parse_args()

    process_metadata(
        args.silver_dir,
        args.gold_dir,
        args.ravdess,
        args.iemocap,
        args.savee,
        args.output,
    )


if __name__ == "__main__":
    main()
