import os
import argparse
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

TARGET_SR = 16000
CHUNK_LEN_SEC = 3.0
HOP_LEN_SEC = 1.5


def process_and_chunk(input_path: Path, output_dir: Path):
    try:
        data, sample_rate = sf.read(str(input_path), dtype="float32")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        target_samples = int(CHUNK_LEN_SEC * TARGET_SR)
        hop_samples = int(HOP_LEN_SEC * TARGET_SR)

        base_name = input_path.stem

        if len(data) <= target_samples:
            pad_len = target_samples - len(data)
            chunk = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")

            out_path = output_dir / f"{base_name}_chunk0.wav"
            sf.write(str(out_path), chunk, TARGET_SR)

        # якщо аудіо довше за 3 секунди
        else:
            start_idx = 0
            chunk_id = 0

            while start_idx < len(data):
                chunk = data[start_idx : start_idx + target_samples]

                if len(chunk) < target_samples:
                    # якщо залишився зовсім малий шматок і це не єдиний чанк - ігноруємо його
                    if len(chunk) < int(0.5 * TARGET_SR) and chunk_id > 0:
                        break

                    # інакше доповнюємо залишок нулями до 3 секунд
                    pad_len = target_samples - len(chunk)
                    chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode="constant")

                out_path = output_dir / f"{base_name}_chunk{chunk_id}.wav"
                sf.write(str(out_path), chunk, TARGET_SR)

                start_idx += hop_samples
                chunk_id += 1

    except Exception as e:
        print(f"\nПомилка чанкування файлу {input_path.name}: {e}")


def chunk_dataset(input_dir: str, output_dir: str, dataset_name: str):
    in_path = Path(input_dir) / dataset_name
    out_path = Path(output_dir) / dataset_name

    if not in_path.exists():
        print(f"Директорія {in_path} не існує. Пропускаємо {dataset_name}.")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    wav_files = list(in_path.rglob("*.wav"))
    if not wav_files:
        print(f"Не знайдено файлів у {in_path}.")
        return

    for wav_path in tqdm(wav_files, desc=f"Чанкування {dataset_name.upper()}"):
        process_and_chunk(wav_path, out_path)


def main():
    parser = argparse.ArgumentParser(description="Чанкування аудіо")

    parser.add_argument("--ravdess", action="store_true", help="Обробити RAVDESS")
    parser.add_argument("--iemocap", action="store_true", help="Обробити IEMOCAP")
    parser.add_argument("--savee", action="store_true", help="Обробити SAVEE")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/silver/enhanced",
        help="Шлях до вхідних даних (silver/enhanced)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/silver/chunked",
        help="Шлях для збереження чанків (silver/chunked)",
    )

    args = parser.parse_args()

    if not args.ravdess and not args.iemocap and not args.savee:
        print("Вкажіть датасет для обробки: --ravdess, --iemocap, --savee")
        return

    if args.ravdess:
        chunk_dataset(args.input_dir, args.output_dir, "ravdess")

    if args.iemocap:
        chunk_dataset(args.input_dir, args.output_dir, "iemocap")

    if args.savee:
        chunk_dataset(args.input_dir, args.output_dir, "savee")


if __name__ == "__main__":
    main()
