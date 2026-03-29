import os
import argparse
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def apply_preemphasis(waveform: torch.Tensor, alpha: float = 0.97) -> torch.Tensor:
    preemphasized = torch.cat(
        (waveform[:, :1], waveform[:, 1:] - alpha * waveform[:, :-1]), dim=1
    )
    return preemphasized


def remove_silence(waveform: torch.Tensor, top_db: int = 30) -> torch.Tensor:
    y = waveform.numpy().squeeze()

    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        return torch.empty((1, 0))

    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])

    return torch.from_numpy(non_silent_audio).unsqueeze(0)


def normalize_volume(waveform: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform


def process_and_save(input_path: Path, output_path: Path, top_db: int):
    try:
        data, sample_rate = sf.read(str(input_path), dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        waveform = torch.from_numpy(data).T

        waveform = remove_silence(waveform, top_db=top_db)

        if waveform.shape[1] == 0:
            print(
                f"\nФайл {input_path.name} виявився суцільною тишею (поріг {top_db}dB). Пропускаємо."
            )
            return

        waveform = apply_preemphasis(waveform)

        waveform = normalize_volume(waveform)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform.T.numpy(), sample_rate)

    except Exception as e:
        print(f"\nПомилка обробки файлу {input_path.name}: {e}")


def enhance_dataset(input_dir: str, output_dir: str, dataset_name: str, top_db: int):
    in_path = Path(input_dir) / dataset_name
    out_path = Path(output_dir) / dataset_name

    if not in_path.exists():
        print(f"Директорія {in_path} не існує. Пропускаємо {dataset_name}.")
        return

    wav_files = list(in_path.rglob("*.wav"))
    if not wav_files:
        print(f"Не знайдено файлів у {in_path}.")
        return

    for wav_path in tqdm(wav_files, desc=f"Акустична обробка {dataset_name.upper()}"):
        out_file_path = out_path / wav_path.name
        process_and_save(wav_path, out_file_path, top_db)


def main():
    parser = argparse.ArgumentParser(
        description="Акустична обробка: VAD, pre-emphasis, normalization"
    )

    parser.add_argument("--ravdess", action="store_true", help="Обробити RAVDESS")
    parser.add_argument("--iemocap", action="store_true", help="Обробити IEMOCAP")
    parser.add_argument("--savee", action="store_true", help="Обробити SAVEE")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/silver/unified",
        help="Шлях до вхідних даних (silver/unified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/silver/enhanced",
        help="Шлях для збереження покращених даних (silver/enhanced)",
    )
    parser.add_argument(
        "--top-db",
        type=int,
        default=30,
        help="Поріг у децибелах для видалення тиші (за замовчуванням: 30)",
    )

    args = parser.parse_args()

    if not args.ravdess and not args.iemocap and not args.savee:
        print("Вкажіть датасет для обробки: --ravdess, --iemocap, --savee")
        return

    if args.ravdess:
        enhance_dataset(args.input_dir, args.output_dir, "ravdess", args.top_db)

    if args.iemocap:
        enhance_dataset(args.input_dir, args.output_dir, "iemocap", args.top_db)

    if args.savee:
        enhance_dataset(args.input_dir, args.output_dir, "savee", args.top_db)


if __name__ == "__main__":
    main()
