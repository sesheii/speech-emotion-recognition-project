import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path


def add_white_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    if signal_power == 0:
        return signal

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    return signal + noise


def process_dataset(input_dir, output_dir, dataset_name, snr_db):
    in_path = Path(input_dir) / dataset_name
    out_path = Path(output_dir) / dataset_name

    if not in_path.exists():
        print(f"Помилка: Директорія {in_path} не знайдена!")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    files = list(in_path.rglob("*.wav"))

    print(
        f"Додавання шуму SNR={snr_db}dB до {len(files)} файлів набору даних {dataset_name}..."
    )

    for filepath in tqdm(files):
        y, sr = librosa.load(filepath, sr=None)
        noisy_y = add_white_noise(y, snr_db)

        rel_path = filepath.relative_to(in_path)
        out_file = out_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        sf.write(out_file, noisy_y, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Створення зашумленої версії аудіо")
    parser.add_argument("--dataset", type=str, required=True, help="ravdess або savee")
    parser.add_argument(
        "--snr", type=float, default=30.0, help="Відношення сигнал/шум у децибелах"
    )
    parser.add_argument("--input-dir", type=str, default="data/silver/chunked")
    parser.add_argument("--output-dir", type=str, default="data/silver/chunked_noisy")
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.dataset, args.snr)
