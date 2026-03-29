import os
import re
import argparse
import torch
import torchaudio.transforms as T
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

TARGET_SR = 16000

TARGET_EMOTIONS = {
    "ang": "Anger",
    "hap": "Happiness",
    "sad": "Sadness",
    "neu": "Neutral",
    "fea": "Fear",
    "sur": "Surprise",
}

RAVDESS_EMO_MAP = {
    "01": "neu",
    "03": "hap",
    "04": "sad",
    "05": "ang",
    "06": "fea",
    "08": "sur",
}

IEMOCAP_EMO_MAP = {
    "ang": "ang",
    "hap": "hap",
    "exc": "hap",
    "sad": "sad",
    "neu": "neu",
    "fea": "fea",
    "sur": "sur",
}

SAVEE_EMO_MAP = {
    "a": "ang",
    "h": "hap",
    "sa": "sad",
    "n": "neu",
    "f": "fea",
    "su": "sur",
}


def process_audio(input_path: str, output_path: str):
    try:
        data, sample_rate = sf.read(input_path, dtype="float32")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        waveform = torch.from_numpy(data).T

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != TARGET_SR:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        sf.write(output_path, waveform.T.numpy(), TARGET_SR)

    except Exception as e:
        print(f"\nПомилка обробки файлу {input_path}: {e}")


def process_ravdess(bronze_dir: str, silver_dir: str):
    input_dir = Path(bronze_dir) / "ravdess"
    output_dir = Path(silver_dir) / "ravdess"
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(input_dir.rglob("*.wav"))
    if not wav_files:
        print("Не знайдено .wav файлів RAVDESS.")
        return

    processed_count = 0

    for wav_path in tqdm(wav_files, desc="RAVDESS"):
        filename = wav_path.name
        parts = filename.replace(".wav", "").split("-")

        if len(parts) != 7:
            continue

        emotion_code = parts[2]
        actor_id = int(parts[6])

        gender = "F" if actor_id % 2 == 0 else "M"

        if emotion_code in RAVDESS_EMO_MAP:
            unified_emo = RAVDESS_EMO_MAP[emotion_code]

            actor_id_str = f"{actor_id:02d}"

            new_filename = (
                f"{unified_emo}_{gender}_actor{actor_id_str}_ravdess_{filename}"
            )
            out_path = output_dir / new_filename

            process_audio(str(wav_path), str(out_path))
            processed_count += 1

    print(f"Успішно оброблено RAVDESS. {processed_count} файлів.")


def extract_iemocap_labels(iemocap_dir: Path) -> dict:
    labels = {}
    eval_files = list(iemocap_dir.rglob("*.txt"))

    pattern = re.compile(r"\[.+?\]\s+(Ses[a-zA-Z0-9_]+)\s+([a-z]{3})\s+\[.+?\]")

    for eval_file in eval_files:
        if eval_file.name.startswith("Ses"):
            with open(eval_file, "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        utterance_id, emotion = match.groups()
                        labels[utterance_id] = emotion
    return labels


def process_iemocap(bronze_dir: str, silver_dir: str):
    input_dir = Path(bronze_dir) / "iemocap"
    output_dir = Path(silver_dir) / "iemocap"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = extract_iemocap_labels(input_dir)
    if not labels:
        print("Увага: Не знайдено міток емоцій для IEMOCAP!")
        return

    wav_files = list(input_dir.rglob("*.wav"))
    if not wav_files:
        print("Не знайдено .wav файлів IEMOCAP.")
        return

    processed_count = 0

    for wav_path in tqdm(wav_files, desc="IEMOCAP"):
        filename = wav_path.name
        utterance_id = wav_path.stem

        if utterance_id not in labels:
            continue

        original_emo = labels[utterance_id]

        if original_emo in IEMOCAP_EMO_MAP:
            unified_emo = IEMOCAP_EMO_MAP[original_emo]

            parts = utterance_id.split("_")
            gender = parts[-1][0].upper() if len(parts) >= 3 else "U"
            if gender not in ["M", "F"]:
                gender = "U"

            session_num = utterance_id[3:5]

            actor_id_str = f"{session_num}{gender}"

            new_filename = (
                f"{unified_emo}_{gender}_actor{actor_id_str}_iemocap_{filename}"
            )
            out_path = output_dir / new_filename

            process_audio(str(wav_path), str(out_path))
            processed_count += 1

    print(f"Успішно оброблено IEMOCAP. {processed_count} файлів.")


def process_savee(bronze_dir: str, silver_dir: str):
    input_dir = Path(bronze_dir) / "savee"
    output_dir = Path(silver_dir) / "savee"
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(input_dir.rglob("*.wav"))
    if not wav_files:
        print("Не знайдено .wav файлів SAVEE.")
        return

    processed_count = 0
    pattern = re.compile(r"^([A-Z]{2})_([a-zA-Z]+)\d+\.wav$")

    for wav_path in tqdm(wav_files, desc="SAVEE"):
        filename = wav_path.name
        match = pattern.match(filename)

        if not match:
            continue

        actor_id, original_emo = match.groups()

        gender = "M"

        if original_emo in SAVEE_EMO_MAP:
            unified_emo = SAVEE_EMO_MAP[original_emo]

            new_filename = f"{unified_emo}_{gender}_actor{actor_id}_savee_{filename}"
            out_path = output_dir / new_filename

            process_audio(str(wav_path), str(out_path))
            processed_count += 1

    print(f"Успішно оброблено SAVEE. {processed_count} файлів.")


def main():
    parser = argparse.ArgumentParser(
        description="Обробка сирих датасетів до Silver шару"
    )

    parser.add_argument("--ravdess", action="store_true", help="Обробити RAVDESS")
    parser.add_argument("--iemocap", action="store_true", help="Обробити IEMOCAP")
    parser.add_argument("--savee", action="store_true", help="Обробити SAVEE")

    parser.add_argument(
        "--bronze-dir",
        "--bronze_dir",
        type=str,
        default="data/bronze",
        help="Шлях до сирих даних",
    )
    parser.add_argument(
        "--silver-dir",
        "--silver_dir",
        type=str,
        default="data/silver/unified",
        help="Шлях для збереження уніфікованих даних",
    )

    args = parser.parse_args()

    if not args.ravdess and not args.iemocap and not args.savee:
        print("Вкажіть датасет для обробки: --ravdess, --iemocap, --savee")
        return

    if args.ravdess:
        process_ravdess(args.bronze_dir, args.silver_dir)

    if args.iemocap:
        process_iemocap(args.bronze_dir, args.silver_dir)

    if args.savee:
        process_savee(args.bronze_dir, args.silver_dir)


if __name__ == "__main__":
    main()
