import argparse
import os
import shutil
import kagglehub
import re


def setup_dir(data_dir: str, dataset_name: str):
    target_path = os.path.abspath(os.path.join(data_dir, dataset_name))
    os.makedirs(target_path, exist_ok=True)
    return target_path


def copy_dataset(src_path, dest_path):
    for item in os.listdir(src_path):
        s = os.path.join(src_path, item)
        d = os.path.join(dest_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def download_ravdess(data_dir: str):
    target = setup_dir(data_dir, "ravdess")
    tmp_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    copy_dataset(tmp_path, target)


def download_iemocap(data_dir: str):
    target = setup_dir(data_dir, "iemocap")
    tmp_path = kagglehub.dataset_download("sangayb/iemocap")
    copy_dataset(tmp_path, target)


def download_savee(data_dir: str):
    target = setup_dir(data_dir, "savee")
    tmp_path = kagglehub.dataset_download(
        "ejlok1/surrey-audiovisual-expressed-emotion-savee"
    )

    pattern = re.compile(r"^[A-Z]{2}_[a-zA-Z]+\d+\.wav$", re.IGNORECASE)

    copied_count = 0
    for root, _, files in os.walk(tmp_path):
        for file in files:
            if pattern.match(file):
                s = os.path.join(root, file)
                d = os.path.join(target, file)

                if not os.path.exists(d):
                    shutil.copy2(s, d)
                    copied_count += 1

    print(f"Успішно скопійовано {copied_count} файлів SAVEE до {target}")


def main():
    parser = argparse.ArgumentParser(description="Завантажити сирі дані")
    parser.add_argument(
        "--ravdess", action="store_true", help="Завантажити набір даних RAVDESS"
    )
    parser.add_argument(
        "--iemocap", action="store_true", help="Завантажити набір даних IEMOCAP"
    )
    parser.add_argument(
        "--savee", action="store_true", help="Завантажити набір даних SAVEE"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/bronze",
        help="Базова директорія для збереження даних",
    )

    args = parser.parse_args()

    if not args.ravdess and not args.iemocap and not args.savee:
        print("Будь ласка, вкажіть датасет (--ravdess, --iemocap або --savee)")
        return

    if args.ravdess:
        download_ravdess(args.data_dir)

    if args.iemocap:
        download_iemocap(args.data_dir)

    if args.savee:
        download_savee(args.data_dir)


if __name__ == "__main__":
    main()
