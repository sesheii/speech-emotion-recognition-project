import os
import argparse
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from data_provider import load_and_merge_data

def generate_scaler(args):
    print(f"Завантаження та об'єднання даних з: {args.base_features}...")
    
    df = load_and_merge_data(
        base_features_path=args.base_features,
        data_dir=args.data_dir,
        use_hubert=args.hubert_features,
        use_vggish=args.vggish_features,
        use_wav2vec2=args.wav2vec_features,
        use_mel=False
    )
    
    if df.empty:
        print("Датасет порожній!")
        return

    metadata_cols = [
        "chunk_name", "original_filename", "chunk_id", "filepath", 
        "file_hash", "dataset", "actor_id", "emotion", "gender", "emotion_enc"
    ]
    
    stats_cols = [c for c in df.columns if not c.startswith("mel_") and c not in metadata_cols]
    
    print(f"Знайдено {len(stats_cols)} колонок ознак.")
    
    if len(stats_cols) != 862:
        print(f"Кількість колонок {len(stats_cols)}, очікувалося 862!")

    X_stats = df[stats_cols].values
    
    scaler = StandardScaler()
    scaler.fit(X_stats)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(scaler, args.output)
    
    print(f"Scaler збережено у {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Генерація StandardScaler для статистичних ознак")
    
    parser.add_argument("--base-features", type=str, required=True, 
                        help="Шляхи до parquet файлів з ознаками через кому")
    parser.add_argument("--data-dir", type=str, default="data", 
                        help="Базова папка з даними")
    parser.add_argument("--output", type=str, default="trained_models/standard_scaler.pkl", 
                        help="Шлях для збереження готового скейлера (.pkl)")
    
    parser.add_argument("--hubert-features", action="store_true", help="Використовувати HuBERT ембединги")
    parser.add_argument("--vggish-features", action="store_true", help="Використовувати VGGish ембединги")
    parser.add_argument("--wav2vec-features", action="store_true", help="Використовувати Wav2Vec2 ембединги")
    
    args = parser.parse_args()
    generate_scaler(args)

if __name__ == "__main__":
    main()

# uv run create_scaler.py --base-features "data/gold_ravdess/features.parquet,data/gold_savee/features.parquet" --data-dir "data" --output "trained_models/standard_scaler.pkl" --hubert-features