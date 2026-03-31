import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split


def greedy_speaker_split(
    df, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_seed=42, split_rank=1
):
    actor_stats = df.groupby(["actor_id", "gender"]).size().reset_index(name="count")
    total_samples = len(df)

    unique_splits = {}

    for i in range(200):
        shuffled_stats = actor_stats.sample(frac=1, random_state=random_seed + i)

        train_act, val_act, test_act = [], [], []
        curr_train, curr_val, curr_test = 0, 0, 0

        for gender in df["gender"].unique():
            gender_actors = shuffled_stats[shuffled_stats["gender"] == gender]
            gender_actors = gender_actors.sort_values(
                "count", ascending=False, kind="mergesort"
            )

            target_train = gender_actors["count"].sum() * train_frac
            target_val = gender_actors["count"].sum() * val_frac
            target_test = gender_actors["count"].sum() * test_frac

            g_curr_train, g_curr_val, g_curr_test = 0, 0, 0

            for _, row in gender_actors.iterrows():
                actor = row["actor_id"]
                count = row["count"]

                def_train = target_train - g_curr_train
                def_val = target_val - g_curr_val
                def_test = target_test - g_curr_test

                max_def = max(def_train, def_val, def_test)

                if max_def == def_train:
                    train_act.append(actor)
                    g_curr_train += count
                    curr_train += count
                elif max_def == def_val:
                    val_act.append(actor)
                    g_curr_val += count
                    curr_val += count
                else:
                    test_act.append(actor)
                    g_curr_test += count
                    curr_test += count

        act_tr_frac = curr_train / total_samples
        act_v_frac = curr_val / total_samples
        act_te_frac = curr_test / total_samples

        error = (
            abs(act_tr_frac - train_frac)
            + abs(act_v_frac - val_frac)
            + abs(act_te_frac - test_frac)
        )

        split_key = (frozenset(train_act), frozenset(val_act), frozenset(test_act))
        if split_key not in unique_splits:
            unique_splits[split_key] = {
                "train": train_act,
                "val": val_act,
                "test": test_act,
                "error": error,
            }

    sorted_splits = sorted(unique_splits.values(), key=lambda x: x["error"])
    actual_rank = min(split_rank, len(sorted_splits))
    if split_rank > len(sorted_splits):
        print(
            f"Знайдено лише {len(sorted_splits)} унікальних розбиттів. Використано ранг {len(sorted_splits)}."
        )

    best_split = sorted_splits[actual_rank - 1]

    train_actors = best_split["train"]
    val_actors = best_split["val"]
    test_actors = best_split["test"]

    train_df = df[df["actor_id"].isin(train_actors)].copy()
    val_df = df[df["actor_id"].isin(val_actors)].copy()
    test_df = df[df["actor_id"].isin(test_actors)].copy()

    print(f"\nСтатистика розбиття")
    print(
        f"Train: {len(train_df)} записів ({len(train_df)/total_samples*100:.1f}%) | Акторів: {len(train_actors)}"
    )
    print(
        f"Val: {len(val_df)} записів ({len(val_df)/total_samples*100:.1f}%) | Акторів: {len(val_actors)}"
    )
    print(
        f"Test: {len(test_df)} записів ({len(test_df)/total_samples*100:.1f}%) | Акторів: {len(test_actors)}"
    )
    print("------------------------------------------------------------------------\n")

    return train_df, val_df, test_df


def random_stratified_split(
    df, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_seed=42
):
    val_test_frac = val_frac + test_frac

    train_df, temp_df = train_test_split(
        df, test_size=val_test_frac, random_state=random_seed, stratify=df["emotion"]
    )

    relative_test_frac = test_frac / val_test_frac

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_frac,
        random_state=random_seed,
        stratify=temp_df["emotion"],
    )

    total_samples = len(df)
    print(f"\nСтатистика розбиття")
    print(
        f"Train: {len(train_df)} записів ({len(train_df)/total_samples*100:.1f}%) | Акторів: {train_df['actor_id'].nunique()}"
    )
    print(
        f"Val: {len(val_df)} записів ({len(val_df)/total_samples*100:.1f}%) | Акторів: {val_df['actor_id'].nunique()}"
    )
    print(
        f"Test: {len(test_df)} записів ({len(test_df)/total_samples*100:.1f}%) | Акторів: {test_df['actor_id'].nunique()}"
    )

    return train_df, val_df, test_df


def load_and_merge_data(
    base_features_path,
    data_dir=None,
    use_hubert=False,
    use_vggish=False,
    use_wav2vec2=False,
    use_mel=False,
):
    """
    Завантажує та об'єднує ознаки з одного або кількох датасетів.
    base_features_path може бути рядком зі шляхами через кому.
    """
    paths = (
        base_features_path.split(",")
        if isinstance(base_features_path, str)
        else base_features_path
    )

    join_keys = [
        "chunk_name",
        "original_filename",
        "chunk_id",
        "filepath",
        "file_hash",
        "dataset",
        "actor_id",
        "emotion",
        "gender",
    ]

    all_dfs = []
    for path in paths:
        path = path.strip()
        if not os.path.exists(path):
            print(f"Файл {path} не знайдено, пропускаємо.")
            continue

        df = pd.read_parquet(path)
        current_dir = os.path.dirname(path)

        if use_hubert:
            df = pd.merge(
                df,
                pd.read_parquet(os.path.join(current_dir, "hubert_features.parquet")),
                on=join_keys,
                how="inner",
            )
        if use_vggish:
            df = pd.merge(
                df,
                pd.read_parquet(os.path.join(current_dir, "vggish_features.parquet")),
                on=join_keys,
                how="inner",
            )
        if use_wav2vec2:
            df = pd.merge(
                df,
                pd.read_parquet(os.path.join(current_dir, "wav2vec2_features.parquet")),
                on=join_keys,
                how="inner",
            )
        if use_mel:
            df = pd.merge(
                df,
                pd.read_parquet(os.path.join(current_dir, "mel_features.parquet")),
                on=join_keys,
                how="inner",
            )

        if "dataset" in df.columns and "actor_id" in df.columns:
            df["actor_id"] = df["dataset"] + "_" + df["actor_id"].astype(str)

        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("Не вдалося завантажити жодного датасету. Перевірте шляхи.")

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Об'єднаний датасет готовий. Розмірність (рядки, колонки): {final_df.shape}")
    return final_df


def get_data_splits(
    data_dir="data/gold",
    base_features_path="data/gold/features.parquet",
    use_hubert=False,
    use_vggish=False,
    use_wav2vec2=False,
    random_seed=42,
    split_rank=1,
    use_mel=False,
    speaker_independent=True,
):

    df = load_and_merge_data(
        base_features_path=base_features_path,
        data_dir=data_dir,
        use_hubert=use_hubert,
        use_vggish=use_vggish,
        use_wav2vec2=use_wav2vec2,
        use_mel=use_mel,
    )

    if df.empty:
        raise ValueError("Фінальний датасет порожній. Перевірте наявність даних.")

    if speaker_independent:
        train_df, val_df, test_df = greedy_speaker_split(
            df,
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
            random_seed=random_seed,
            split_rank=split_rank,
        )
    else:
        train_df, val_df, test_df = random_stratified_split(
            df, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_seed=random_seed
        )

    metadata_cols = [
        "chunk_name",
        "original_filename",
        "chunk_id",
        "filepath",
        "file_hash",
        "dataset",
        "actor_id",
        "emotion",
        "gender",
    ]

    split_metadata = {
        "random_seed": random_seed,
        "split_rank": split_rank if speaker_independent else None,
        "speaker_independent": speaker_independent,
        "features": {
            "use_hubert": use_hubert,
            "use_vggish": use_vggish,
            "use_wav2vec2": use_wav2vec2,
            "use_mel": use_mel,
        },
        "splits": {
            "train": {
                "indices": train_df.index.tolist(),
                "file_hashes": train_df["file_hash"].tolist(),
                "actors": train_df["actor_id"].unique().tolist(),
            },
            "val": {
                "indices": val_df.index.tolist(),
                "file_hashes": val_df["file_hash"].tolist(),
                "actors": val_df["actor_id"].unique().tolist(),
            },
            "test": {
                "indices": test_df.index.tolist(),
                "file_hashes": test_df["file_hash"].tolist(),
                "actors": test_df["actor_id"].unique().tolist(),
            },
        },
    }

    X_train = train_df.drop(columns=metadata_cols)
    y_train = train_df["emotion"]

    X_val = val_df.drop(columns=metadata_cols)
    y_val = val_df["emotion"]

    X_test = test_df.drop(columns=metadata_cols)
    y_test = test_df["emotion"]

    return X_train, y_train, X_val, y_val, X_test, y_test, split_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Провайдер даних")
    parser.add_argument("--data-dir", "--data_dir", type=str, default="data/gold")
    parser.add_argument(
        "--base-features",
        "--base_features",
        type=str,
        default="data/gold/features.parquet",
    )
    parser.add_argument("--hubert-features", "--hubert_features", action="store_true")
    parser.add_argument("--vggish-features", "--vggish_features", action="store_true")
    parser.add_argument("--wav2vec-features", "--wav2vec_features", action="store_true")
    parser.add_argument("--mel-features", "--mel_features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-rank", type=int, default=1)
    parser.add_argument("--speaker-dependent", action="store_true")

    args = parser.parse_args()

    try:
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = get_data_splits(
            data_dir=args.data_dir,
            base_features_path=args.base_features,
            use_hubert=args.hubert_features,
            use_vggish=args.vggish_features,
            use_wav2vec2=args.wav2vec_features,
            use_mel=args.mel_features,
            random_seed=args.seed,
            split_rank=args.split_rank,
            speaker_independent=not args.speaker_dependent,
        )
        print(f"X_train shape: {X_train.shape}")
    except Exception as e:
        print(f"Виникла помилка: {e}")
