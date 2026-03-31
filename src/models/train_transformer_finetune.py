import argparse
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.pytorch
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)

from data_provider import get_data_splits


class RawAudioEmotionDataset(Dataset):
    def __init__(
        self,
        df_indices,
        features_df,
        feature_extractor,
        max_length_sec=3.0,
        target_sr=16000,
    ):
        self.df = features_df.iloc[df_indices].reset_index(drop=True)
        self.filepaths = self.df["filepath"].values
        self.labels = self.df["emotion_enc"].values

        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.target_samples = int(max_length_sec * target_sr)

        self.audios = []
        for path in tqdm(self.filepaths):
            y, sr = librosa.load(path, sr=self.target_sr)

            if len(y) > self.target_samples:
                y = y[: self.target_samples]
            else:
                y = np.pad(y, (0, self.target_samples - len(y)), mode="constant")
            self.audios.append(y)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = self.audios[idx]
        label = self.labels[idx]

        inputs = self.feature_extractor(
            y, sampling_rate=self.target_sr, return_tensors="pt"
        )

        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.long)


def objective_factory(
    train_idx, val_idx, full_df, class_weights_tensor, le, args, device, model_id
):

    num_classes = len(le.classes_)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    train_dataset = RawAudioEmotionDataset(train_idx, full_df, feature_extractor)
    val_dataset = RawAudioEmotionDataset(val_idx, full_df, feature_extractor)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 8e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        config = AutoConfig.from_pretrained(model_id, num_labels=num_classes)
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, config=config
        ).to(device)
        model.freeze_feature_encoder()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.log_params(
                {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "model_id": model_id,
                }
            )

            best_val_f1 = -1.0
            train_losses, val_losses = [], []
            best_preds, best_labels_orig = [], []

            for epoch in range(args.epochs):
                model.train()
                running_loss = 0.0

                for input_values, labels in tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                    leave=False,
                ):
                    input_values, labels = input_values.to(device), labels.to(device)
                    optimizer.zero_grad()

                    outputs = model(input_values)
                    logits = outputs.logits

                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                train_loss = running_loss / len(train_loader)

                model.eval()
                running_val_loss = 0.0
                all_preds, all_labels = [], []

                with torch.no_grad():
                    for input_values, labels in tqdm(
                        val_loader,
                        desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                        leave=False,
                    ):
                        input_values, labels = input_values.to(device), labels.to(
                            device
                        )
                        outputs = model(input_values)
                        logits = outputs.logits

                        loss = criterion(logits, labels)
                        running_val_loss += loss.item()

                        _, predicted = torch.max(logits, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_loss = running_val_loss / len(val_loader)
                val_f1 = f1_score(
                    all_labels, all_preds, average="weighted", zero_division=0
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_f1_epoch", val_f1, step=epoch)

                if val_f1 >= best_val_f1:
                    best_val_f1 = val_f1
                    best_preds = le.inverse_transform(all_preds)
                    best_labels_orig = le.inverse_transform(all_labels)

            os.makedirs("temp", exist_ok=True)
            safe_run_name = args.run_name.replace(" ", "_")
            trial_curves_path = (
                f"temp/{safe_run_name}_trial_{trial.number}_learning_curves.png"
            )
            trial_cm_path = f"temp/{safe_run_name}_trial_{trial.number}_cm.png"

            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.legend()
            plt.title(f"Loss over Epochs (Trial {trial.number})")
            plt.savefig(trial_curves_path, bbox_inches="tight")
            mlflow.log_artifact(trial_curves_path, "plots")
            plt.close()

            if len(best_preds) > 0:
                report = classification_report(
                    best_labels_orig, best_preds, zero_division=0
                )
                mlflow.log_text(report, "metrics/classification_report.txt")

                cm = confusion_matrix(best_labels_orig, best_preds)
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                )
                plt.ylabel("Справжня емоція")
                plt.xlabel("Передбачена емоція")
                plt.title(f"Confusion Matrix (Trial {trial.number})")
                plt.savefig(trial_cm_path, bbox_inches="tight")
                mlflow.log_artifact(trial_cm_path, "plots")
                plt.close()

        return best_val_f1

    return objective


def main():
    parser = argparse.ArgumentParser(description="End-to-End Fine-Tuning Трансформерів")
    parser.add_argument("--data-dir", type=str, default="data/gold")
    parser.add_argument(
        "--base-features", type=str, default="data/gold/features.parquet"
    )

    parser.add_argument(
        "--model-type", type=str, choices=["hubert", "wav2vec2"], default="hubert"
    )

    parser.add_argument(
        "--speaker-independent",
        action="store_true",
        default=True,
        help="Увімкнути Speaker-Independent розбиття",
    )
    parser.add_argument(
        "--speaker-dependent",
        action="store_false",
        dest="speaker_independent",
        help="Вимкнути Speaker-Independent розбиття",
    )

    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--final-epochs", type=int, default=15, help="Епохи для фінальної моделі"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="Transformer_FineTuning")
    parser.add_argument("--run-name", type=str, default="Optuna_Finetune")

    args, _ = parser.parse_known_args()

    mlflow.set_experiment(args.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")

    if args.model_type == "hubert":
        model_id = "facebook/hubert-base-ls960"
    else:
        model_id = "facebook/wav2vec2-base"

    X_train, y_train, X_val, y_val, X_test, y_test, metadata = get_data_splits(
        data_dir=args.data_dir,
        base_features_path=args.base_features,
        use_hubert=False,
        use_vggish=False,
        use_wav2vec2=False,
        use_mel=False,
        random_seed=args.seed,
        split_rank=1,
        speaker_independent=args.speaker_independent,
    )

    paths = args.base_features.split(",")
    full_df = pd.concat([pd.read_parquet(p.strip()) for p in paths], ignore_index=True)

    if "dataset" in full_df.columns and "actor_id" in full_df.columns:
        full_df["actor_id"] = full_df["dataset"] + "_" + full_df["actor_id"].astype(str)
    le = LabelEncoder()
    full_df["emotion_enc"] = le.fit_transform(full_df["emotion"])

    train_idx = metadata["splits"]["train"]["indices"]
    val_idx = metadata["splits"]["val"]["indices"]
    test_idx = metadata["splits"]["test"]["indices"]

    y_train_enc = full_df.iloc[train_idx]["emotion_enc"].values
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_enc), y=y_train_enc
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"\nЗапуск Optuna HPO ({args.n_trials} trials по {args.epochs} епох)...")
    with mlflow.start_run(run_name=args.run_name) as parent_run:
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("speaker_independent", args.speaker_independent)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        objective = objective_factory(
            train_idx,
            val_idx,
            full_df,
            class_weights_tensor,
            le,
            args,
            device,
            model_id,
        )
        study.optimize(objective, n_trials=args.n_trials)

        print(f"\nНайкращий trial: #{study.best_trial.number}")
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})

        print(f"\nТренування фінальної моделі на {args.final_epochs} епохах...")

        best_params = study.best_trial.params
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        train_dataset = RawAudioEmotionDataset(train_idx, full_df, feature_extractor)
        val_dataset = RawAudioEmotionDataset(val_idx, full_df, feature_extractor)
        test_dataset = RawAudioEmotionDataset(test_idx, full_df, feature_extractor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=best_params["batch_size"], shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=best_params["batch_size"], shuffle=False
        )

        config = AutoConfig.from_pretrained(model_id, num_labels=len(le.classes_))
        final_model = AutoModelForAudioClassification.from_pretrained(
            model_id, config=config
        ).to(device)
        final_model.freeze_feature_encoder()

        optimizer = optim.AdamW(
            final_model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

        train_losses, val_losses = [], []
        best_final_val_f1 = -1.0

        os.makedirs("temp", exist_ok=True)
        safe_run_name = args.run_name.replace(" ", "_")
        model_path = f"temp/{safe_run_name}_best_model.pth"
        curves_path = f"temp/{safe_run_name}_learning_curves.png"
        roc_path = f"temp/{safe_run_name}_roc_curve.png"
        cm_path = f"temp/{safe_run_name}_confusion_matrix.png"

        for epoch in range(args.final_epochs):
            final_model.train()
            running_loss = 0.0

            for input_values, labels in tqdm(
                train_loader,
                desc=f"Final Epoch {epoch+1}/{args.final_epochs} [Train]",
                leave=False,
            ):
                input_values, labels = input_values.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = final_model(input_values)
                loss = nn.CrossEntropyLoss(weight=class_weights_tensor)(
                    outputs.logits, labels
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            final_model.eval()
            running_val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for input_values, labels in tqdm(
                    val_loader,
                    desc=f"Final Epoch {epoch+1}/{args.final_epochs} [Val]",
                    leave=False,
                ):
                    input_values, labels = input_values.to(device), labels.to(device)
                    outputs = final_model(input_values)
                    loss = nn.CrossEntropyLoss(weight=class_weights_tensor)(
                        outputs.logits, labels
                    )
                    running_val_loss += loss.item()

                    _, predicted = torch.max(outputs.logits, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = running_val_loss / len(val_loader)
            val_losses.append(val_loss)

            val_f1 = f1_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )

            mlflow.log_metric("final_train_loss", train_loss, step=epoch)
            mlflow.log_metric("final_val_loss", val_loss, step=epoch)
            mlflow.log_metric("final_val_f1", val_f1, step=epoch)

            if val_f1 >= best_final_val_f1:
                best_final_val_f1 = val_f1
                torch.save(final_model.state_dict(), model_path)

        print(
            f"Тренування завершено. Завантаження найкращої епохи (Val F1: {best_final_val_f1:.4f})..."
        )
        final_model.load_state_dict(torch.load(model_path))

        final_model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for input_values, labels in test_loader:
                input_values, labels = input_values.to(device), labels.to(device)
                outputs = final_model(input_values)

                probs = F.softmax(outputs.logits, dim=1)
                _, predicted = torch.max(outputs.logits, 1)

                test_probs.extend(probs.cpu().numpy())
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_probs = np.array(test_probs)
        test_preds_orig = le.inverse_transform(test_preds)
        test_labels_orig = le.inverse_transform(test_labels)

        test_acc = accuracy_score(test_labels_orig, test_preds_orig)
        test_f1 = f1_score(
            test_labels_orig, test_preds_orig, average="weighted", zero_division=0
        )

        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_f1_weighted", test_f1)

        y_test_bin = label_binarize(test_labels, classes=range(len(le.classes_)))
        try:
            roc_auc_macro = roc_auc_score(
                y_test_bin, test_probs, multi_class="ovr", average="macro"
            )
            mlflow.log_metric("final_test_roc_auc_macro", roc_auc_macro)
        except ValueError:
            print(
                "Не вдалося обчислити ROC-AUC, оскільки один або більше класів відсутні в тестовій вибірці."
            )

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title(f"Final Model Loss over Epochs ({args.run_name})")
        plt.savefig(curves_path, bbox_inches="tight")
        mlflow.log_artifact(curves_path, "plots")
        plt.close()

        plt.figure(figsize=(10, 8))
        for i in range(len(le.classes_)):
            if np.sum(y_test_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
                roc_auc_class = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    lw=2,
                    label=f"{le.classes_[i]} (AUC = {roc_auc_class:.2f})",
                )

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic (One-vs-Rest) - {args.run_name}")
        plt.legend(loc="lower right")
        plt.savefig(roc_path, bbox_inches="tight")
        mlflow.log_artifact(roc_path, "plots")
        plt.close()

        report = classification_report(
            test_labels_orig, test_preds_orig, zero_division=0
        )
        print("\nClassification Report (TEST - Фінальна модель):")
        print(report)
        mlflow.log_text(report, "metrics/final_test_classification_report.txt")

        cm = confusion_matrix(test_labels_orig, test_preds_orig)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
        )
        plt.ylabel("Справжня емоція")
        plt.xlabel("Передбачена емоція")
        plt.title(f"Confusion Matrix (Test Data) - {args.run_name}")
        plt.savefig(cm_path, bbox_inches="tight")
        mlflow.log_artifact(cm_path, "plots")
        plt.close()

        mlflow.log_artifact(model_path, "transformer_model")
        mlflow.sklearn.log_model(le, "label_encoder")


if __name__ == "__main__":
    main()
