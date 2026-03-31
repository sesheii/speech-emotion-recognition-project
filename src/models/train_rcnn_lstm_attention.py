import argparse
import os
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.pytorch
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
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

from data_provider import load_and_merge_data


class PrecomputedEmotionDataset(Dataset):
    def __init__(self, mel_features, stats_features, labels):
        self.mels = torch.tensor(mel_features, dtype=torch.float32).view(-1, 1, 128, 94)
        self.stats = torch.tensor(stats_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mels[idx], self.stats[idx], self.labels[idx]


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_output):
        energy = torch.tanh(self.attention_weights(rnn_output))
        attention_scores = F.softmax(energy, dim=1)
        context_vector = rnn_output * attention_scores
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_scores


class ImprovedEmotionCRNN(nn.Module):
    def __init__(
        self, num_stats_features, num_classes=6, dropout_cnn=0.3, dropout_fc=0.5
    ):
        super(ImprovedEmotionCRNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_cnn1 = nn.Dropout2d(dropout_cnn)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_cnn2 = nn.Dropout2d(dropout_cnn)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_cnn3 = nn.Dropout2d(dropout_cnn + 0.1)

        self.lstm_input_size = 128 * 16
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=64,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size=128)

        self.stats_fc = nn.Linear(num_stats_features, 64)
        self.stats_bn = nn.BatchNorm1d(64)
        self.stats_drop = nn.Dropout(0.4)

        self.fusion_fc1 = nn.Linear(128 + 64, 128)
        self.fusion_bn1 = nn.BatchNorm1d(128)
        self.fusion_drop1 = nn.Dropout(dropout_fc)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, mel_spec, stats):
        x = self.drop_cnn1(self.pool1(F.relu(self.bn1(self.conv1(mel_spec)))))
        x = self.drop_cnn2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop_cnn3(self.pool3(F.relu(self.bn3(self.conv3(x)))))

        x = x.permute(0, 3, 1, 2).contiguous()
        batch_size, time_steps, channels, freq = x.size()
        x = x.view(batch_size, time_steps, channels * freq)

        lstm_out, _ = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)

        y = self.stats_drop(F.relu(self.stats_bn(self.stats_fc(stats))))

        combined = torch.cat((context_vector, y), dim=1)
        z = self.fusion_drop1(F.relu(self.fusion_bn1(self.fusion_fc1(combined))))
        logits = self.classifier(z)

        return logits


def objective_factory(df_cv, mel_cols, stats_cols, le, args, device):
    num_stats_features = len(stats_cols)
    num_classes = len(le.classes_)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        dropout_cnn = trial.suggest_float("dropout_cnn", 0.2, 0.4, step=0.1)
        dropout_fc = trial.suggest_float("dropout_fc", 0.4, 0.6, step=0.1)

        epochs = args.epochs

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.log_params(
                {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "dropout_cnn": dropout_cnn,
                    "dropout_fc": dropout_fc,
                    "folds": args.folds,
                }
            )

            if not args.speaker_dependent:
                cv = GroupKFold(n_splits=args.folds)
                splits = list(
                    cv.split(df_cv, df_cv["emotion_enc"], groups=df_cv["actor_id"])
                )
            else:
                cv = StratifiedKFold(
                    n_splits=args.folds, shuffle=True, random_state=args.seed
                )
                splits = list(cv.split(df_cv, df_cv["emotion_enc"]))

            fold_val_f1s = []

            for fold, (train_idx, val_idx) in enumerate(splits):
                df_train_fold = df_cv.iloc[train_idx]
                df_val_fold = df_cv.iloc[val_idx]

                X_mel_tr = df_train_fold[mel_cols].values
                X_stats_tr = df_train_fold[stats_cols].values
                y_tr = df_train_fold["emotion_enc"].values

                X_mel_v = df_val_fold[mel_cols].values
                X_stats_v = df_val_fold[stats_cols].values
                y_v = df_val_fold["emotion_enc"].values

                scaler = StandardScaler()
                X_stats_tr_sc = scaler.fit_transform(X_stats_tr)
                X_stats_v_sc = scaler.transform(X_stats_v)

                cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
                cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
                criterion = nn.CrossEntropyLoss(weight=cw_tensor)

                train_dataset = PrecomputedEmotionDataset(X_mel_tr, X_stats_tr_sc, y_tr)
                val_dataset = PrecomputedEmotionDataset(X_mel_v, X_stats_v_sc, y_v)

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                model = ImprovedEmotionCRNN(
                    num_stats_features=num_stats_features,
                    num_classes=num_classes,
                    dropout_cnn=dropout_cnn,
                    dropout_fc=dropout_fc,
                ).to(device)

                optimizer = optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )

                best_fold_val_f1 = -1.0

                for epoch in range(epochs):
                    model.train()
                    for mels, stats, labels in train_loader:
                        mels, stats, labels = (
                            mels.to(device),
                            stats.to(device),
                            labels.to(device),
                        )
                        optimizer.zero_grad()
                        outputs = model(mels, stats)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for mels, stats, labels in val_loader:
                            mels, stats, labels = (
                                mels.to(device),
                                stats.to(device),
                                labels.to(device),
                            )
                            outputs = model(mels, stats)
                            _, predicted = torch.max(outputs.data, 1)
                            all_preds.extend(predicted.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())

                    val_f1 = f1_score(
                        all_labels, all_preds, average="weighted", zero_division=0
                    )
                    if val_f1 > best_fold_val_f1:
                        best_fold_val_f1 = val_f1

                fold_val_f1s.append(best_fold_val_f1)
                mlflow.log_metric(f"fold_{fold}_val_f1", best_fold_val_f1)

            mean_cv_f1 = np.mean(fold_val_f1s)
            mlflow.log_metric("mean_cv_val_f1", mean_cv_f1)

        return mean_cv_f1

    return objective


def main():
    parser = argparse.ArgumentParser(description="Тренування CRNN моделі")
    parser.add_argument("--data-dir", type=str, default="data/gold")
    parser.add_argument(
        "--base-features", type=str, default="data/gold/features.parquet"
    )
    parser.add_argument("--hubert-features", action="store_true")
    parser.add_argument("--vggish-features", action="store_true")
    parser.add_argument("--wav2vec-features", action="store_true")

    parser.add_argument(
        "--speaker-dependent",
        action="store_true",
        help="Вимкнути Speaker-Independent розбиття",
    )
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--folds", type=int, default=5, help="Кількість фолдів для крос-валідації"
    )
    parser.add_argument(
        "--final-epochs", type=int, default=40, help="Епохи для фінальної моделі"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="PyTorch_CRNN_CV")
    parser.add_argument("--run-name", type=str, default="Optuna_5Fold_CV")

    args, _ = parser.parse_known_args()

    mlflow.set_experiment(args.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")

    df = load_and_merge_data(
        base_features_path=args.base_features,
        data_dir=args.data_dir,
        use_hubert=args.hubert_features,
        use_vggish=args.vggish_features,
        use_wav2vec2=args.wav2vec_features,
        use_mel=True,
    )

    if df.empty:
        raise ValueError("Датасет порожній. Перевірте шляхи до файлів.")

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
    mel_cols = [c for c in df.columns if c.startswith("mel_")]
    stats_cols = [
        c for c in df.columns if not c.startswith("mel_") and c not in metadata_cols
    ]

    le = LabelEncoder()
    df["emotion_enc"] = le.fit_transform(df["emotion"])

    if not args.speaker_dependent:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
        cv_idx, test_idx = next(gss.split(df, groups=df["actor_id"]))
    else:
        cv_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.15,
            random_state=args.seed,
            stratify=df["emotion_enc"],
        )

    df_cv = df.iloc[cv_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"Дані для K-Fold CV: {len(df_cv)} записів")
    print(f"Дані для фінального тесту: {len(df_test)} записів")

    run_metadata = {
        "features": {
            "use_hubert": args.hubert_features,
            "use_vggish": args.vggish_features,
            "use_wav2vec2": args.wav2vec_features,
            "use_mel": True,
        },
        "speaker_dependent": args.speaker_dependent,
        "folds": args.folds,
    }

    print(f"\nЗапуск Optuna {args.folds}-Fold CV HPO ({args.n_trials} trials)...")
    with mlflow.start_run(run_name=args.run_name) as parent_run:
        mlflow.log_params(run_metadata["features"])
        mlflow.log_param("speaker_dependent", args.speaker_dependent)
        mlflow.log_param("n_trials", args.n_trials)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        objective = objective_factory(
            df_cv=df_cv,
            mel_cols=mel_cols,
            stats_cols=stats_cols,
            le=le,
            args=args,
            device=device,
        )

        study.optimize(objective, n_trials=args.n_trials)

        print(f"\nНайкращий Trial: #{study.best_trial.number}")
        print(f"Найкраща середня метрика (CV Mean F1): {study.best_trial.value:.4f}")
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
        mlflow.log_metric("best_cv_mean_f1", study.best_trial.value)

        best_params = study.best_trial.params

        if not args.speaker_dependent:
            gss_final = GroupShuffleSplit(
                n_splits=1, test_size=0.15, random_state=args.seed
            )
            f_train_idx, f_val_idx = next(
                gss_final.split(df_cv, groups=df_cv["actor_id"])
            )
        else:
            f_train_idx, f_val_idx = train_test_split(
                np.arange(len(df_cv)),
                test_size=0.15,
                random_state=args.seed,
                stratify=df_cv["emotion_enc"],
            )

        df_ftrain = df_cv.iloc[f_train_idx]
        df_fval = df_cv.iloc[f_val_idx]

        X_mel_ftrain, X_stats_ftrain, y_ftrain = (
            df_ftrain[mel_cols].values,
            df_ftrain[stats_cols].values,
            df_ftrain["emotion_enc"].values,
        )
        X_mel_fval, X_stats_fval, y_fval = (
            df_fval[mel_cols].values,
            df_fval[stats_cols].values,
            df_fval["emotion_enc"].values,
        )
        X_mel_test, X_stats_test, y_test = (
            df_test[mel_cols].values,
            df_test[stats_cols].values,
            df_test["emotion_enc"].values,
        )

        scaler = StandardScaler()
        X_stats_ftrain_sc = scaler.fit_transform(X_stats_ftrain)
        X_stats_fval_sc = scaler.transform(X_stats_fval)
        X_stats_test_sc = scaler.transform(X_stats_test)

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_ftrain), y=y_ftrain
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            device
        )

        train_dataset = PrecomputedEmotionDataset(
            X_mel_ftrain, X_stats_ftrain_sc, y_ftrain
        )
        val_dataset = PrecomputedEmotionDataset(X_mel_fval, X_stats_fval_sc, y_fval)
        test_dataset = PrecomputedEmotionDataset(X_mel_test, X_stats_test_sc, y_test)

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

        final_model = ImprovedEmotionCRNN(
            num_stats_features=X_stats_ftrain_sc.shape[1],
            num_classes=len(le.classes_),
            dropout_cnn=best_params["dropout_cnn"],
            dropout_fc=best_params["dropout_fc"],
        ).to(device)

        optimizer = optim.Adam(
            final_model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

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

            for mels, stats, labels in train_loader:
                mels, stats, labels = (
                    mels.to(device),
                    stats.to(device),
                    labels.to(device),
                )
                optimizer.zero_grad()
                outputs = final_model(mels, stats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            final_model.eval()
            running_val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for mels, stats, labels in val_loader:
                    mels, stats, labels = (
                        mels.to(device),
                        stats.to(device),
                        labels.to(device),
                    )
                    outputs = final_model(mels, stats)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
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
            for mels, stats, labels in test_loader:
                mels, stats, labels = (
                    mels.to(device),
                    stats.to(device),
                    labels.to(device),
                )
                outputs = final_model(mels, stats)

                probs = F.softmax(outputs.data, dim=1)
                _, predicted = torch.max(outputs.data, 1)

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
        print("\nClassification Report (TEST Set - Фінальна модель):")
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
        plt.title(f"Confusion Matrix (Hold-out Test Data) - {args.run_name}")
        plt.savefig(cm_path, bbox_inches="tight")
        mlflow.log_artifact(cm_path, "plots")
        plt.close()

        mlflow.log_dict(run_metadata, "metadata/split_metadata.json")
        mlflow.log_artifact(model_path, "pytorch_crnn_model")
        mlflow.sklearn.log_model(scaler, "standard_scaler")
        mlflow.sklearn.log_model(le, "label_encoder")


if __name__ == "__main__":
    main()
