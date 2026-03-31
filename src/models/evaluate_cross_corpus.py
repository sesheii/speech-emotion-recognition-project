import argparse
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
)

from data_provider import load_and_merge_data


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

        self.lstm = nn.LSTM(
            input_size=128 * 16, hidden_size=64, bidirectional=True, batch_first=True
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
        context_vector, _ = self.attention(lstm_out)

        y = self.stats_drop(F.relu(self.stats_bn(self.stats_fc(stats))))

        combined = torch.cat((context_vector, y), dim=1)
        z = self.fusion_drop1(F.relu(self.fusion_bn1(self.fusion_fc1(combined))))
        return self.classifier(z)


class CRNNEvalDataset(Dataset):
    def __init__(self, mel_features, stats_features, labels):
        self.mels = torch.tensor(mel_features, dtype=torch.float32).view(-1, 1, 128, 94)
        self.stats = torch.tensor(stats_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mels[idx], self.stats[idx], self.labels[idx]


class TransformerEvalDataset(Dataset):
    def __init__(self, filepaths, labels, feature_extractor, target_sr=16000):
        self.filepaths = filepaths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.target_samples = int(3.0 * target_sr)

        print(f"Завантаження {len(filepaths)} аудіофайлів для оцінки...")
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
        inputs = self.feature_extractor(
            self.audios[idx], sampling_rate=self.target_sr, return_tensors="pt"
        )
        return inputs.input_values.squeeze(0), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/gold")
    parser.add_argument(
        "--base-features", type=str, default="data/gold/features.parquet"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--model-type", type=str, choices=["crnn", "hubert", "wav2vec2"], required=True
    )
    parser.add_argument("--run-name", type=str, required=True)

    parser.add_argument("--hubert-features", action="store_true")
    parser.add_argument("--wav2vec-features", action="store_true")
    parser.add_argument("--vggish-features", action="store_true")

    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Оцінка моделі {args.model_type} на пристрої: {device}")

    if args.model_type == "crnn":
        df = load_and_merge_data(
            base_features_path=args.base_features,
            data_dir=args.data_dir,
            use_hubert=args.hubert_features,
            use_vggish=args.vggish_features,
            use_wav2vec2=args.wav2vec_features,
            use_mel=True,
        )
    else:
        paths = args.base_features.split(",")
        df = pd.concat([pd.read_parquet(p.strip()) for p in paths], ignore_index=True)

    if df.empty:
        raise ValueError("Датасет порожній. Перевірте шляхи до файлів.")

    le = LabelEncoder()
    le.fit(["ang", "fea", "hap", "neu", "sad", "sur"])

    df = df[df["emotion"].isin(le.classes_)].reset_index(drop=True)
    labels = le.transform(df["emotion"].values)

    test_preds, test_probs = [], []

    if args.model_type == "crnn":
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

        X_mel = df[mel_cols].values
        X_stats = df[stats_cols].values

        X_stats_sc = StandardScaler().fit_transform(X_stats)

        dataset = CRNNEvalDataset(X_mel, X_stats_sc, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        model = ImprovedEmotionCRNN(
            num_stats_features=len(stats_cols), num_classes=6
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for mels, stats, lbls in tqdm(loader, desc="Inference CRNN"):
                mels, stats = mels.to(device), stats.to(device)
                outputs = model(mels, stats)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                test_probs.extend(probs.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())

    else:
        model_id = (
            "facebook/hubert-base-ls960"
            if args.model_type == "hubert"
            else "facebook/wav2vec2-base"
        )
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        dataset = TransformerEvalDataset(df["filepath"].values, labels, extractor)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        config = AutoConfig.from_pretrained(model_id, num_labels=6)
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, config=config
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for inputs, lbls in tqdm(loader, desc=f"Inference {args.model_type}"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs.logits, dim=1)
                _, preds = torch.max(outputs.logits, 1)

                test_probs.extend(probs.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())

    test_probs = np.array(test_probs)
    test_preds_orig = le.inverse_transform(test_preds)
    true_labels_orig = le.inverse_transform(labels)

    print(f"\n--- Результати Cross-Corpus оцінки ({args.run_name}) ---")
    print(classification_report(true_labels_orig, test_preds_orig, zero_division=0))

    os.makedirs("temp", exist_ok=True)
    safe_run_name = args.run_name.replace(" ", "_")

    cm = confusion_matrix(true_labels_orig, test_preds_orig)
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
    plt.title(f"Cross-Corpus Confusion Matrix - {args.run_name}")
    plt.savefig(f"temp/CrossCorpus_{safe_run_name}_cm.png", bbox_inches="tight")
    plt.close()

    y_test_bin = label_binarize(labels, classes=range(len(le.classes_)))
    plt.figure(figsize=(10, 8))
    for i in range(len(le.classes_)):
        if np.sum(y_test_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
            plt.plot(
                fpr, tpr, lw=2, label=f"{le.classes_[i]} (AUC = {auc(fpr, tpr):.2f})"
            )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Cross-Corpus ROC-AUC - {args.run_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"temp/CrossCorpus_{safe_run_name}_roc.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
