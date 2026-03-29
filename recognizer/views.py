import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from transformers import (
    AutoConfig,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    HubertModel,
)


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


TARGET_SR = 16000
TARGET_SAMPLES = int(3.0 * TARGET_SR)


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


def process_and_chunk_in_memory(audio_file_obj):
    data, sample_rate = sf.read(audio_file_obj, dtype="float32")

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sample_rate != TARGET_SR:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_SR)

    data = data.reshape(-1, 1)
    waveform = torch.from_numpy(data).T

    waveform = remove_silence(waveform, top_db=30)
    if waveform.shape[1] == 0:
        return []

    waveform = apply_preemphasis(waveform)
    waveform = normalize_volume(waveform)

    data_enhanced = waveform.T.numpy().squeeze()
    hop_samples = int(1.5 * TARGET_SR)

    chunks = []
    if len(data_enhanced) <= TARGET_SAMPLES:
        pad_len = TARGET_SAMPLES - len(data_enhanced)
        chunk = np.pad(data_enhanced, (0, pad_len), mode="constant")
        chunks.append(chunk)
    else:
        start_idx = 0
        while start_idx < len(data_enhanced):
            chunk = data_enhanced[start_idx : start_idx + TARGET_SAMPLES]
            if len(chunk) < TARGET_SAMPLES:
                if len(chunk) < int(0.5 * TARGET_SR) and len(chunks) > 0:
                    break
                pad_len = TARGET_SAMPLES - len(chunk)
                chunk = np.pad(chunk, (0, pad_len), mode="constant")
            chunks.append(chunk)
            start_idx += hop_samples

    return chunks


def extract_acoustic_features(y, sr=TARGET_SR):
    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
        features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta_mfcc[i])
        features[f"delta_mfcc_{i+1}_std"] = np.std(delta_mfcc[i])
        features[f"delta2_mfcc_{i+1}_mean"] = np.mean(delta2_mfcc[i])
        features[f"delta2_mfcc_{i+1}_std"] = np.std(delta2_mfcc[i])

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    features["spectral_centroid_mean"] = np.mean(cent)
    features["spectral_centroid_std"] = np.std(cent)
    features["spectral_rolloff_mean"] = np.mean(rolloff)
    features["spectral_rolloff_std"] = np.std(rolloff)
    features["spectral_bandwidth_mean"] = np.mean(bw)
    features["spectral_bandwidth_std"] = np.std(bw)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    features["rms_max"] = np.max(rms)

    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    f0 = librosa.yin(y, fmin=50, fmax=500, frame_length=2048)
    f0 = f0[~np.isnan(f0)]

    if len(f0) > 0:
        features["pitch_mean"] = np.mean(f0)
        features["pitch_std"] = np.std(f0)
        features["pitch_min"] = np.min(f0)
        features["pitch_max"] = np.max(f0)
        features["pitch_range"] = np.max(f0) - np.min(f0)
    else:
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0
        features["pitch_range"] = 0.0

    return features


def extract_mel_spectrogram(y, sr=TARGET_SR, n_mels=128, n_fft=1024, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    flat_mel = log_mel_spec.flatten()
    return {f"mel_{i}": val for i, val in enumerate(flat_mel)}


def extract_hubert_features(y, processor, hubert_model, device, sr=TARGET_SR):
    if processor is None or hubert_model is None:
        return {}

    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = hubert_model(input_values)

    hidden_states = outputs.last_hidden_state
    emb_mean = hidden_states.mean(dim=1).squeeze().cpu().numpy()

    return {f"hubert_{i}": val for i, val in enumerate(emb_mean)}


MODELS_DIR = os.path.join(settings.BASE_DIR, "trained_models")
EMOTIONS = ["Anger", "Happiness", "Sadness", "Neutral", "Fear", "Surprise"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert_model.eval()
    hubert_model.to(device)
except Exception as e:
    print(f"Помилка завантаження базового HuBERT: {e}")
    processor, hubert_model = None, None

LOADED_MODELS = {}


def index(request):
    return render(request, "index.html")


def get_models(request):
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = [
        f
        for f in os.listdir(MODELS_DIR)
        if f.endswith((".pth", ".pkl", ".h5", ".joblib"))
    ]
    return JsonResponse({"models": models})


@csrf_exempt
def predict_emotion(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests allowed"}, status=405)

    audio_file = request.FILES.get("audio")
    model_name = request.POST.get("model_name")

    if not audio_file or not model_name:
        return JsonResponse({"error": "Missing audio or model_name"}, status=400)

    try:
        chunks = process_and_chunk_in_memory(audio_file)
        if not chunks:
            return JsonResponse(
                {"error": "Audio contains only silence or is unreadable"}, status=400
            )

        chunk_probabilities = []
        model_path = os.path.join(MODELS_DIR, model_name)

        if model_name not in LOADED_MODELS:
            if "HuBERT" in model_name:
                config = AutoConfig.from_pretrained(
                    "facebook/hubert-base-ls960", num_labels=6
                )
                model = AutoModelForAudioClassification.from_pretrained(
                    "facebook/hubert-base-ls960", config=config
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                LOADED_MODELS[model_name] = {"model": model, "type": "hubert"}

            elif "CRNN" in model_name:
                model = ImprovedEmotionCRNN(num_stats_features=862, num_classes=6)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()

                scaler_path = os.path.join(MODELS_DIR, "standard_scaler.pkl")
                scaler = (
                    joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                )
                if not scaler:
                    print("standard_scaler.pkl не знайдено!")

                LOADED_MODELS[model_name] = {
                    "model": model,
                    "type": "crnn",
                    "scaler": scaler,
                }

        active_model_data = LOADED_MODELS[model_name]
        model = active_model_data["model"]
        model_type = active_model_data["type"]

        for i, chunk in enumerate(chunks):
            if model_type == "hubert":
                inputs = processor(
                    chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
                )
                input_values = inputs.input_values.to(device)

                with torch.no_grad():
                    outputs = model(input_values)

                probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
                chunk_probabilities.append(probs)

            elif model_type == "crnn":
                stats_dict = extract_acoustic_features(chunk)
                mel_dict = extract_mel_spectrogram(chunk)
                hubert_dict = extract_hubert_features(
                    chunk, processor, hubert_model, device
                )

                mel_array = np.array(list(mel_dict.values()), dtype=np.float32)
                mel_tensor = torch.tensor(mel_array).view(1, 1, 128, 94).to(device)

                combined_stats = {**stats_dict, **hubert_dict}
                stats_array = np.array(
                    list(combined_stats.values()), dtype=np.float32
                ).reshape(1, -1)

                scaler = active_model_data.get("scaler")
                if scaler:
                    stats_array = scaler.transform(stats_array)

                stats_tensor = torch.tensor(stats_array).to(device)

                with torch.no_grad():
                    logits = model(mel_tensor, stats_tensor)

                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                chunk_probabilities.append(probs)

        avg_probs = np.mean(chunk_probabilities, axis=0)
        pred_idx = np.argmax(avg_probs)
        final_emotion = EMOTIONS[pred_idx]
        confidence = float(avg_probs[pred_idx])
        prob_dict = {EMOTIONS[i]: float(avg_probs[i]) for i in range(len(EMOTIONS))}

        return JsonResponse(
            {
                "emotion": final_emotion,
                "confidence": f"{confidence:.2f}",
                "probabilities": prob_dict,
            }
        )

    except Exception as e:
        print(f"Помилка в predict_emotion: {e}")
        return JsonResponse({"error": str(e)}, status=500)
