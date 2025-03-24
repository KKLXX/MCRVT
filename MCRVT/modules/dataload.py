import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
from sklearn.model_selection import KFold
import random
from functools import wraps


def protected_loader(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            return None

    return wrapper


class SERDataset(Dataset):
    def __init__(self, audio_paths, labels, max_length=7, sr=16000, train=True):
        if not audio_paths or not labels or len(audio_paths) != len(labels):
            raise ValueError("Invalid input: audio_paths and labels must be non-empty and of equal length")

        self.audio_paths = audio_paths
        self.labels = labels
        self.max_length = max(1, min(max_length, 10))  # Constrained to 1-10 seconds
        self.sr = max(8000, min(sr, 48000))  # Constrained to 8k-48kHz
        self.train = train

        self.n_fft = 512
        self.hop_length = 160
        self.n_mels = 64

        try:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio transforms: {str(e)}")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        try:
            # Validate index
            if idx >= len(self.audio_paths):
                raise IndexError("Index out of bounds")

            # Load and validate audio file
            waveform, sample_rate = self._load_audio(self.audio_paths[idx])

            # Process waveform
            waveform = self._process_waveform(waveform, sample_rate)

            # Convert to log-Mel spectrogram
            log_mel = self._waveform_to_logmel(waveform)

            # Get and validate label
            label = self._get_label(idx)

            return log_mel.squeeze(0), label

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # Return zero tensor and default label on error
            dummy_spec = torch.zeros(self.n_mels,
                                     int(self.sr * self.max_length / self.hop_length) + 1)
            return dummy_spec, 0

    def _load_audio(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sample_rate = torchaudio.load(path)
        if waveform.nelement() == 0:
            raise ValueError(f"Empty audio file: {path}")

        return waveform, sample_rate

    def _process_waveform(self, waveform, sample_rate):
        if sample_rate != self.sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sr)
            waveform = resampler(waveform)

        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        target_length = self.sr * self.max_length
        if waveform.shape[1] > target_length:
            start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        elif waveform.shape[1] < target_length:
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        if self.train:
            waveform = self._augment_waveform(waveform)

        return waveform

    def _waveform_to_logmel(self, waveform):
        mel_spec = self.mel_transform(waveform)
        log_mel = self.amplitude_to_db(mel_spec)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        return log_mel

    def _get_label(self, idx):
        label = self.labels[idx]
        if not isinstance(label, (int, float)):
            try:
                label = int(label)
            except:
                label = 0
        return label

    def _augment_waveform(self, waveform):
        try:
            if random.random() > 0.5:
                gain = random.uniform(0.8, 1.2)
                waveform = waveform * gain

            if random.random() > 0.7:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise

            if random.random() > 0.5:
                mask_length = random.randint(100, 500)
                start = random.randint(0, max(0, waveform.shape[1] - mask_length))
                waveform[:, start:start + mask_length] = 0

            return waveform
        except:
            return waveform  # Return original if augmentation fails


class SERDataLoaderFactory:
    def __init__(self, data_dir, batch_size=32, num_workers=4, k_fold=5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_fold = k_fold
        self.audio_paths, self.labels, self.speaker_ids = self._load_metadata()

    def _load_metadata(self):
        try:
            return load_metadata(self.data_dir)
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            return [], [], []

    @protected_loader
    def get_train_loader(self, fold_idx, speaker_independent=False):

        if not self.audio_paths:
            raise ValueError("No valid audio paths available")

        if speaker_independent:
            return self._get_speaker_independent_loaders(fold_idx)[0]
        else:
            return self._get_standard_loaders(fold_idx)[0]

    @protected_loader
    def get_test_loader(self, fold_idx, speaker_independent=False):
        if not self.audio_paths:
            raise ValueError("No valid audio paths available")

        if speaker_independent:
            return self._get_speaker_independent_loaders(fold_idx)[1]
        else:
            return self._get_standard_loaders(fold_idx)[1]

    def _get_standard_loaders(self, fold_idx):
        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        all_indices = list(range(len(self.audio_paths)))

        for i, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
            if i == fold_idx:
                train_paths = [self.audio_paths[i] for i in train_idx]
                train_labels = [self.labels[i] for i in train_idx]
                val_paths = [self.audio_paths[i] for i in val_idx]
                val_labels = [self.labels[i] for i in val_idx]

                train_dataset = SERDataset(train_paths, train_labels, train=True)
                val_dataset = SERDataset(val_paths, val_labels, train=False)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True
                )

                return train_loader, val_loader

        raise ValueError(f"Invalid fold index: {fold_idx}")

    def _get_speaker_independent_loaders(self, fold_idx):
        unique_speakers = sorted(list(set(self.speaker_ids)))
        k_fold = min(self.k_fold, len(unique_speakers))

        if fold_idx >= k_fold:
            raise ValueError(f"Fold index {fold_idx} exceeds available speakers {k_fold}")

        val_speakers = unique_speakers[fold_idx::k_fold]

        train_paths, train_labels = [], []
        val_paths, val_labels = [], []

        for path, label, speaker in zip(self.audio_paths, self.labels, self.speaker_ids):
            if speaker in val_speakers:
                val_paths.append(path)
                val_labels.append(label)
            else:
                train_paths.append(path)
                train_labels.append(label)

        train_dataset = SERDataset(train_paths, train_labels, train=True)
        val_dataset = SERDataset(val_paths, val_labels, train=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


def load_metadata(data_dir):

    audio_paths = []
    labels = []
    speaker_ids = []

    try:
        for session_dir in os.listdir(data_dir):
            session_path = os.path.join(data_dir, session_dir)
            if not os.path.isdir(session_path):
                continue

            label_file = os.path.join(session_path, "labels.csv")
            if not os.path.exists(label_file):
                continue

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        audio_path = os.path.join(session_path, parts[0])
                        if os.path.exists(audio_path):
                            audio_paths.append(audio_path)
                            labels.append(int(parts[1]))  # Convert to int
                            speaker_ids.append(parts[2])

    except Exception as e:
        print(f"Error loading metadata: {str(e)}")

    return audio_paths, labels, speaker_ids