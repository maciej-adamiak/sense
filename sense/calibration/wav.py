from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def wav_to_patches(
    wav_path: Path,
    mel_spectrogram: T.MelSpectrogram,
    sample_rate: int = 16000,
    log_offset=0.001,
    patch_frames: int = 96,
) -> list[np.ndarray]:
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    spec = mel_spectrogram(waveform).squeeze(0)
    log_mel = torch.log(spec + log_offset).transpose(0, 1)  # (frames, n_mels)

    patches = []

    for start in range(0, log_mel.shape[0] - patch_frames + 1, patch_frames):
        patch = log_mel[start : start + patch_frames]
        patches.append(patch.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32))

    return patches
