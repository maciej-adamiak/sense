import logging
import urllib.request
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import torchaudio.transforms as T
from jsonargparse import auto_cli
from onnx import version_converter
from tqdm import tqdm

from sense.calibration.samples import download_esc_50
from sense.calibration.wav import wav_to_patches

logger = logging.getLogger(__name__)

ONNX_ZIP_URL = (
    'https://qaihub-public-assets.s3.us-west-2.amazonaws.com/' 'qai-hub-models/models/yamnet/releases/v0.48.0/'
)

YamNetFlavour = Literal['yamnet-onnx-float']


class PreparationCLI:
    def __init__(self, workspace_dir: Path = Path('workspace')):
        calibration_dir = workspace_dir / 'calibration'
        model_dir = workspace_dir / 'model'
        samples_dir = workspace_dir / 'samples'

        workspace_dir.mkdir(exist_ok=True)

        self.workspace_dir = workspace_dir
        self.calibration_dir = calibration_dir
        self.model_dir = model_dir
        self.samples_dir = samples_dir

    def download_yamnet_onnx(self, model_flavour: YamNetFlavour = 'yamnet-onnx-float', target_opset: int = 17):
        self.model_dir.mkdir(exist_ok=True)
        model_path = self.model_dir / model_flavour / 'yamnet.onnx'

        zip_url = ONNX_ZIP_URL + f'{model_flavour}.zip'

        if not (self.model_dir / model_flavour).exists():
            with TemporaryDirectory() as tmp_dir:
                zip_path = Path(tmp_dir) / f'{model_flavour}.zip'

                logging.info(f'Downloading ONNX bundle from {zip_url}')
                urllib.request.urlretrieve(zip_url, zip_path)

                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(self.model_dir)

                zip_path.unlink(missing_ok=True)

        model = onnx.load(model_path)
        converted = version_converter.convert_version(model, target_opset)
        onnx.checker.check_model(converted)
        onnx.save(converted, model_path)

        logger.info(f'ONNX ready at {self.model_dir / model_flavour}')

    def build_calibration_set(
        self,
        model_flavour: YamNetFlavour = 'yamnet-onnx-float',
        sample_rate: int = 16000,
        patch_frames: int = 96,
        sample_audio_files: int = 0,
    ):
        self.calibration_dir.mkdir(exist_ok=True)

        if sample_audio_files > 0:
            logging.info(f'Downloading {sample_audio_files} sample audio files')
            self.samples_dir.mkdir(exist_ok=True)
            download_esc_50(self.samples_dir, sample_audio_files)

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=int(0.025 * sample_rate),
            hop_length=int(0.010 * sample_rate),
            f_min=125.0,
            f_max=7500,
            n_mels=64,
            power=2.0,
            center=False,
        )

        samples = sorted(self.samples_dir.glob('*.wav'))
        if not samples:
            raise RuntimeError(f'No .wav calibration samples available in {self.samples_dir}.')

        model = onnx.load(str(self.model_dir / model_flavour / 'yamnet.onnx'))

        input_name = model.graph.input[0].name
        logger.info(f'ONNX input tensor name: {input_name}')

        idx = 0
        for idx, wav_path in tqdm(enumerate(samples), desc='Building calibration set'):
            for patch in wav_to_patches(wav_path=wav_path, mel_spectrogram=mel_spectrogram, patch_frames=patch_frames):
                np.savez(self.calibration_dir / f'sample_{idx:04d}.npz', **{input_name: patch})

        logger.info(f'Wrote {idx} calibration patches to {self.calibration_dir}')

    def dry_run(
        self,
        model_flavour: YamNetFlavour = 'yamnet-onnx-float',
    ):
        inference_session = ort.InferenceSession(
            str(self.model_dir / model_flavour / 'yamnet.onnx'), providers=['CPUExecutionProvider']
        )
        sample = next(self.calibration_dir.glob('*.npz'))
        name = inference_session.get_inputs()[0].name
        arr = np.load(sample)[name]
        out = inference_session.run(None, {name: arr})[0]
        top5 = out[0].argsort()[-5:][::-1]

        logger.info(f'Smoke test OK. Output shape: {out.shape}, top-5 idx: {top5}')


if __name__ == '__main__':
    auto_cli(PreparationCLI)
