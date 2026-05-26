import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests


def download_esc_50(samples_dir: Path, n: int, seed: int = 42, timeout: int = 60) -> list[Path]:
    zip_path = samples_dir / 'esc50.zip'

    if not zip_path.exists():
        response = requests.get('https://codeload.github.com/karolpiczak/ESC-50/zip/master', timeout=timeout)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if '/audio/' in member and member.endswith('.wav'):
                    (temp_dir / Path(member).name).write_bytes(zf.read(member))
                elif member.endswith('esc50.csv'):
                    (temp_dir / 'esc50.csv').write_bytes(zf.read(member))

        sampled = pd.read_csv(temp_dir / 'esc50.csv').sample(n, random_state=seed)
        return [shutil.copy2(temp_dir / f, samples_dir / f) and (samples_dir / f) for f in sampled['filename']]
