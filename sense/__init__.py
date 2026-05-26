import logging
from pathlib import Path

from semver import Version

__author__ = 'Maciej Adamiak'
__version__: Version = Version.parse('0.1.0')

project_root_dir = Path(__file__).parent.parent


class AnsiColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        start_style = {
            'DEBUG': '\033[90m',
            'INFO': '\033[0m',
            'WARNING': '\033[93m',
            'ERROR': '\033[31m',
            'CRITICAL': '\033[91m\033[91m',
        }.get(record.levelname, '\033[0m')
        return f'{start_style}{super().format(record)}\033[0m'


handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(
    AnsiColorFormatter('{asctime} | {levelname:<8s} | {name:<25s} | {message}', style='{'),
)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
