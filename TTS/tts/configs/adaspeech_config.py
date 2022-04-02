from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class AdaSpeechConfig(BaseTTSConfig):
    model: str = "adaspeech"