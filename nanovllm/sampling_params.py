from dataclasses import dataclass
import math


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if not isinstance(self.ignore_eos, bool):
            raise ValueError("ignore_eos must be boolean")
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")
