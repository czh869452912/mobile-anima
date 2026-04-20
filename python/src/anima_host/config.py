from dataclasses import dataclass

SUPPORTED_RESOLUTIONS = {(1024, 1024), (768, 1024), (1024, 768)}
SUPPORTED_STEPS = {8, 12, 20}
MIN_CFG = 3.0
MAX_CFG = 7.0
DEFAULT_MAX_TOKENS = 256


@dataclass(frozen=True)
class GenerationConfig:
    width: int
    height: int
    steps: int
    cfg: float
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if (self.width, self.height) not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Unsupported resolution: {(self.width, self.height)}"
            )
        if self.steps not in SUPPORTED_STEPS:
            raise ValueError(f"Unsupported steps: {self.steps}")
        if not MIN_CFG <= self.cfg <= MAX_CFG:
            raise ValueError("cfg must be between 3.0 and 7.0")
        if self.max_tokens != DEFAULT_MAX_TOKENS:
            raise ValueError(
                f"max_tokens must remain fixed at {DEFAULT_MAX_TOKENS} for v1"
            )

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return (1, 4, self.height // 8, self.width // 8)
