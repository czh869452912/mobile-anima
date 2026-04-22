from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_RESOLUTIONS = {(1024, 1024), (768, 1024), (1024, 768)}
SUPPORTED_STEPS = {8, 12, 20}
MIN_CFG = 3.0
MAX_CFG = 7.0
DEFAULT_MAX_TOKENS = 256

DEFAULT_ANIMA_MODEL_ID = "circlestone-labs/Anima"
DEFAULT_TEXT_ENCODER_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_VAE_MODEL_ID = "circlestone-labs/Anima"

# Anima is based on Cosmos-2B architecture (NVIDIA Cosmos-Predict2-2B-Text2Image)
ANIMA_DIFFUSION_DIM = 2048
ANIMA_NUM_LAYERS = 24
ANIMA_NUM_HEADS = 32

LATENT_CHANNELS = 4

# Denoiser ONNX tensor naming conventions
DENOISER_INPUT_LATENT = "latent"
DENOISER_INPUT_TIMESTEP = "timestep"
DENOISER_INPUT_COND = "cond"
DENOISER_INPUT_UNCOND = "uncond"
DENOISER_OUTPUT_NOISE_PRED = "noise_pred"

# QNN HTP operator support reference version
QNN_HTP_OP_SUPPORT_VERSION = "2.28"

SUBMODEL_PATHS = {
    "diffusion": "split_files/diffusion_models/anima-preview.safetensors",
    "text_encoder": "split_files/text_encoders/qwen_3_06b_base.safetensors",
    "vae": "split_files/vae/qwen_image_vae.safetensors",
}


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
        return (1, LATENT_CHANNELS, self.height // 8, self.width // 8)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = DEFAULT_ANIMA_MODEL_ID
    text_encoder_model_id: str = DEFAULT_TEXT_ENCODER_MODEL_ID
    vae_model_id: str = DEFAULT_VAE_MODEL_ID
    submodel_paths: dict[str, str] = field(default_factory=lambda: dict(SUBMODEL_PATHS))
    torch_dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path
    model_config: ModelConfig = field(default_factory=ModelConfig)
    resolutions: list[tuple[int, int]] = field(
        default_factory=lambda: list(SUPPORTED_RESOLUTIONS)
    )
    max_tokens: int = DEFAULT_MAX_TOKENS
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.resolutions:
            raise ValueError("resolutions must not be empty")
        for w, h in self.resolutions:
            if (w, h) not in SUPPORTED_RESOLUTIONS:
                raise ValueError(f"Unsupported resolution: {(w, h)}")

    @property
    def denoiser_onnx_path(self) -> Path:
        return self.output_dir / "denoiser.onnx"

    @property
    def text_encoder_onnx_path(self) -> Path:
        return self.output_dir / "text_encoder.onnx"

    @property
    def vae_onnx_path(self) -> Path:
        return self.output_dir / "vae.onnx"

    @property
    def denoiser_qdq_path(self) -> Path:
        return self.output_dir / "denoiser_qdq.onnx"

    @property
    def build_dir(self) -> Path:
        return self.output_dir / "build"

    @property
    def calibration_data_path(self) -> Path:
        return self.output_dir / "calibration_data.npz"
