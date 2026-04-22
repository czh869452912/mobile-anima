from anima_host.config import ModelConfig
from anima_host.model_loader import AnimaModelBundle, ArchitectureSummary, load_anima_bundle


def test_bundle_fallback_when_no_weights():
    config = ModelConfig()
    bundle = AnimaModelBundle(config)
    bundle.load()

    assert not bundle.available
    assert bundle.load_error is not None
    assert bundle.text_encoder is not None
    assert bundle.denoiser is not None
    assert bundle.vae_decoder is not None
    assert bundle.tokenizer is not None
    assert bundle.scheduler is not None


def test_bundle_architecture_summary():
    bundle = AnimaModelBundle()
    summary = bundle.architecture_summary()

    assert isinstance(summary, ArchitectureSummary)
    assert summary.backbone == "NVIDIA Cosmos-Predict2-2B-Text2Image (DiT)"
    assert summary.params_billion == 2.0
    assert summary.num_layers == 24
    assert summary.hidden_dim == 2048
    assert summary.num_heads == 32
    assert summary.latent_channels == 4
    assert summary.text_encoder_name == "Qwen3-0.6B"
    assert summary.vae_name == "Qwen-Image VAE"


def test_bundle_describe_contains_key_info():
    bundle = AnimaModelBundle()
    desc = bundle.describe()

    assert "NVIDIA Cosmos-Predict2-2B-Text2Image" in desc
    assert "2.0B" in desc
    assert "24" in desc
    assert "2048" in desc
    assert "Qwen3-0.6B" in desc
    assert "Weights available: False" in desc


def test_load_anima_bundle_returns_bundle():
    bundle = load_anima_bundle()
    assert isinstance(bundle, AnimaModelBundle)
    assert bundle.text_encoder is not None
    assert bundle.denoiser is not None
    assert bundle.vae_decoder is not None


def test_dummy_text_encoder_produces_expected_shape():
    bundle = load_anima_bundle()
    tokenizer_result = bundle.tokenizer("a cat", return_tensors=None)
    input_ids = tokenizer_result["input_ids"]

    encoder_out = bundle.text_encoder(input_ids)
    hidden = encoder_out.last_hidden_state

    assert hidden.shape[0] == 1
    assert hidden.shape[1] == 256
    assert hidden.shape[2] == 16


def test_dummy_denoiser_preserves_latent_shape():
    bundle = load_anima_bundle()
    import numpy as np

    latent = np.zeros((1, 4, 128, 128), dtype=np.float32)
    timestep = np.zeros((1,), dtype=np.float32)
    cond = np.zeros((1, 256, 16), dtype=np.float32)
    uncond = np.zeros((1, 256, 16), dtype=np.float32)

    out = bundle.denoiser(latent, timestep, cond, uncond)
    assert out.shape == (1, 4, 128, 128)


def test_dummy_vae_decoder_expands_latent_to_image():
    bundle = load_anima_bundle()
    import numpy as np

    latent = np.zeros((1, 4, 128, 128), dtype=np.float32)
    out = bundle.vae_decoder(latent)
    image = out.sample

    assert image.shape == (1, 3, 1024, 1024)


def test_dummy_tokenizer_batching():
    bundle = load_anima_bundle()
    result = bundle.tokenizer(["prompt one", "prompt two"])

    assert result["input_ids"].shape[0] == 2
    assert result["attention_mask"].shape[0] == 2
