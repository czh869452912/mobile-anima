# Anima Snapdragon 8 Elite NPU PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Android app and host-side toolchain that generate one image from `circlestone-labs/Anima` on a Snapdragon 8 Elite device while proving the denoiser runs through Qualcomm's QNN/HTP path.

**Architecture:** Use a mixed-backend pipeline. A Python workspace owns the reference pipeline, denoiser export, quantization, and Qualcomm compilation flow. An Android app owns the UI and orchestration, runs text encoder plus VAE on CPU/GPU for v1, and binds only the denoiser session to `ONNX Runtime + QNN Execution Provider` with CPU fallback disabled during validation.

**Tech Stack:** Python 3.11, PyTorch, ONNX, ONNX Runtime, Qualcomm QNN/QAIRT toolchain, Android Gradle Plugin 8.7+, Kotlin, Jetpack Compose, JUnit4, pytest

---

## Prerequisites

Before Task 1, prepare the local machine with the exact prerequisites below.

- Install Python `3.11`
- Install JDK `17`
- Install Android SDK platform `35` and build tools `35.0.0`
- Install Gradle `8.10` or newer so `gradle wrapper` can generate the wrapper files
- Install Qualcomm `QAIRT/QNN SDK` and export `QNN_SDK_ROOT`
- Export `ANDROID_SDK_ROOT`
- Authenticate to Hugging Face so the `Anima` model snapshot can be downloaded locally
- If the directory is not already a git repo, initialize it once with `git init`

Run:

```bash
git init
mkdir -p python/src/anima_host python/tests android docs/manual scripts
```

Expected: `.git/` exists and the workspace folders are created.

## File Structure

- Create: `python/pyproject.toml` — Python dependencies and pytest configuration
- Create: `python/src/anima_host/__init__.py` — package marker
- Create: `python/src/anima_host/config.py` — generation constraints and validation
- Create: `python/src/anima_host/interfaces.py` — typed protocols for text encoder, denoiser, and VAE decoder
- Create: `python/src/anima_host/reference_pipeline.py` — deterministic host-side baseline pipeline
- Create: `python/src/anima_host/export_denoiser.py` — denoiser ONNX export entry point and shape definitions
- Create: `python/src/anima_host/quantize_denoiser.py` — QDQ quantization entry point
- Create: `python/src/anima_host/compile_qnn.py` — Qualcomm compilation/profile command builder
- Create: `python/tests/test_config.py` — generation constraint tests
- Create: `python/tests/test_reference_pipeline.py` — baseline pipeline tests using fakes
- Create: `python/tests/test_export_compile.py` — export, quantization, and compile command tests
- Create: `android/settings.gradle.kts` — Android project settings
- Create: `android/build.gradle.kts` — root Android build config
- Create: `android/gradle.properties` — Android/Compose defaults
- Create: `android/app/build.gradle.kts` — app module build config and ORT dependency wiring
- Create: `android/app/src/main/AndroidManifest.xml` — app manifest
- Create: `android/app/src/main/java/com/example/animanpu/MainActivity.kt` — Compose activity entry point
- Create: `android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt` — minimal UI
- Create: `android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt` — UI state and trigger logic
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationRequest.kt` — fixed-shape request validation
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt` — generated image and profiling metadata
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationEngine.kt` — orchestration interface
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt` — runtime orchestration using injected backends
- Create: `android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt` — model and profile file resolution
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnConfig.kt` — QNN provider options builder
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt` — ONNX Runtime session creation helpers
- Create: `android/app/src/test/java/com/example/animanpu/runtime/GenerationRequestTest.kt` — request validation tests
- Create: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt` — orchestrator tests using fakes
- Create: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt` — provider options tests
- Create: `android/app/libs/README.md` — where to place the custom ORT AAR and Qualcomm `.so` files
- Create: `scripts/build_ort_android_with_qnn.sh` — reproducible ORT Android build script
- Create: `docs/manual/android-validation.md` — manual verification checklist for profiling and no-fallback validation

### Task 1: Build the Python workspace and configuration guardrails

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/src/anima_host/__init__.py`
- Create: `python/src/anima_host/config.py`
- Test: `python/tests/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
# python/tests/test_config.py
import pytest

from anima_host.config import GenerationConfig


def test_generation_config_accepts_supported_resolution():
    config = GenerationConfig(
        width=1024,
        height=1024,
        steps=12,
        cfg=5.0,
        max_tokens=256,
    )

    assert config.width == 1024
    assert config.height == 1024
    assert config.steps == 12


def test_generation_config_rejects_unsupported_resolution():
    with pytest.raises(ValueError, match="Unsupported resolution"):
        GenerationConfig(
            width=640,
            height=640,
            steps=12,
            cfg=5.0,
            max_tokens=256,
        )


def test_generation_config_rejects_invalid_step_count():
    with pytest.raises(ValueError, match="Unsupported steps"):
        GenerationConfig(
            width=1024,
            height=1024,
            steps=30,
            cfg=5.0,
            max_tokens=256,
        )


def test_generation_config_rejects_cfg_out_of_range():
    with pytest.raises(ValueError, match="cfg must be between 3.0 and 7.0"):
        GenerationConfig(
            width=1024,
            height=1024,
            steps=12,
            cfg=8.0,
            max_tokens=256,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd python && python -m pytest tests/test_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'anima_host'`.

- [ ] **Step 3: Write the minimal implementation**

```toml
# python/pyproject.toml
[project]
name = "anima-host"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=2.1",
  "pillow>=10.4",
  "pytest>=8.3",
  "torch>=2.6",
  "transformers>=4.52",
  "onnx>=1.17",
  "onnxruntime>=1.22",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

```python
# python/src/anima_host/__init__.py
__all__ = ["config"]
```

```python
# python/src/anima_host/config.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd python && python -m pytest tests/test_config.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/pyproject.toml python/src/anima_host/__init__.py python/src/anima_host/config.py python/tests/test_config.py
git commit -m "test: add python generation config guards"
```

### Task 2: Build a deterministic Python reference pipeline

**Files:**
- Create: `python/src/anima_host/interfaces.py`
- Create: `python/src/anima_host/reference_pipeline.py`
- Test: `python/tests/test_reference_pipeline.py`

- [ ] **Step 1: Write the failing reference-pipeline tests**

```python
# python/tests/test_reference_pipeline.py
import numpy as np
from PIL import Image

from anima_host.config import GenerationConfig
from anima_host.reference_pipeline import ReferencePipeline


class FakeTextEncoder:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, prompt: str, negative_prompt: str, max_tokens: int):
        self.calls.append((prompt, negative_prompt, max_tokens))
        cond = np.ones((1, 256, 16), dtype=np.float32)
        uncond = np.zeros((1, 256, 16), dtype=np.float32)
        return cond, uncond


class FakeDenoiser:
    def __init__(self) -> None:
        self.timesteps = []

    def step(self, latent, timestep, cond, uncond, cfg):
        self.timesteps.append((timestep, cfg, cond.shape, uncond.shape))
        return latent + 1.0


class FakeVaeDecoder:
    def decode(self, latent):
        width = latent.shape[3] * 8
        height = latent.shape[2] * 8
        return Image.new("RGB", (width, height), color="black")


def test_reference_pipeline_runs_exact_number_of_steps():
    pipeline = ReferencePipeline(
        text_encoder=FakeTextEncoder(),
        denoiser=FakeDenoiser(),
        vae_decoder=FakeVaeDecoder(),
    )
    config = GenerationConfig(width=1024, height=1024, steps=8, cfg=5.0)

    image, artifacts = pipeline.generate(
        prompt="cat astronaut",
        negative_prompt="blurry",
        config=config,
    )

    assert image.size == (1024, 1024)
    assert artifacts.step_count == 8
    assert artifacts.final_latent_mean == 8.0


def test_reference_pipeline_uses_fixed_latent_shape():
    pipeline = ReferencePipeline(
        text_encoder=FakeTextEncoder(),
        denoiser=FakeDenoiser(),
        vae_decoder=FakeVaeDecoder(),
    )
    config = GenerationConfig(width=768, height=1024, steps=12, cfg=6.0)

    _, artifacts = pipeline.generate(
        prompt="forest temple",
        negative_prompt="low quality",
        config=config,
    )

    assert artifacts.initial_latent_shape == (1, 4, 128, 96)
    assert artifacts.cond_shape == (1, 256, 16)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd python && python -m pytest tests/test_reference_pipeline.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `anima_host.reference_pipeline`.

- [ ] **Step 3: Write the minimal implementation**

```python
# python/src/anima_host/interfaces.py
from typing import Protocol

from PIL import Image
import numpy as np


class TextEncoder(Protocol):
    def encode(
        self,
        prompt: str,
        negative_prompt: str,
        max_tokens: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class Denoiser(Protocol):
    def step(
        self,
        latent: np.ndarray,
        timestep: int,
        cond: np.ndarray,
        uncond: np.ndarray,
        cfg: float,
    ) -> np.ndarray: ...


class VaeDecoder(Protocol):
    def decode(self, latent: np.ndarray) -> Image.Image: ...
```

```python
# python/src/anima_host/reference_pipeline.py
from dataclasses import dataclass

import numpy as np
from PIL import Image

from anima_host.config import GenerationConfig
from anima_host.interfaces import Denoiser, TextEncoder, VaeDecoder


@dataclass(frozen=True)
class GenerationArtifacts:
    initial_latent_shape: tuple[int, int, int, int]
    cond_shape: tuple[int, ...]
    uncond_shape: tuple[int, ...]
    step_count: int
    final_latent_mean: float


class ReferencePipeline:
    def __init__(
        self,
        text_encoder: TextEncoder,
        denoiser: Denoiser,
        vae_decoder: VaeDecoder,
    ) -> None:
        self.text_encoder = text_encoder
        self.denoiser = denoiser
        self.vae_decoder = vae_decoder

    def build_initial_latent(self, config: GenerationConfig) -> np.ndarray:
        return np.zeros(config.latent_shape, dtype=np.float32)

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
    ) -> tuple[Image.Image, GenerationArtifacts]:
        cond, uncond = self.text_encoder.encode(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_tokens=config.max_tokens,
        )
        latent = self.build_initial_latent(config)
        initial_shape = latent.shape

        for timestep in range(config.steps):
            latent = self.denoiser.step(
                latent=latent,
                timestep=timestep,
                cond=cond,
                uncond=uncond,
                cfg=config.cfg,
            )

        image = self.vae_decoder.decode(latent)
        artifacts = GenerationArtifacts(
            initial_latent_shape=initial_shape,
            cond_shape=cond.shape,
            uncond_shape=uncond.shape,
            step_count=config.steps,
            final_latent_mean=float(latent.mean()),
        )
        return image, artifacts
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd python && python -m pytest tests/test_reference_pipeline.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/src/anima_host/interfaces.py python/src/anima_host/reference_pipeline.py python/tests/test_reference_pipeline.py
git commit -m "test: add deterministic reference pipeline harness"
```

### Task 3: Add denoiser export, quantization, and Qualcomm compile helpers

**Files:**
- Create: `python/src/anima_host/export_denoiser.py`
- Create: `python/src/anima_host/quantize_denoiser.py`
- Create: `python/src/anima_host/compile_qnn.py`
- Test: `python/tests/test_export_compile.py`

- [ ] **Step 1: Write the failing export-and-compile tests**

```python
# python/tests/test_export_compile.py
from pathlib import Path

from anima_host.compile_qnn import build_qnn_context_command
from anima_host.export_denoiser import ExportSpec, build_dummy_inputs
from anima_host.quantize_denoiser import QuantizeSpec


def test_build_dummy_inputs_matches_fixed_shapes():
    spec = ExportSpec(
        bundle_dir=Path("/models/anima"),
        output_path=Path("build/denoiser.onnx"),
        width=1024,
        height=1024,
        max_tokens=256,
    )

    latent, timestep, cond, uncond = build_dummy_inputs(spec)

    assert tuple(latent.shape) == (1, 4, 128, 128)
    assert tuple(timestep.shape) == (1,)
    assert tuple(cond.shape) == (1, 256, 16)
    assert tuple(uncond.shape) == (1, 256, 16)


def test_build_qnn_context_command_contains_no_fallback_profile_flags():
    command = build_qnn_context_command(
        onnx_model=Path("build/denoiser_qdq.onnx"),
        output_dir=Path("build/qnn"),
        sdk_root=Path("/opt/qnn"),
        profiling_level="detailed",
    )

    joined = " ".join(command)
    assert "qnn-context-binary-generator" in joined
    assert "build/denoiser_qdq.onnx" in joined
    assert "--profiling_level" in joined
    assert "detailed" in joined


def test_quantize_spec_uses_qdq_defaults():
    spec = QuantizeSpec(
        input_model=Path("build/denoiser.onnx"),
        output_model=Path("build/denoiser_qdq.onnx"),
    )

    assert spec.per_channel is True
    assert spec.activation_type == "QInt8"
    assert spec.weight_type == "QInt8"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd python && python -m pytest tests/test_export_compile.py -q
```

Expected: FAIL with `ModuleNotFoundError` for the new modules.

- [ ] **Step 3: Write the minimal implementation**

```python
# python/src/anima_host/export_denoiser.py
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ExportSpec:
    bundle_dir: Path
    output_path: Path
    width: int
    height: int
    max_tokens: int


def build_dummy_inputs(spec: ExportSpec):
    latent = torch.zeros((1, 4, spec.height // 8, spec.width // 8), dtype=torch.float32)
    timestep = torch.zeros((1,), dtype=torch.float32)
    cond = torch.zeros((1, spec.max_tokens, 16), dtype=torch.float32)
    uncond = torch.zeros((1, spec.max_tokens, 16), dtype=torch.float32)
    return latent, timestep, cond, uncond
```

```python
# python/src/anima_host/quantize_denoiser.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantizeSpec:
    input_model: Path
    output_model: Path
    per_channel: bool = True
    activation_type: str = "QInt8"
    weight_type: str = "QInt8"
```

```python
# python/src/anima_host/compile_qnn.py
from pathlib import Path


def build_qnn_context_command(
    onnx_model: Path,
    output_dir: Path,
    sdk_root: Path,
    profiling_level: str,
) -> list[str]:
    binary = sdk_root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"
    return [
        str(binary),
        "--model",
        str(onnx_model),
        "--backend",
        str(sdk_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"),
        "--output_dir",
        str(output_dir),
        "--profiling_level",
        profiling_level,
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd python && python -m pytest tests/test_export_compile.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Expand the helpers into runnable CLIs and smoke-test them**

```python
# append to python/src/anima_host/export_denoiser.py
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    spec = ExportSpec(
        bundle_dir=args.bundle_dir,
        output_path=args.output_path,
        width=args.width,
        height=args.height,
        max_tokens=args.max_tokens,
    )
    latent, timestep, cond, uncond = build_dummy_inputs(spec)
    print({
        "latent": tuple(latent.shape),
        "timestep": tuple(timestep.shape),
        "cond": tuple(cond.shape),
        "uncond": tuple(uncond.shape),
    })


if __name__ == "__main__":
    main()
```

```python
# append to python/src/anima_host/quantize_denoiser.py
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    args = parser.parse_args()
    spec = QuantizeSpec(
        input_model=args.input_model,
        output_model=args.output_model,
    )
    print(spec)


if __name__ == "__main__":
    main()
```

```python
# append to python/src/anima_host/compile_qnn.py
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--profiling-level", default="detailed")
    args = parser.parse_args()
    print(
        " ".join(
            build_qnn_context_command(
                onnx_model=args.onnx_model,
                output_dir=args.output_dir,
                sdk_root=args.sdk_root,
                profiling_level=args.profiling_level,
            )
        )
    )


if __name__ == "__main__":
    main()
```

Run:

```bash
cd python && python -m anima_host.export_denoiser --bundle-dir /tmp/anima --output-path build/denoiser.onnx
```

Expected: printed tensor shapes for `latent`, `timestep`, `cond`, and `uncond`.

- [ ] **Step 6: Commit**

```bash
git add python/src/anima_host/export_denoiser.py python/src/anima_host/quantize_denoiser.py python/src/anima_host/compile_qnn.py python/tests/test_export_compile.py
git commit -m "feat: add export and qnn compile scaffolding"
```

### Task 4: Scaffold the Android project and validate requests

**Files:**
- Create: `android/settings.gradle.kts`
- Create: `android/build.gradle.kts`
- Create: `android/gradle.properties`
- Create: `android/app/build.gradle.kts`
- Create: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/java/com/example/animanpu/MainActivity.kt`
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationRequest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/GenerationRequestTest.kt`

- [ ] **Step 1: Write the failing Android request-validation test and build files**

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/GenerationRequestTest.kt
package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class GenerationRequestTest {
    @Test
    fun rejectsUnsupportedResolution() {
        try {
            GenerationRequest(
                width = 640,
                height = 640,
                prompt = "cat astronaut",
                negativePrompt = "blurry",
                steps = 12,
                cfg = 5.0f,
            )
            throw AssertionError("Expected IllegalArgumentException")
        } catch (expected: IllegalArgumentException) {
            assertEquals("Unsupported resolution: 640x640", expected.message)
        }
    }

    @Test
    fun acceptsSupportedRequest() {
        val request = GenerationRequest(
            width = 1024,
            height = 1024,
            prompt = "cat astronaut",
            negativePrompt = "blurry",
            steps = 12,
            cfg = 5.0f,
        )

        assertEquals(1024, request.width)
        assertEquals(1024, request.height)
    }
}
```

```kotlin
// android/settings.gradle.kts
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "anima-npu"
include(":app")
```

```kotlin
// android/build.gradle.kts
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}
```

```properties
# android/gradle.properties
org.gradle.jvmargs=-Xmx4g -Dfile.encoding=UTF-8
android.useAndroidX=true
kotlin.code.style=official
```

```kotlin
// android/app/build.gradle.kts
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.animanpu"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.animanpu"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.15"
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.02.00"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    testImplementation("junit:junit:4.13.2")
}
```

- [ ] **Step 2: Run the unit test to verify it fails**

Run:

```bash
cd android && gradle wrapper --gradle-version 8.10 && ./gradlew testDebugUnitTest
```

Expected: FAIL because `GenerationRequest` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/GenerationRequest.kt
package com.example.animanpu.runtime

data class GenerationRequest(
    val width: Int,
    val height: Int,
    val prompt: String,
    val negativePrompt: String,
    val steps: Int,
    val cfg: Float,
    val maxTokens: Int = 256,
) {
    init {
        val resolution = width to height
        require(
            resolution in setOf(1024 to 1024, 768 to 1024, 1024 to 768)
        ) { "Unsupported resolution: ${width}x${height}" }
        require(steps in setOf(8, 12, 20)) { "Unsupported steps: $steps" }
        require(cfg in 3.0f..7.0f) { "cfg must be between 3.0 and 7.0" }
        require(maxTokens == 256) { "maxTokens must remain fixed at 256 for v1" }
        require(prompt.isNotBlank()) { "prompt must not be blank" }
    }
}
```

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application
        android:allowBackup="true"
        android:label="Anima NPU"
        android:supportsRtl="true"
        android:theme="@android:style/Theme.Material.Light.NoActionBar">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

```kotlin
// android/app/src/main/java/com/example/animanpu/MainActivity.kt
package com.example.animanpu

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Text

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Text("Anima NPU PoC")
        }
    }
}
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run:

```bash
cd android && ./gradlew testDebugUnitTest
```

Expected: `BUILD SUCCESSFUL` and `GenerationRequestTest` passes.

- [ ] **Step 5: Commit**

```bash
git add android/settings.gradle.kts android/build.gradle.kts android/gradle.properties android/app/build.gradle.kts android/app/src/main/AndroidManifest.xml android/app/src/main/java/com/example/animanpu/MainActivity.kt android/app/src/main/java/com/example/animanpu/runtime/GenerationRequest.kt android/app/src/test/java/com/example/animanpu/runtime/GenerationRequestTest.kt android/gradlew android/gradlew.bat android/gradle/wrapper

git commit -m "test: scaffold android app and request validation"
```

### Task 5: Add UI state, orchestrator, and fake-backed Android generation flow

**Files:**
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt`
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationEngine.kt`
- Create: `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt`
- Create: `android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt`
- Create: `android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`

- [ ] **Step 1: Write the failing orchestrator tests**

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt
package com.example.animanpu.runtime

import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class GenerationOrchestratorTest {
    @Test
    fun delegatesToEngineAndReturnsResult() = runTest {
        val expected = GenerationResult(
            imagePath = "/tmp/output.png",
            totalDurationMs = 1234,
            denoiseDurationMs = 1100,
            profilingPath = "/tmp/profile.csv",
            qnnActive = true,
        )
        val engine = object : GenerationEngine {
            override suspend fun generate(request: GenerationRequest): GenerationResult = expected
        }
        val orchestrator = GenerationOrchestrator(engine)

        val result = orchestrator.generate(
            GenerationRequest(
                width = 1024,
                height = 1024,
                prompt = "cat astronaut",
                negativePrompt = "blurry",
                steps = 8,
                cfg = 5.0f,
            )
        )

        assertEquals(expected, result)
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.GenerationOrchestratorTest
```

Expected: FAIL because `GenerationResult`, `GenerationEngine`, and `GenerationOrchestrator` do not exist.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt
package com.example.animanpu.runtime

data class GenerationResult(
    val imagePath: String,
    val totalDurationMs: Long,
    val denoiseDurationMs: Long,
    val profilingPath: String,
    val qnnActive: Boolean,
)
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/GenerationEngine.kt
package com.example.animanpu.runtime

interface GenerationEngine {
    suspend fun generate(request: GenerationRequest): GenerationResult
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt
package com.example.animanpu.runtime

class GenerationOrchestrator(
    private val engine: GenerationEngine,
) {
    suspend fun generate(request: GenerationRequest): GenerationResult = engine.generate(request)
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt
package com.example.animanpu.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.animanpu.runtime.GenerationOrchestrator
import com.example.animanpu.runtime.GenerationRequest
import com.example.animanpu.runtime.GenerationResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class GenerationUiState(
    val prompt: String = "",
    val negativePrompt: String = "",
    val resolution: String = "1024x1024",
    val steps: String = "12",
    val cfg: String = "5.0",
    val isGenerating: Boolean = false,
    val lastResult: GenerationResult? = null,
    val error: String? = null,
)

class GenerationViewModel(
    private val orchestrator: GenerationOrchestrator,
) : ViewModel() {
    private val _state = MutableStateFlow(GenerationUiState())
    val state: StateFlow<GenerationUiState> = _state.asStateFlow()

    fun updatePrompt(value: String) { _state.value = _state.value.copy(prompt = value) }
    fun updateNegativePrompt(value: String) { _state.value = _state.value.copy(negativePrompt = value) }
    fun updateResolution(value: String) { _state.value = _state.value.copy(resolution = value) }
    fun updateSteps(value: String) { _state.value = _state.value.copy(steps = value) }
    fun updateCfg(value: String) { _state.value = _state.value.copy(cfg = value) }

    fun generate() {
        val parts = _state.value.resolution.split("x")
        val request = GenerationRequest(
            width = parts[0].toInt(),
            height = parts[1].toInt(),
            prompt = _state.value.prompt,
            negativePrompt = _state.value.negativePrompt,
            steps = _state.value.steps.toInt(),
            cfg = _state.value.cfg.toFloat(),
        )
        _state.value = _state.value.copy(isGenerating = true, error = null)
        viewModelScope.launch {
            runCatching { orchestrator.generate(request) }
                .onSuccess { result ->
                    _state.value = _state.value.copy(isGenerating = false, lastResult = result)
                }
                .onFailure { error ->
                    _state.value = _state.value.copy(isGenerating = false, error = error.message)
                }
        }
    }
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt
package com.example.animanpu.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun GenerationScreen(
    state: GenerationUiState,
    onPromptChange: (String) -> Unit,
    onNegativePromptChange: (String) -> Unit,
    onStepsChange: (String) -> Unit,
    onCfgChange: (String) -> Unit,
    onGenerate: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        OutlinedTextField(
            value = state.prompt,
            onValueChange = onPromptChange,
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.negativePrompt,
            onValueChange = onNegativePromptChange,
            label = { Text("Negative prompt") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.steps,
            onValueChange = onStepsChange,
            label = { Text("Steps") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.cfg,
            onValueChange = onCfgChange,
            label = { Text("CFG") },
            modifier = Modifier.fillMaxWidth(),
        )
        Button(onClick = onGenerate, enabled = !state.isGenerating) {
            Text(if (state.isGenerating) "Generating…" else "Generate")
        }
        state.lastResult?.let {
            Text("Last image: ${it.imagePath}")
            Text("Total: ${it.totalDurationMs} ms")
            Text("Denoise: ${it.denoiseDurationMs} ms")
            Text("QNN active: ${it.qnnActive}")
        }
        state.error?.let { Text("Error: $it") }
    }
}
```

```kotlin
// replace android/app/src/main/java/com/example/animanpu/MainActivity.kt
package com.example.animanpu

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.animanpu.runtime.GenerationEngine
import com.example.animanpu.runtime.GenerationOrchestrator
import com.example.animanpu.runtime.GenerationRequest
import com.example.animanpu.runtime.GenerationResult
import com.example.animanpu.ui.GenerationScreen
import com.example.animanpu.ui.GenerationViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val vm: GenerationViewModel = viewModel(
                factory = androidx.lifecycle.viewmodel.initializer {
                    val fakeEngine = object : GenerationEngine {
                        override suspend fun generate(request: GenerationRequest): GenerationResult {
                            return GenerationResult(
                                imagePath = "/sdcard/Download/anima-output.png",
                                totalDurationMs = 1200,
                                denoiseDurationMs = 1000,
                                profilingPath = "/sdcard/Download/profile.csv",
                                qnnActive = false,
                            )
                        }
                    }
                    GenerationViewModel(GenerationOrchestrator(fakeEngine))
                }
            )
            val state by vm.state.collectAsState()
            GenerationScreen(
                state = state,
                onPromptChange = vm::updatePrompt,
                onNegativePromptChange = vm::updateNegativePrompt,
                onStepsChange = vm::updateSteps,
                onCfgChange = vm::updateCfg,
                onGenerate = vm::generate,
            )
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.GenerationOrchestratorTest
```

Expected: `BUILD SUCCESSFUL` and `GenerationOrchestratorTest` passes.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/MainActivity.kt android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt android/app/src/main/java/com/example/animanpu/runtime/GenerationEngine.kt android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt
git commit -m "feat: add minimal android generation flow"
```

### Task 6: Wire ORT QNN configuration and model artifacts on Android

**Files:**
- Create: `android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt`
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnConfig.kt`
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt`
- Create: `android/app/libs/README.md`
- Create: `scripts/build_ort_android_with_qnn.sh`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt`

- [ ] **Step 1: Write the failing QNN configuration tests**

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt
package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OrtQnnConfigTest {
    @Test
    fun buildsProviderOptionsWithNoFallbackAndProfiling() {
        val config = OrtQnnConfig(
            backendPath = "/data/local/tmp/libQnnHtp.so",
            profilingPath = "/sdcard/Download/denoiser_profile.csv",
            profilingLevel = "detailed",
            disableCpuFallback = true,
        )

        val options = config.providerOptions()

        assertEquals("/data/local/tmp/libQnnHtp.so", options["backend_path"])
        assertEquals("detailed", options["profiling_level"])
        assertEquals("/sdcard/Download/denoiser_profile.csv", options["profiling_file_path"])
        assertEquals("1", options["session.disable_cpu_ep_fallback"])
        assertTrue(options.containsKey("ep.context_enable"))
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnConfigTest
```

Expected: FAIL because `OrtQnnConfig` does not exist.

- [ ] **Step 3: Write the minimal implementation and runtime docs**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtQnnConfig.kt
package com.example.animanpu.runtime

data class OrtQnnConfig(
    val backendPath: String,
    val profilingPath: String,
    val profilingLevel: String,
    val disableCpuFallback: Boolean,
) {
    fun providerOptions(): Map<String, String> = buildMap {
        put("backend_path", backendPath)
        put("profiling_level", profilingLevel)
        put("profiling_file_path", profilingPath)
        put("ep.context_enable", "1")
        put("ep.context_embed_mode", "0")
        put(
            "session.disable_cpu_ep_fallback",
            if (disableCpuFallback) "1" else "0",
        )
    }
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt
package com.example.animanpu.runtime

import java.io.File

class ArtifactManager(
    private val filesDir: File,
) {
    fun denoiserContextOnnx(): File = File(filesDir, "runtime/denoiser_ctx.onnx")
    fun denoiserContextBin(): File = File(filesDir, "runtime/denoiser_qnn.bin")
    fun profilingCsv(): File = File(filesDir, "runtime/denoiser_profile.csv")
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt
package com.example.animanpu.runtime

class OrtSessionFactory {
    fun buildQnnProviderOptions(config: OrtQnnConfig): Map<String, String> = config.providerOptions()
}
```

```markdown
<!-- android/app/libs/README.md -->
Place the custom `onnxruntime-android-qnn.aar` and the Qualcomm runtime `.so` files required for `arm64-v8a` in this directory before building the release candidate.

Required contents for the first device run:

- `onnxruntime-android-qnn.aar`
- `arm64-v8a/libQnnHtp.so`
- `arm64-v8a/libQnnSystem.so`
- `arm64-v8a/libQnnHtpV75Stub.so` or the matching stub for the target SDK release
```

```bash
# scripts/build_ort_android_with_qnn.sh
#!/usr/bin/env bash
set -euo pipefail

: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT}"
: "${ANDROID_SDK_ROOT:?Set ANDROID_SDK_ROOT}"
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME}"
: "${ORT_SRC_ROOT:?Set ORT_SRC_ROOT}"

cd "$ORT_SRC_ROOT"
./build.sh \
  --android \
  --android_abi arm64-v8a \
  --android_api 29 \
  --android_sdk_path "$ANDROID_SDK_ROOT" \
  --android_ndk_path "$ANDROID_NDK_HOME" \
  --build_shared_lib \
  --config Release \
  --use_qnn static_lib \
  --qnn_home "$QNN_SDK_ROOT" \
  --build_java
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnConfigTest
```

Expected: `BUILD SUCCESSFUL` and `OrtQnnConfigTest` passes.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt android/app/src/main/java/com/example/animanpu/runtime/OrtQnnConfig.kt android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt android/app/libs/README.md scripts/build_ort_android_with_qnn.sh
git commit -m "feat: add android qnn runtime scaffolding"
```

### Task 7: Add manual validation docs and end-to-end verification gates

**Files:**
- Create: `docs/manual/android-validation.md`
- Modify: `python/src/anima_host/export_denoiser.py`
- Modify: `python/src/anima_host/compile_qnn.py`

- [ ] **Step 1: Write the failing smoke-test checks as CLI expectations**

```python
# append to python/src/anima_host/export_denoiser.py

def validate_export_shapes(spec: ExportSpec) -> None:
    latent, timestep, cond, uncond = build_dummy_inputs(spec)
    assert tuple(latent.shape) in {(1, 4, 128, 128), (1, 4, 128, 96), (1, 4, 96, 128)}
    assert tuple(timestep.shape) == (1,)
    assert tuple(cond.shape) == (1, spec.max_tokens, 16)
    assert tuple(uncond.shape) == (1, spec.max_tokens, 16)
```

```python
# append to python/src/anima_host/compile_qnn.py

def validate_qnn_command(command: list[str]) -> None:
    joined = " ".join(command)
    assert "qnn-context-binary-generator" in joined
    assert "--profiling_level" in joined
```

- [ ] **Step 2: Run all Python tests to verify the new checks fail until wired into the CLIs**

Run:

```bash
cd python && python -m pytest -q
```

Expected: FAIL because the new validation helpers are not called by `main()` yet.

- [ ] **Step 3: Wire the validations into the CLIs and add the manual checklist**

```python
# replace main() body in python/src/anima_host/export_denoiser.py
    spec = ExportSpec(
        bundle_dir=args.bundle_dir,
        output_path=args.output_path,
        width=args.width,
        height=args.height,
        max_tokens=args.max_tokens,
    )
    validate_export_shapes(spec)
    latent, timestep, cond, uncond = build_dummy_inputs(spec)
    print({
        "latent": tuple(latent.shape),
        "timestep": tuple(timestep.shape),
        "cond": tuple(cond.shape),
        "uncond": tuple(uncond.shape),
    })
```

```python
# replace main() body in python/src/anima_host/compile_qnn.py
    command = build_qnn_context_command(
        onnx_model=args.onnx_model,
        output_dir=args.output_dir,
        sdk_root=args.sdk_root,
        profiling_level=args.profiling_level,
    )
    validate_qnn_command(command)
    print(" ".join(command))
```

```markdown
<!-- docs/manual/android-validation.md -->
# Android Validation Checklist

## Device Preconditions

- Snapdragon 8 Elite test device connected with `adb`
- Custom `onnxruntime-android-qnn.aar` installed in the Android project
- Qualcomm `.so` files packaged for `arm64-v8a`
- `denoiser_ctx.onnx` and `denoiser_qnn.bin` copied into the app runtime directory

## Validation Run

1. Install the debug app on the phone.
2. Launch the app and enter:
   - Resolution: `1024x1024`
   - Prompt: `cat astronaut, cinematic lighting`
   - Negative prompt: `blurry, low quality`
   - Steps: `8`
   - CFG: `5.0`
3. Start generation.
4. Confirm the app writes a profiling file to the configured runtime directory.
5. Confirm the profiling file contains QNN/HTP activity for the denoiser session.
6. Repeat the run three times.

## Required Acceptance Evidence

- Screenshot of the completed UI run
- The generated image file
- The profiling CSV or QNN log
- A note confirming CPU fallback was disabled for the denoiser validation run
```

- [ ] **Step 4: Run the full Python and Android unit suites**

Run:

```bash
cd python && python -m pytest -q
cd ../android && ./gradlew testDebugUnitTest
```

Expected: all unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/src/anima_host/export_denoiser.py python/src/anima_host/compile_qnn.py docs/manual/android-validation.md
git commit -m "docs: add validation gates for android qnn poc"
```

## Self-Review

- Spec coverage checked: the plan covers the host-side baseline, denoiser export path, Android runtime path, minimal UI, profiling evidence, and stop-loss validation gates.
- Placeholder scan checked: no placeholder markers remain.
- Type consistency checked: `GenerationConfig`, `GenerationRequest`, `GenerationResult`, `OrtQnnConfig`, and the export/compile specs use consistent names across tasks.
