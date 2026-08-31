"""A100-oriented DnCNN fine-tuning utilities.

The model follows the official cszn/KAIR merged-BN grayscale blind DnCNN:
20 convolutions, ReLU activations, and residual noise subtraction.
"""

from __future__ import annotations

import copy
import random
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import Dataset


NOISE_RANGES: dict[str, tuple[float, float]] = {
    "gaussian": (0.0, 0.1),
    "rician": (0.0, 0.15),
    "uniform": (0.0, 0.2),
    "salt_and_pepper": (0.0, 0.2),
}


class DnCNN(nn.Module):
    """Official merged-BN DnCNN topology used by dncnn_gray_blind.pth."""

    def __init__(
        self,
        depth: int = 20,
        channels: int = 1,
        features: int = 64,
    ) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("depth must be at least 3")

        layers: list[nn.Module] = [
            nn.Conv2d(channels, features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, 3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, noisy: Tensor) -> Tensor:
        if noisy.ndim != 4:
            raise ValueError(f"expected BCHW tensor, got shape {tuple(noisy.shape)}")
        predicted_noise = self.model(noisy)
        return noisy - predicted_noise


def _random_tensor_like(image: Tensor, generator: torch.Generator) -> Tensor:
    return torch.rand(
        image.shape,
        dtype=image.dtype,
        device=image.device,
        generator=generator,
    )


def _normal_tensor_like(image: Tensor, generator: torch.Generator) -> Tensor:
    return torch.randn(
        image.shape,
        dtype=image.dtype,
        device=image.device,
        generator=generator,
    )


def add_synthetic_noise(
    clean: Tensor,
    noise_type: str,
    sigma: float,
    generator: torch.Generator,
) -> Tensor:
    """Apply one challenge noise type without clipping the observation."""
    if noise_type not in NOISE_RANGES:
        raise ValueError(f"unsupported noise type: {noise_type}")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    if noise_type == "gaussian":
        return clean + _normal_tensor_like(clean, generator) * sigma
    if noise_type == "rician":
        real = clean + _normal_tensor_like(clean, generator) * sigma
        imag = _normal_tensor_like(clean, generator) * sigma
        return torch.sqrt(real.square() + imag.square())
    if noise_type == "uniform":
        uniform = _random_tensor_like(clean, generator) * 2.0 - 1.0
        return clean + uniform * sigma

    draw = _random_tensor_like(clean, generator)
    noisy = clean.clone()
    noisy[draw < sigma / 2.0] = 0.0
    noisy[(draw >= sigma / 2.0) & (draw < sigma)] = clean.max()
    return noisy


def discover_clean_files(directories: Iterable[str | Path]) -> list[Path]:
    """Find clean npy files while hard-blocking test-named directories."""
    files: list[Path] = []
    for raw_directory in directories:
        directory = Path(raw_directory)
        if any(part.lower().startswith("test") for part in directory.parts):
            raise ValueError(f"test split cannot be used for training: {directory}")
        files.extend(sorted(directory.glob("*.npy")))
    if not files:
        raise FileNotFoundError("no .npy clean images found")
    return files


def validate_disjoint_splits(
    train_files: Sequence[str | Path],
    validation_files: Sequence[str | Path],
) -> None:
    train_keys = {str(Path(path).resolve(strict=False)) for path in train_files}
    validation_keys = {
        str(Path(path).resolve(strict=False)) for path in validation_files
    }
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"train/validation overlap detected: {len(overlap)} files")


class CleanDenoisingDataset(Dataset):
    """Clean-only dataset that synthesizes noisy inputs on demand."""

    def __init__(
        self,
        files: Sequence[str | Path],
        patch_size: int | None,
        training: bool,
        base_seed: int = 2026,
        identity_probability: float = 0.08,
    ) -> None:
        if not files:
            raise ValueError("files must not be empty")
        self.files = [Path(path) for path in files]
        self.patch_size = patch_size
        self.training = training
        self.base_seed = base_seed
        self.identity_probability = identity_probability if training else 0.0

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load(path: Path) -> Tensor:
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2:
            raise ValueError(f"expected 2D grayscale array: {path} -> {array.shape}")
        return torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0)

    def _crop_and_augment(self, clean: Tensor) -> Tensor:
        if self.patch_size is not None:
            _, height, width = clean.shape
            if self.patch_size > min(height, width):
                raise ValueError("patch_size exceeds image dimensions")
            top = int(torch.randint(0, height - self.patch_size + 1, ()).item())
            left = int(torch.randint(0, width - self.patch_size + 1, ()).item())
            clean = clean[
                :,
                top : top + self.patch_size,
                left : left + self.patch_size,
            ]
        if torch.rand(()) < 0.5:
            clean = clean.flip(-1)
        if torch.rand(()) < 0.5:
            clean = clean.flip(-2)
        if torch.rand(()) < 0.5:
            clean = clean.transpose(-1, -2)
        return clean.contiguous()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        path = self.files[index]
        clean = self._load(path)
        if self.training:
            clean = self._crop_and_augment(clean)
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
            chooser = random.Random(seed)
        else:
            seed = (zlib.crc32(path.name.encode()) + self.base_seed) & 0x7FFFFFFF
            chooser = random.Random(seed)

        generator = torch.Generator().manual_seed(seed)
        if self.training and chooser.random() < self.identity_probability:
            noisy = clean.clone()
        else:
            noise_type = chooser.choice(tuple(NOISE_RANGES))
            low, high = NOISE_RANGES[noise_type]
            sigma = chooser.uniform(low, high)
            noisy = add_synthetic_noise(clean, noise_type, sigma, generator)
        return clean, noisy, path.name


def calculate_psnr(estimate: Tensor, reference: Tensor) -> Tensor:
    """Match the provided notebook's per-image-reference-peak PSNR."""
    if estimate.shape != reference.shape or estimate.ndim != 4:
        raise ValueError("estimate and reference must be matching BCHW tensors")
    mse = (estimate - reference).square().mean(dim=(1, 2, 3))
    reference_peak = reference.amax(dim=(1, 2, 3))
    return 10.0 * torch.log10(reference_peak.square() / (mse + 1e-12))


def calculate_ssim(
    estimate: Tensor,
    reference: Tensor,
    window_size: int = 11,
) -> Tensor:
    """Match the uniform-window SSIM implementation in the test notebook."""
    if estimate.shape != reference.shape or estimate.ndim != 4:
        raise ValueError("estimate and reference must be matching BCHW tensors")
    if min(estimate.shape[-2:]) < window_size:
        raise ValueError("images are smaller than the SSIM window")

    channels = estimate.shape[1]
    window = torch.full(
        (channels, 1, window_size, window_size),
        1.0 / window_size**2,
        dtype=estimate.dtype,
        device=estimate.device,
    )
    kwargs = {"groups": channels}
    ux = functional.conv2d(estimate, window, **kwargs)
    uy = functional.conv2d(reference, window, **kwargs)
    uxx = functional.conv2d(estimate * estimate, window, **kwargs)
    uyy = functional.conv2d(reference * reference, window, **kwargs)
    uxy = functional.conv2d(estimate * reference, window, **kwargs)

    covariance_norm = window_size**2 / (window_size**2 - 1)
    vx = covariance_norm * (uxx - ux.square())
    vy = covariance_norm * (uyy - uy.square())
    vxy = covariance_norm * (uxy - ux * uy)
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
        (ux.square() + uy.square() + c1) * (vx + vy + c2)
    )
    return score.mean(dim=(1, 2, 3))


@torch.no_grad()
def x8_inference(model: nn.Module, image: Tensor) -> Tensor:
    """Dihedral x8 self-ensemble, matching KAIR's optional x8 evaluation."""
    predictions: list[Tensor] = []
    for flip in (False, True):
        for rotation in range(4):
            transformed = torch.rot90(image, rotation, dims=(-2, -1))
            if flip:
                transformed = transformed.flip(-1)
            predicted = model(transformed)
            if flip:
                predicted = predicted.flip(-1)
            predicted = torch.rot90(predicted, -rotation, dims=(-2, -1))
            predictions.append(predicted)
    return torch.stack(predictions).mean(0)
class DataKey(IntEnum):
    Label = 0
    Noisy = 1
    Name = 2


class PairedDenoisingDataset(Dataset):
    """Paired noisy/clean data reserved for final evaluation only."""

    def __init__(self, label_dir: str | Path, noisy_dir: str | Path) -> None:
        self.label_dir = Path(label_dir)
        self.noisy_dir = Path(noisy_dir)
        label_names = {path.name for path in self.label_dir.glob("*.npy")}
        noisy_names = {path.name for path in self.noisy_dir.glob("*.npy")}
        if label_names != noisy_names:
            missing_noisy = label_names - noisy_names
            missing_label = noisy_names - label_names
            raise ValueError(
                "pairing mismatch: "
                f"missing noisy={len(missing_noisy)}, missing label={len(missing_label)}"
            )
        if not label_names:
            raise FileNotFoundError("no paired test arrays found")
        self.names = sorted(label_names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        name = self.names[index]
        label = CleanDenoisingDataset._load(self.label_dir / name)
        noisy = CleanDenoisingDataset._load(self.noisy_dir / name)
        return label, noisy, name


def _unwrap_state_dict(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict", "params"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    if not checkpoint or not all(isinstance(value, Tensor) for value in checkpoint.values()):
        raise TypeError("checkpoint does not contain tensor weights")
    return {
        str(key).removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in checkpoint.items()
    }


def load_pretrained_weights(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)


class ExponentialMovingAverage:
    """Maintain a validation/inference copy with smoothed parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source_state = model.state_dict()
        for name, ema_value in self.model.state_dict().items():
            source_value = source_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(
                    source_value, alpha=1.0 - self.decay
                )
            else:
                ema_value.copy_(source_value)


@dataclass(frozen=True)
class TimeBudget:
    max_minutes: float
    reserve_minutes: float = 5.0
    start_time: float = 0.0

    @classmethod
    def start(cls, max_minutes: float, reserve_minutes: float = 5.0) -> "TimeBudget":
        return cls(max_minutes, reserve_minutes, time.monotonic())

    def expired(self, now: float | None = None) -> bool:
        if self.max_minutes <= self.reserve_minutes:
            raise ValueError("max_minutes must exceed reserve_minutes")
        current = time.monotonic() if now is None else now
        usable_seconds = (self.max_minutes - self.reserve_minutes) * 60.0
        return current - self.start_time >= usable_seconds


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_config: dict[str, int],
    epoch: int,
    val_psnr: float,
    val_ssim: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_config),
            "epoch": int(epoch),
            "val_psnr": float(val_psnr),
            "val_ssim": float(val_ssim),
        },
        path,
    )


def load_finetuned_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DnCNN, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint:
        raise ValueError("invalid fine-tuned checkpoint")
    model_config = dict(checkpoint["model_config"])
    model = DnCNN(**model_config)
    model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)
    metadata = {
        key: checkpoint[key]
        for key in ("epoch", "val_psnr", "val_ssim", "model_config")
        if key in checkpoint
    }
    return model.to(device).eval(), metadata

def composite_loss(
    restored: Tensor,
    clean: Tensor,
    ssim_weight: float = 0.02,
) -> Tensor:
    """PSNR-oriented MSE with a small SSIM term for structure retention."""
    if ssim_weight < 0:
        raise ValueError("ssim_weight must be nonnegative")
    restored_float = restored.float()
    clean_float = clean.float()
    mse = functional.mse_loss(restored_float, clean_float)
    ssim_penalty = 1.0 - calculate_ssim(restored_float, clean_float).mean()
    return mse + ssim_weight * ssim_penalty


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor, object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ssim_weight: float = 0.02,
    scaler: object | None = None,
    scheduler: object | None = None,
    ema: ExponentialMovingAverage | None = None,
    gradient_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_images = 0
    use_amp = scaler is not None and device.type == "cuda"

    for clean, noisy, _ in data_loader:
        clean = clean.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        noisy = noisy.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            restored = model(noisy)
            loss = composite_loss(restored, clean, ssim_weight=ssim_weight)

        if scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        batch_size = clean.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_images += batch_size

    if total_images == 0:
        raise ValueError("training data loader is empty")
    return total_loss / total_images


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor, object]],
    device: torch.device,
    clamp: bool = True,
) -> tuple[float, float]:
    model.eval()
    psnr_values: list[Tensor] = []
    ssim_values: list[Tensor] = []
    for clean, noisy, _ in data_loader:
        clean = clean.to(device, non_blocking=True)
        noisy = noisy.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            restored = model(noisy)
        restored = restored.float()
        if clamp:
            restored = restored.clamp(0.0, 1.0)
        clean = clean.float()
        psnr_values.append(calculate_psnr(restored, clean).cpu())
        ssim_values.append(calculate_ssim(restored, clean).cpu())
    if not psnr_values:
        raise ValueError("validation data loader is empty")
    return (
        float(torch.cat(psnr_values).mean()),
        float(torch.cat(ssim_values).mean()),
    )


class TestTimeDenoiser(nn.Module):
    """Drop-in model for test_denoising.ipynb with optional x8 and clamping."""

    def __init__(
        self,
        model: nn.Module,
        use_x8: bool = True,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.use_x8 = use_x8
        self.clamp = clamp

    def forward(self, noisy: Tensor) -> Tensor:
        restored = (
            x8_inference(self.model, noisy)
            if self.use_x8
            else self.model(noisy)
        )
        return restored.clamp(0.0, 1.0) if self.clamp else restored