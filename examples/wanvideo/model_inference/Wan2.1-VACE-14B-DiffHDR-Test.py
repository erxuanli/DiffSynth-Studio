import os
import torch
from torch import Tensor
import numpy as np
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import OpenEXR



def srgb_to_linear(x: Tensor) -> Tensor:
    """
    IEC 61966-2-1 exact piecewise sRGB → linear conversion.

    Threshold: 0.04045  (NOT the approximate 0.03928)
    Exponent : 2.4      (NOT the approximate 2.2)

    Input is clamped to [0, 1] before conversion.
    """
    x = x.clamp(0.0, 1.0)
    # Compute both branches; select by threshold
    linear_low = x / 12.92
    linear_high = ((x + 0.055) / 1.055) ** 2.4
    return torch.where(x <= 0.04045, linear_low, linear_high)


def linear_to_srgb(x: Tensor) -> Tensor:
    """
    IEC 61966-2-1 exact piecewise linear → sRGB conversion.

    Threshold: 0.0031308
    Exponent : 1/2.4

    Input is clamped to [0, 1] before conversion.
    """
    x = x.clamp(0.0, 1.0)
    srgb_low = 12.92 * x
    srgb_high = 1.055 * (x ** (1.0 / 2.4)) - 0.055
    return torch.where(x <= 0.0031308, srgb_low, srgb_high)


class LogGammaMapping:
    """
    Log-Gamma color mapping between HDR linear radiance and a [0, 1]-bounded
    representation compatible with the pretrained LDR-trained VAE.

    The same instance must be shared between training and inference, with
    identical gamma and M loaded from the checkpoint config.

    Parameters
    ----------
    gamma : float
        Compression strength. Default 2.2 — NOT specified by the paper.
        Physically motivated (matches standard perceptual gamma encoding).
        MUST be saved with the LoRA checkpoint and restored at inference.
    M : float
        Maximum representable radiance. Default 1000.0 — NOT specified by the
        paper. Approximately 10 stops above SDR white (100 nits → 100,000 nits
        peak HDR). MUST be saved with the LoRA checkpoint and restored at inference.
    """

    def __init__(self, gamma: float = 2.2, M: float = 1000.0):
        self.gamma = float(gamma)
        self.M = float(M)

    # ------------------------------------------------------------------
    # Forward: HDR linear → Log-Gamma space
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Map HDR linear radiance to Log-Gamma space.

        T(x) = ( log(1 + γ·x) / log(1 + γ·M) )^(1/γ)

        - x is clamped to [0, M] before the log.
        - All arithmetic is performed in float32 regardless of input dtype.
        - Output is bounded in [0, 1] for x ∈ [0, M].
        """
        orig_dtype = x.dtype
        x = x.float()

        gamma = self.gamma
        M = self.M

        x_clamped = x.clamp(0.0, M)
        # log(1 + γ·M) is a constant — compute once
        log_denom = float(torch.log1p(torch.tensor(gamma * M)).item())

        numerator = torch.log1p(gamma * x_clamped)
        y = (numerator / log_denom) ** (1.0 / gamma)

        return y.to(orig_dtype)

    # ------------------------------------------------------------------
    # Inverse: Log-Gamma space → HDR linear
    # ------------------------------------------------------------------

    def inverse(self, y: Tensor) -> Tensor:
        """
        Map Log-Gamma space back to HDR linear radiance.

        T⁻¹(y) = ( exp( y^γ · log(1 + γ·M) ) - 1 ) / γ

        - y is clamped to [0, 1] before y^γ to prevent domain errors.
        - All arithmetic is performed in float32 regardless of input dtype.
        """
        orig_dtype = y.dtype
        y = y.float()

        gamma = self.gamma
        M = self.M

        y_clamped = y.clamp(0.0, 1.0)
        log_denom = float(torch.log1p(torch.tensor(gamma * M)).item())

        x = (torch.exp(y_clamped ** gamma * log_denom) - 1.0) / gamma

        return x.to(orig_dtype)

    def __call__(self, x: Tensor) -> Tensor:
        """Alias for forward()."""
        return self.forward(x)



def save_video_srgb_exposed(frames, save_path, fps, quality=5, ev_stops=(-4, -2, 0, 2, 4)):
    """Save one LDR sRGB video per EV stop from sRGB PIL frames."""
    base, ext = os.path.splitext(save_path)
    for ev in ev_stops:
        adjusted = []
        for frame in frames:
            t = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
            t = srgb_to_linear(t)
            t = (t * (2.0 ** ev)).clamp(0.0, 1.0)
            t = linear_to_srgb(t)
            adjusted.append(Image.fromarray((t.numpy() * 255).round().astype(np.uint8)))
        label = f"_ev{ev:+.0f}"
        save_video(adjusted, f"{base}{label}{ext}", fps=fps, quality=quality)


def save_video_linear_exposed(linear_frames, save_path, fps, quality=5, ev_stops=(-4, -2, 0, 2, 4)):
    """Save one LDR sRGB video per EV stop from linear HDR tensors.

    linear_frames: list of float32 tensors [H, W, C] in linear HDR space.
    """
    base, ext = os.path.splitext(save_path)
    for ev in ev_stops:
        adjusted = []
        for t in linear_frames:
            t_exposed = (t * (2.0 ** ev)).clamp(0.0, 1.0)
            t_srgb = linear_to_srgb(t_exposed)
            adjusted.append(Image.fromarray((t_srgb.numpy() * 255).round().astype(np.uint8)))
        label = f"_ev{ev:+.0f}"
        save_video(adjusted, f"{base}{label}{ext}", fps=fps, quality=quality)


output_dir = "/scratch/eli/disney/DiffSynth-Studio/output/03/"
source_video_path = "/scratch/eli/disney/DiffSynth-Studio/data/hdm-hdr-2014/carousel_fireworks/mp4/sRGB.mp4"
reference_image_exr_path = "/scratch/eli/disney/DiffSynth-Studio/data/hdm-hdr-2014/carousel_fireworks/exr/carousel_fireworks_02_000961.exr"
H, W = 480, 832

log_gamma = LogGammaMapping(gamma=2.2, M=1000.0)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Load a source video to inpaint
source_video = VideoData(source_video_path, height=H, width=W)
frames = [source_video[i] for i in range(len(source_video))]

# Save original source video at different exposure levels for comparison
save_video_srgb_exposed(frames, output_dir+"video_Wan2.1-VACE-14B-Original.mp4", fps=15, quality=5)

# sRGB → linear → log-gamma for pipeline input
log_gamma_frames = []
for frame in frames:
    t = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
    t = srgb_to_linear(t)
    t = log_gamma.forward(t)
    log_gamma_frames.append(Image.fromarray((t.numpy() * 255).round().astype(np.uint8)))

# Load HDR reference image (EXR, linear), encode with log-gamma, resize to match video
exr_file = OpenEXR.File(reference_image_exr_path)
exr_pixels = exr_file.parts[0].channels['RGB'].pixels.astype(np.float32)
ref_linear = torch.from_numpy(exr_pixels)
ref_log_gamma = log_gamma.forward(ref_linear)
ref_image = Image.fromarray((ref_log_gamma.numpy() * 255).round().astype(np.uint8))
ref_image = ref_image.resize((W, H), Image.LANCZOS)
ref_image.save(output_dir + "video_Wan2.1-VACE-14B-Reference.jpg")

# Build luminance-based mask: detect over- and underexposed regions in linearized frames.
# Rec.709 luminance; thresholds: τ_high = 0.95 (overexposed), τ_low = 0.05 (underexposed).
tau_high, tau_low = 0.95, 0.05
mask_frames = []
mask_arrays = []
for frame in frames:
    t = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
    t_linear = srgb_to_linear(t)
    lum = 0.2126 * t_linear[..., 0] + 0.7152 * t_linear[..., 1] + 0.0722 * t_linear[..., 2]
    mask = ((lum > tau_high) | (lum < tau_low)).numpy().astype(np.uint8) * 255
    mask_arrays.append(mask)
    mask_frames.append(Image.fromarray(mask).convert("RGB"))

# Save mask as video
save_video(mask_frames, output_dir+"video_Wan2.1-VACE-14B-Mask.mp4", fps=15, quality=5)

# Save source video with mask overlay (red tint on masked region)
overlay_frames = []
for frame, mask_arr in zip(frames, mask_arrays):
    mask_alpha = mask_arr.astype(np.float32) / 255.0
    arr = np.array(frame).astype(np.float32)
    tint = np.zeros_like(arr)
    tint[..., 0] = 255
    arr = arr * (1 - 0.4 * mask_alpha[..., None]) + tint * (0.4 * mask_alpha[..., None])
    overlay_frames.append(Image.fromarray(arr.clip(0, 255).astype(np.uint8)))
save_video(overlay_frames, output_dir+"video_Wan2.1-VACE-14B-MaskedInput.mp4", fps=15, quality=5)

# Video inpainting: feed log-gamma encoded frames to the pipeline.
# Mask convention: white (255/1) = regenerate, black (0) = preserve from input.
video = pipe(
    prompt="曝光过度：恢复灯泡和灯光，恢复它们最亮的部分和高光；曝光不足：恢复天空和星空，恢复狮子的细节, 修复画作中的细节; HDR, 非常详细, 电影灯光, 高对比度", # Overexposed: Recover the light bulbs and lights, recover their brightest points and highlights; Underexposed: Recover the sky and the stars in the sky, recover the details on the lion, restore the details in the paintings; HDR, extremely detailed, cinematic lighting, high contrast.
    negative_prompt="静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=log_gamma_frames,
    vace_video_mask=mask_frames,
    vace_reference_image=ref_image,
    seed=1, tiled=False
)

# Inverse log-gamma on pipeline output → linear HDR
linear_hdr_frames = []
for frame in video:
    t = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
    t = log_gamma.inverse(t)
    linear_hdr_frames.append(t)

save_video_linear_exposed(linear_hdr_frames, output_dir+"video_Wan2.1-VACE-14B-Inpainting.mp4", fps=15, quality=5)
