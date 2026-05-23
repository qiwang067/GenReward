from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import cv2
from pathlib import Path

from s3dg import S3D
import vae3d
from vae3d import read_video_frames
from fb_net import FBNetworkManager
from tools import AttrDict

DEFAULT_VAE_MODEL = "/code/cogkit/models/CogVideoX-5b-I2V"
DEFAULT_VAE_LORA = "/code/tesseract/tesseract_v01e_rgb_lora"
DEFAULT_VAE_PATH = "/code/taskpipeline/vae"

DEFAULT_S3D_DICT = "s3d/s3d_dict.npy"
DEFAULT_S3D_WEIGHTS = "s3d/s3d_howto100m.pth"


@dataclass
class RewardContext:
    frame: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    env_info: Dict[str, Any] = field(default_factory=dict)
    global_step: int = 0
    local_step: int = 0


class BaseReward:
    def __init__(self, name: str, weight: float = 1.0, interval: int = 1, start_step: int = 0):
        self.name = name
        self.weight = float(weight)
        self.interval = max(1, int(interval))
        self.start_step = max(0, int(start_step))
        self._force_enabled = False
        self._step = 0

    def setup(self):
        pass

    def reset(self):
        self._step = 0

    def enable(self):
        self._force_enabled = True

    def disable(self):
        self._force_enabled = False

    def is_active(self, global_step: int) -> bool:
        return self._force_enabled or global_step >= self.start_step

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = ctx or RewardContext()
        self._step += 1
        ctx.local_step = self._step
        return ctx

    def ready(self, ctx: Optional[RewardContext] = None) -> bool:
        return (self._step % self.interval) == 0

    def compute(self, ctx: Optional[RewardContext] = None) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {}

    def train(self, data: Any, ctx: Optional[RewardContext] = None) -> Dict[str, Any]:
        return {}

    def state_dict(self) -> Optional[Dict[str, Any]]:
        return None

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    def sync_from(self, other: "BaseReward"):
        if other and other.state_dict():
            self.load_state_dict(other.state_dict())


class FrameBuffer:
    def __init__(self, keep: int = 128):
        self.keep = keep
        self.frames: List[np.ndarray] = []

    def push(self, frame: np.ndarray):
        self.frames.append(frame)
        if len(self.frames) > self.keep:
            self.frames = self.frames[-self.keep:]

    def pop_all(self) -> List[np.ndarray]:
        frames = self.frames
        self.frames = []
        return frames

    def last(self) -> Optional[np.ndarray]:
        return self.frames[-1] if self.frames else None

    def __len__(self):
        return len(self.frames)


class DiffusionReward(BaseReward):
    """
    Diffusion Reward from https://arxiv.org/abs/2312.14134

    Uses pre-trained VQ-Diffusion (discrete diffusion model) to compute reward
    based on variational lower bound (VLB) estimation.

    Two reward types:
    - 'entropy': Uses denoised samples for VLB estimation
    - 'likelihood': Uses observed samples for VLB estimation

    Pre-trained models available at: https://huggingface.co/tauhuang/diffusion_reward
    """

    def __init__(self,
                 vqgan_checkpoint_path: str,
                 vqgan_config_path: str,
                 vqdiffusion_checkpoint_path: str,
                 vqdiffusion_config_path: str,
                 stat_path: Optional[str] = None,
                 task_name: str = 'pick-place',
                 reward_type: str = 'entropy',
                 use_std: bool = True,
                 skip_step: int = 0,
                 num_sample: int = 1,
                 noise: bool = True,
                 noise_scale: float = 1e-6,
                 device: str = 'cuda:0',
                 keep: int = 256,
                 interval: int = 256,
                 weight: float = 1.0,
                 expl_scale: float = 0.0):
        """
        Args:
            vqgan_checkpoint_path: Path to VQGAN checkpoint (vqgan.pt)
            vqgan_config_path: Path to VQGAN config
            vqdiffusion_checkpoint_path: Path to VQ-Diffusion checkpoint (vqdiffusion.pt)
            vqdiffusion_config_path: Path to VQ-Diffusion config
            stat_path: Path to reward statistics YAML (for normalization)
            task_name: Task name for statistics lookup
            reward_type: 'entropy' or 'likelihood'
            use_std: If True, normalize reward using mean/std from stat_path
            skip_step: Number of diffusion steps to skip
            num_sample: Number of samples for entropy estimation
            noise: Whether to add noise during sampling
            noise_scale: Scale of noise
            device: Device to run model on
            keep: Number of frames to buffer
            interval: Compute reward every N steps
            weight: Reward weight in RewardManager
            expl_scale: Exploration reward scale (1 - expl_scale used for main reward)
        """
        super().__init__("diffusion_reward", weight=weight, interval=interval)

        self.device = device
        self.reward_type = reward_type
        self.use_std = use_std
        self.task_name = task_name
        self.skip_step = skip_step
        self.num_sample = num_sample
        self.noise = noise
        self.noise_scale = noise_scale
        self.expl_scale = expl_scale

        # Checkpoint paths
        self.vqgan_checkpoint_path = vqgan_checkpoint_path
        self.vqgan_config_path = vqgan_config_path
        self.vqdiffusion_checkpoint_path = vqdiffusion_checkpoint_path
        self.vqdiffusion_config_path = vqdiffusion_config_path
        self.stat_path = stat_path

        # Models (lazy load)
        self.model = None  # VQ-Diffusion model
        self.model_cfg = None
        self.stat = None  # (mean, std) for normalization

        # Frame buffer
        self._buf = FrameBuffer(keep=keep)

    def setup(self):
        """Load VQ-Diffusion model and statistics"""
        try:
            import sys
            diffusion_reward_path = '/code/baseline1/diffusion_reward'
            if diffusion_reward_path not in sys.path:
                sys.path.insert(0, diffusion_reward_path)

            from diffusion_reward.models.video_models.vqdiffusion.modeling.build import build_model
            from diffusion_reward.models.video_models.vqdiffusion.utils.io import load_yaml_config
            from diffusion_reward.models.video_models.vqdiffusion.utils.misc import get_model_parameters_info

            # Load config
            self.model_cfg = load_yaml_config(self.vqdiffusion_config_path)

            # Build and load model
            self.model = build_model(self.model_cfg)
            model_parameters = get_model_parameters_info(self.model)
            print(f"[DiffusionReward] Model parameters: {model_parameters}")

            # Load checkpoint
            if Path(self.vqdiffusion_checkpoint_path).exists():
                ckpt = torch.load(self.vqdiffusion_checkpoint_path, map_location=self.device)

                # Load model weights
                missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
                print(f'[DiffusionReward] Model missing keys: {missing}')
                print(f'[DiffusionReward] Model unexpected keys: {unexpected}')

                # Load EMA model if available
                if 'ema' in ckpt:
                    print("[DiffusionReward] Loading EMA model")
                    ema_model = self.model.get_ema_model()
                    missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {self.vqdiffusion_checkpoint_path}")

            self.model = self.model.to(self.device)
            self.model.eval()

            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Load normalization statistics
            if self.use_std and self.stat_path:
                import yaml
                with open(self.stat_path, 'r') as f:
                    # print("self.stat_path:", self.stat_path)
                    stats = yaml.safe_load(f)
                    # stats should be {task_name: {skip_step: [mean, std]}}
                    # Try both int and string keys for skip_step (YAML may parse as int)
                    if self.task_name in stats:
                        if self.skip_step in stats[self.task_name]:
                            self.stat = stats[self.task_name][self.skip_step]
                            print(f"[DiffusionReward] Loaded statistics for task={self.task_name}, skip_step={self.skip_step}: {self.stat}")
                        elif str(self.skip_step) in stats[self.task_name]:
                            self.stat = stats[self.task_name][str(self.skip_step)]
                            print(f"[DiffusionReward] Loaded statistics for task={self.task_name}, skip_step={self.skip_step}: {self.stat}")
                        else:
                            print(f"[DiffusionReward] Warning: No statistics found for task={self.task_name}, skip_step={self.skip_step}")
                            self.stat = [0.0, 1.0]
                    else:
                        print(f"[DiffusionReward] Warning: Task {self.task_name} not found in statistics file")
                        self.stat = [0.0, 1.0]

            print(f"[DiffusionReward] Loaded VQ-Diffusion from {self.vqdiffusion_checkpoint_path}")
            print(f"[DiffusionReward] Reward type: {self.reward_type}, use_std: {self.use_std}")
            print(f"[DiffusionReward] Skip step: {self.skip_step}, num_sample: {self.num_sample}")

        except Exception as e:
            print(f"[DiffusionReward] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def reset(self):
        super().reset()
        self._buf = FrameBuffer(keep=self._buf.keep)

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = super().on_step(ctx)
        if ctx.frame is not None:
            self._buf.push(ctx.frame)
        return ctx

    def ready(self, ctx: Optional[RewardContext] = None) -> bool:
        """Check if we have enough frames to compute reward"""
        if self.model is None:
            return False

        # Need at least (num_cond_frames + 1) * frame_skip frames
        condition_config = self.model.cfg.params.get('condition_emb_config', {})
        num_cond_frames = condition_config.get('params', {}).get('num_cond_frames', 1)
        frame_skip = self.model.frame_skip
        min_frames = (num_cond_frames + 1) * frame_skip + 1

        has_enough_frames = len(self._buf.frames) >= min_frames
        return super().ready(ctx) and has_enough_frames

    def compute(self, ctx: Optional[RewardContext] = None) -> Tuple[float, Dict[str, Any]]:
        if not self.ready(ctx):
            return 0.0, {}

        frames = self._buf.frames
        if not frames:
            return 0.0, {}

        # Ensure uint8
        frames = ensure_uint8_frames(frames)

        # Compute Diffusion reward
        reward = self._calc_diffusion_reward(frames)
        # breakpoint()

        return reward, {
            "diffusion_reward": reward,
            "diffusion_type": self.reward_type,
            "diffusion_num_frames": len(frames),
        }

    @torch.no_grad()
    def _calc_diffusion_reward(self, frames: List[np.ndarray]) -> float:
        """
        Core Diffusion Reward computation
        Ported from diffusion_reward/models/reward_models/diffusion_reward.py
        """
        # Convert to tensor: 1 × T × H × W × C
        frames_np = np.stack(frames, axis=0)
        imgs = torch.from_numpy(frames_np).float().unsqueeze(0) / 255.0  # Normalize to [0, 1]
        imgs = imgs.to(self.device)

        # Encode frames to tokens and prepare batches
        content, condition, _ = self._imgs_to_batch(imgs, reward_type=self.reward_type)
        content_token = content['content_token']
        condition_token = condition['condition_token']

        # Compute VLB
        rewards = self._calc_vlb(content_token, condition_token)
        

        # Normalize
        if self.use_std and self.stat is not None:
            rewards = (rewards - self.stat[0]) / self.stat[1]

        # Scale by (1 - expl_scale)
        scaled_rewards = (1 - self.expl_scale) * rewards

        return float(scaled_rewards.mean().item())

    @torch.no_grad()
    def _imgs_to_batch(self, x, reward_type='entropy'):
        """
        Encode video frames to token batches
        Ported from diffusion_reward.py:84-131
        """
        assert x.max() <= 1.0, "Images should be normalized to [0, 1]"

        seq_len = x.shape[1]
        num_frames = self.model.cfg.params['condition_emb_config']['params']['num_cond_frames'] + 1
        n_skip = self.model.frame_skip
        subseq_len = (num_frames - 1) * n_skip

        # Encode: B × T × H × W × C -> B × T × C × H × W
        x = x.permute(0, 1, 4, 2, 3)
        _, indices = self.model.content_codec.encode_to_z(x)
        assert indices.shape[0] == 1
        indices = indices.reshape(indices.shape[0], seq_len, -1)

        if reward_type == 'entropy':
            # Only return conditional frames
            post_idxes = list(range(seq_len - subseq_len + n_skip))
            batch_indices = [indices[:, idx:idx+subseq_len:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)

            if subseq_len - n_skip > 0:
                pre_batch_indices = [indices[:, idx].tile((1, num_frames - 1)) for idx in range(subseq_len-n_skip)]
                pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
                batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)
            cond = {'condition_token': batch_indices}

        elif reward_type == 'likelihood':
            # Return conditional frames + current frame
            post_idxes = list(range(seq_len - subseq_len))
            batch_indices = [indices[:, idx:idx+subseq_len+n_skip:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)

            if subseq_len - n_skip > 0:
                pre_batch_indices = [indices[:, idx].tile((1, num_frames)) for idx in range(subseq_len)]
                pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
                batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)
            cond = {'condition_token': batch_indices}
        else:
            raise NotImplementedError(f"Unknown reward_type: {reward_type}")

        x = x.flatten(0, 1)
        cont = {'content_token': indices[0]}
        return cont, cond, indices[0]

    @torch.no_grad()
    def _calc_vlb(self, cont_emb, cond_emb):
        """
        Calculate variational lower bound
        Ported from diffusion_reward.py:146-244
        """
        from diffusion_reward.models.video_models.vqdiffusion.modeling.transformers.diffusion_transformer import (
            index_to_log_onehot, log_categorical, log_onehot_to_index, sum_except_batch
        )

        x = cont_emb
        # breakpoint()
        b, device = x.size(0), self.device
        transformer = self.model.transformer
        cond_emb = transformer.condition_emb(cond_emb).float()

        # t=0
        start_step = transformer.num_timesteps
        x_start = x
        t = torch.full((b,), start_step-1, device=device, dtype=torch.long)
        log_x_start = index_to_log_onehot(x_start, transformer.num_classes)

        # t=T (mask state)
        zero_logits = torch.zeros((b, transformer.num_classes-1, transformer.shape), device=device)
        one_logits = torch.ones((b, 1, transformer.shape), device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        # Denoised time steps
        diffusion_list = [index for index in range(start_step-1, -1, -1-self.skip_step)]
        if diffusion_list[-1] != 0:
            diffusion_list.append(0)

        vlbs = []
        if self.reward_type == 'entropy':
            # Use denoised samples for estimation
            for _ in range(self.num_sample):
                start_step = transformer.num_timesteps
                x_start = x
                t = torch.full((b,), start_step-1, device=device, dtype=torch.long)
                log_x_start = index_to_log_onehot(x_start, transformer.num_classes)

                # t=T
                zero_logits = torch.zeros((b, transformer.num_classes-1, transformer.shape), device=device)
                one_logits = torch.ones((b, 1, transformer.shape), device=device)
                mask_logits = torch.cat((zero_logits, one_logits), dim=1)
                log_z = torch.log(mask_logits)

                model_log_probs = []
                log_zs = []
                ts = []
                vlb = []
                for diffusion_index in diffusion_list:
                    t = torch.full((b,), diffusion_index, device=device, dtype=torch.long)
                    log_x_recon = transformer.cf_predict_start(log_z, cond_emb, t)
                    log_zs.append(log_z)
                    if diffusion_index > self.skip_step:
                        model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-self.skip_step)
                        ts.append(t-self.skip_step)
                    else:
                        model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)
                        ts.append(t)

                    model_log_probs.append(model_log_prob)
                    log_z = transformer.log_sample_categorical(model_log_prob, noise=self.noise, noise_scale=self.noise_scale)

                x_start = log_onehot_to_index(log_z)
                log_x_start = index_to_log_onehot(x_start, transformer.num_classes)
                for i, model_log_prob in enumerate(model_log_probs[:-1]):
                    log_true_prob = transformer.q_posterior(log_x_start=log_x_start, log_x_t=log_zs[i], t=ts[i])
                    kl = transformer.multinomial_kl(log_true_prob, model_log_prob)
                    kl = sum_except_batch(kl).unsqueeze(1)
                    vlb.append(-kl)

                log_probs = model_log_probs[-1].permute(0, 2, 1)
                target = F.one_hot(x_start, num_classes=transformer.num_classes)
                rewards = (log_probs * target).sum(-1).sum(-1)
                rewards += torch.concat(vlb, dim=1).sum(dim=1)
                vlbs.append(rewards)

        elif self.reward_type == 'likelihood':
            # Use observed samples for estimation
            for diffusion_index in diffusion_list:
                t = torch.full((b,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon = transformer.cf_predict_start(log_z, cond_emb, t)
                if diffusion_index > self.skip_step:
                    model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-self.skip_step)
                    log_true_prob = transformer.q_posterior(log_x_start=log_x_start, log_x_t=log_z, t=t-self.skip_step)
                else:
                    model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)
                    log_true_prob = transformer.q_posterior(log_x_start=log_x_start, log_x_t=log_z, t=t)

                log_z = transformer.log_sample_categorical(model_log_prob, noise=self.noise, noise_scale=self.noise_scale)

                # -KL if t != 0 else LL
                if diffusion_index != 0:
                    kl = transformer.multinomial_kl(log_true_prob, model_log_prob)
                    kl = sum_except_batch(kl).unsqueeze(1)
                    vlbs.append(-kl)
                else:
                    decoder_ll = log_categorical(log_x_start, model_log_prob)
                    decoder_ll = sum_except_batch(decoder_ll).unsqueeze(1)
                    vlbs.append(decoder_ll)
        else:
            raise NotImplementedError(f"Unknown reward_type: {self.reward_type}")

        rewards = torch.stack(vlbs, dim=1).mean(1)
        return rewards


class RewardManager:
    def __init__(self, rewards: List[BaseReward]):
        self.rewards = rewards
        self._global_step = 0

    def setup(self):
        for r in self.rewards:
            r.setup()

    def reset(self):
        self._global_step = 0
        for r in self.rewards:
            r.reset()

    def _advance_step(self, global_step: Optional[int]) -> int:
        if global_step is None:
            self._global_step += 1
        else:
            self._global_step = global_step
        return self._global_step

    def on_step(self, frame: Optional[np.ndarray], action: Optional[np.ndarray], env_info: Dict[str, Any], global_step: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        step = self._advance_step(global_step)
        total = 0.0
        info: Dict[str, Any] = {}
        env_info = env_info or {}
        for reward in self.rewards:
            ctx = RewardContext(
                frame=frame,
                action=action,
                env_info=env_info,
                global_step=step,
                local_step=reward._step,
            )
            ctx = reward.on_step(ctx)
            if not reward.is_active(step) or not reward.ready(ctx):
                continue
            value, extras = reward.compute(ctx)
            weighted = value * reward.weight
            total += weighted
            if extras:
                info.update(extras)
            info.setdefault(f"{reward.name}_reward", value)
            info.setdefault(f"{reward.name}_reward_weighted", weighted)
        return total, info

    def train(self, batches: Any, global_step: Optional[int] = None) -> Dict[str, Any]:
        if not batches:
            return {}
        step = self._global_step if global_step is None else global_step
        results: Dict[str, Any] = {}
        for reward in self.rewards:
            payload = batches.get(reward.name) if isinstance(batches, dict) else batches
            if payload is None:
                continue
            ctx = RewardContext(global_step=step, local_step=reward._step)
            out = reward.train(payload, ctx)
            if out:
                results[reward.name] = out
        return results

    def enable(self, name: str):
        reward = self.get(name)
        if reward:
            reward.enable()

    def disable(self, name: str):
        reward = self.get(name)
        if reward:
            reward.disable()

    def get(self, name: str) -> Optional[BaseReward]:
        for reward in self.rewards:
            if reward.name == name:
                return reward
        return None

    def sync_from(self, other: "RewardManager"):
        if other is None:
            return
        for reward in self.rewards:
            peer = other.get(reward.name)
            if peer:
                reward.sync_from(peer)

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for reward in self.rewards:
            payload = reward.state_dict()
            if payload is not None:
                state[reward.name] = payload
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name, payload in (state_dict or {}).items():
            reward = self.get(name)
            if reward:
                reward.load_state_dict(payload)


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0+size, x0:x0+size]


def ensure_uint8_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for f in frames:
        if f.dtype != np.uint8:
            if f.max() <= 1.0:
                f = (f * 255).astype(np.uint8)
            else:
                f = f.astype(np.uint8)
        out.append(f)
    return out


class CalvinSuccessReward(BaseReward):
    """
     CALVIN  task_oracle =10
     env_info: start_info/current_info/task_oracle/task_name
    """
    def __init__(self, weight: float = 1.0):
        super().__init__("success", weight=weight, interval=1)
        self._last = 0.0

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = super().on_step(ctx)
        env_info = ctx.env_info or {}
        start_info = env_info.get("start_info")
        current_info = env_info.get("current_info")
        task_oracle = env_info.get("task_oracle")
        task_name = env_info.get("task_name")
        ok = False
        if start_info is not None and current_info is not None and task_oracle is not None and task_name:
            ok = len(task_oracle.get_task_info_for_set(start_info, current_info, {task_name})) > 0
        self._last = 1.0 if ok else 0.0
        return ctx

    def compute(self, ctx: Optional[RewardContext] = None):
        return self._last, {"success_reward": self._last, "success": bool(self._last)}


class VAEReward(BaseReward):
    def __init__(self, target_video_path: Optional[str],
                 device: Optional[str] = None,
                 crop: bool = True, size=(480, 480),
                 keep=128, weight=1.0, interval=128):
        super().__init__("vae", weight=weight, interval=interval)
        self._buf = FrameBuffer(keep=keep)
        self._vae = None
        self._target_video_path = target_video_path
        self._target_embedding = None
        self._crop = crop
        self._size = size
        self._device = device or "cuda:0"

    def _build_vae(self):
        return vae3d.VAETester(DEFAULT_VAE_MODEL, device=self._device,
                               lora_path=DEFAULT_VAE_LORA, vae_path=DEFAULT_VAE_PATH)

    def setup(self):
        self._vae = self._build_vae()
        if self._target_video_path:
            frames = read_video_frames(self._target_video_path)
            frames = ensure_uint8_frames(frames)
            video_tensor = self._vae.preprocess_video_for_vae(frames, self._size, 16)
            self._target_embedding = self._vae.encode_video(video_tensor)

    def reset(self):
        super().reset()
        self._buf = FrameBuffer(keep=self._buf.keep)

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = super().on_step(ctx)
        if ctx.frame is not None:
            self._buf.push(ctx.frame)
        return ctx

    def compute(self, ctx: Optional[RewardContext] = None):
        if not self.ready(ctx) or self._vae is None or self._target_embedding is None:
            return 0.0, {}
        frames = self._buf.pop_all()
        if not frames:
            return 0.0, {}
        if self._crop:
            frames = [center_crop_square(f) for f in frames]
        frames = ensure_uint8_frames(frames)
        video_tensor = self._vae.preprocess_video_for_vae(frames, self._size, 16)
        latents = self._vae.encode_video(video_tensor)
        similarity = self._vae.compute_vae_similarity(
            self._target_embedding,
            latents,
            method='cosine_flatten'
        )
        val = float(similarity)
        return val, {"vae_reward": val}


class S3DReward(BaseReward):
    def __init__(self,
                 target_video_path: Optional[str] = None,
                 target_text: Optional[str] = None,
                 crop: bool = True, keep=128,
                 weight: float = 1.0, interval=128):
        super().__init__("s3d", weight=weight, interval=interval)
        self._buf = FrameBuffer(keep=keep)
        self._s3d = None
        self._target_embedding = None
        self._target_video_path = target_video_path
        self._target_text = target_text
        self._crop = crop

    def _build_s3d(self):
        net = S3D(DEFAULT_S3D_DICT, 512)
        net.load_state_dict(torch.load(DEFAULT_S3D_WEIGHTS, map_location="cpu"))
        return net.eval()

    def _encode_video(self, frames: List[np.ndarray]) -> torch.Tensor:
        arr = np.array(frames)
        if self._crop:
            arr = np.array([center_crop_square(f) for f in arr])
        arr = arr[None]
        arr = arr.transpose(0, 4, 1, 2, 3)
        arr = arr[:, :, ::4, :, :]
        video = torch.from_numpy(arr).float()
        with torch.no_grad():
            out = self._s3d(video)
        return out['video_embedding']

    def setup(self):
        self._s3d = self._build_s3d()
        if self._target_video_path:
            frames = read_video_frames(self._target_video_path)
            frames = ensure_uint8_frames(frames)
            self._target_embedding = self._encode_video(frames)
        elif self._target_text:
            if hasattr(self._s3d, "text_module"):
                text_out = self._s3d.text_module([self._target_text])
                self._target_embedding = text_out['text_embedding']
            else:
                raise ValueError("S3D model has no text_module; provide target_video_path instead.")
        else:
            self._target_embedding = None

    def reset(self):
        super().reset()
        self._buf = FrameBuffer(keep=self._buf.keep)

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = super().on_step(ctx)
        if ctx.frame is not None:
            self._buf.push(ctx.frame)
        return ctx

    def compute(self, ctx: Optional[RewardContext] = None):
        if not self.ready(ctx) or self._s3d is None or self._target_embedding is None:
            return 0.0, {}
        frames = self._buf.pop_all()
        if not frames:
            return 0.0, {}
        frames = ensure_uint8_frames(frames)
        video_embedding = self._encode_video(frames)
        sim = torch.matmul(self._target_embedding, video_embedding.t())
        val = float(sim.detach().cpu().numpy()[0][0])
        return val, {"s3d_reward": val}


class IV2Reward(BaseReward):
    """
     IV2  embedding 
     IV2.read_config_from_file / IV2.build_from_config / video2feature
    """
    def __init__(self, target_video_path: str,
                 interval: int = 128, keep: int = 128, weight: float = 1.0):
        super().__init__("iv2", weight=weight, interval=interval)
        self._buf = FrameBuffer(keep=keep)
        self._target_video_path = target_video_path
        self._net = None
        self._tokenizer = None
        self._target_embedding = None

    def _build_iv2(self):
        from IV2 import read_config_from_file, build_from_config
        cfg = read_config_from_file()
        net, tok = build_from_config(cfg)
        return net, tok

    def _video2feature(self, frames: List[np.ndarray]) -> torch.Tensor:
        frames = ensure_uint8_frames(frames)
        from IV2 import video2feature
        feat = video2feature(frames, self._net, config={})
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        return feat

    def setup(self):
        self._net, self._tokenizer = self._build_iv2()
        frames = read_video_frames(self._target_video_path)
        self._target_embedding = self._video2feature(frames)

    def reset(self):
        super().reset()
        self._buf = FrameBuffer(keep=self._buf.keep)

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = super().on_step(ctx)
        if ctx.frame is not None:
            self._buf.push(ctx.frame)
        return ctx

    def compute(self, ctx: Optional[RewardContext] = None):
        if not self.ready(ctx) or self._net is None or self._target_embedding is None:
            return 0.0, {}
        frames = self._buf.pop_all()
        if not frames:
            return 0.0, {}
        cur_emb = self._video2feature(frames)
        sim = torch.matmul(self._target_embedding, cur_emb.t())
        val = float(sim.detach().cpu().numpy()[0][0])
        return val, {"iv2_reward": val}


class FBReward(BaseReward):
    """
     FBNetworkManager 
    interval=1 >1 
    
    """
    def __init__(self,
                 target_video_path: str,
                 fb_config: Optional[Dict[str, Any]],
                 device: Optional[str] = None,
                 interval: int = 1,
                 keep: int = 1,
                 weight: float = 1.0):
        cfg = AttrDict(fb_config or {})
        start_step = int(cfg.get("fb_reward_start", cfg.get("fb_train_until", 0) or 0))
        super().__init__("fb", weight=weight, interval=interval, start_step=start_step)
        self._cfg = cfg
        self._device = device or cfg.get("device", "cuda:0")
        self._keep = keep
        self._buf = FrameBuffer(keep=keep)
        self._target_video_path = target_video_path
        self._target_frame_index = int(cfg.get("target_frame_index", -1))
        self._max_train_step = cfg.get("fb_train_until", None)
        self._manager: Optional[FBNetworkManager] = None
        self._target_frame: Optional[np.ndarray] = None
        self._target_tensor: Optional[torch.Tensor] = None
        self._last_action: Optional[np.ndarray] = None

    def _build_fb(self) -> FBNetworkManager:
        cfg = deepcopy(self._cfg)
        cfg["device"] = self._device
        return FBNetworkManager(AttrDict(cfg))

    def setup(self):
        super().setup()
        self._manager = self._build_fb()
        ckpt_path = self._cfg.get("fb_ckpt")
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            self._manager.load_state_dict(state)
        if self._target_video_path:
            frames = read_video_frames(self._target_video_path)
            if not frames:
                raise ValueError(f"Target video `{self._target_video_path}` contains no frames.")
            idx = self._target_frame_index
            if idx < 0:
                idx = len(frames) + idx
            idx = int(np.clip(idx, 0, len(frames) - 1))
            self._target_frame = ensure_uint8_frames([frames[idx]])[0]
            self._target_tensor = self._to_chw_tensor(self._target_frame).unsqueeze(0)

    def reset(self):
        super().reset()
        self._buf = FrameBuffer(keep=self._keep)
        self._last_action = None

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = super().on_step(ctx)
        if ctx.frame is not None:
            self._buf.push(ctx.frame)
        if ctx.action is not None:
            self._last_action = np.array(ctx.action, copy=True)
        return ctx

    def ready(self, ctx: Optional[RewardContext] = None) -> bool:
        if ctx is not None and not self.is_active(ctx.global_step):
            return False
        return super().ready(ctx)

    def compute(self, ctx: Optional[RewardContext] = None) -> Tuple[float, Dict[str, Any]]:
        active = True
        global_step = None
        if ctx is not None:
            global_step = ctx.global_step
            active = self.is_active(global_step)
        frame = self._buf.last()
        if not active or self._manager is None or self._target_tensor is None or frame is None or self._last_action is None:
            return 0.0, {"fb_reward": 0.0, "fb_reward_active": False}
        obs = self._to_chw_tensor(frame).unsqueeze(0)
        target = self._target_tensor
        action = torch.as_tensor(self._last_action, dtype=torch.float32).unsqueeze(0)
        reward = self._manager.compute_reward_from_img_pair(obs, action, target)
        if hasattr(reward, "item"):
            reward_val = float(reward.item())
        elif isinstance(reward, (np.ndarray, list, tuple)):
            reward_val = float(np.array(reward).flatten()[0])
        else:
            reward_val = float(reward)
        return reward_val, {"fb_reward": reward_val, "fb_reward_active": True, "fb_reward_step": global_step}

    def train(self, data: Any, ctx: Optional[RewardContext] = None) -> Dict[str, Any]:
        if self._manager is None or data is None:
            return {}
        global_step = ctx.global_step if ctx else None
        if self._max_train_step is not None and global_step is not None and global_step >= self._max_train_step:
            return {}
        try:
            batch = next(data)
        except StopIteration:
            return {}
        obs = batch["image"]
        batch_size = obs.shape[0]

        if self._target_frame is not None:
            goal_image = np.repeat(self._target_frame[np.newaxis, :, :, :], batch_size, axis=0)
        else:
            goal_image = obs[:, 1]

        payload = {
            "obs": obs[:, 0],
            "action": batch["action"][:, 0],
            "next_action": batch["action"][:, 1],
            "next_obs": obs[:, 1],
            "goal_image": goal_image,
            "discount": batch["discount"][:, 0],
        }
        return self._manager.update(payload)

    def state_dict(self) -> Optional[Dict[str, Any]]:
        if self._manager is None:
            return None
        return {
            "fb_manager": self._manager.state_dict(),
            "target_frame_index": self._target_frame_index,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if not state_dict:
            return
        if self._manager and "fb_manager" in state_dict:
            self._manager.load_state_dict(state_dict["fb_manager"])
        if "target_frame_index" in state_dict:
            self._target_frame_index = int(state_dict["target_frame_index"])

    def sync_from(self, other: "FBReward"):
        if not other or other._manager is None or self._manager is None:
            return
        self._manager.sync_from(other._manager)

    def _to_chw_tensor(self, img: np.ndarray) -> torch.Tensor:
        if img.dtype != np.uint8:
            img = ensure_uint8_frames([img])[0]
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return tensor


def build_reward_manager_for_calvin(specs: Optional[List[Dict[str, Any]]],
                                    device: Optional[str] = None) -> RewardManager:
    """
    specs 
      - {"type": "success", "weight": 1.0}
      - {"type": "vae", "target_video_path": "...", "weight": 0.3, "interval": 128}
      - {"type": "s3d", "target_video_path": "..."}  {"type": "s3d", "target_text": "..." }
      - {"type": "iv2", "target_video_path": "...", "interval": 128}
      - {"type": "fb",  "target_video_path": "...", "fb_config": "...", "interval": 1}
     success
    """
    rewards: List[BaseReward] = []
    specs = specs or []
    has_success = any(s.get("type") == "success" for s in specs)
    if not has_success:
        rewards.append(CalvinSuccessReward(weight=1.0))

    for s in specs:
        typ = s.get("type")
        if typ == "success":
            rewards.append(CalvinSuccessReward(weight=float(s.get("weight", 1.0))))
        elif typ == "vae":
            rewards.append(VAEReward(
                target_video_path=s.get("target_video_path"),
                device=device or s.get("device"),
                crop=s.get("crop", True),
                size=s.get("size", (480, 480)),
                keep=int(s.get("keep", 128)),
                weight=float(s.get("weight", 1.0)),
                interval=int(s.get("interval", 128)),
            ))
        elif typ == "s3d":
            rewards.append(S3DReward(
                target_video_path=s.get("target_video_path"),
                target_text=s.get("target_text"),
                crop=s.get("crop", True),
                keep=int(s.get("keep", 128)),
                weight=float(s.get("weight", 1.0)),
                interval=int(s.get("interval", 128)),
            ))
        elif typ == "iv2":
            target = s.get("target_video_path")
            if not target:
                raise ValueError("IV2Reward requires target_video_path.")
            rewards.append(IV2Reward(
                target_video_path=target,
                interval=int(s.get("interval", 128)),
                keep=int(s.get("keep", 128)),
                weight=float(s.get("weight", 1.0)),
            ))
        elif typ == "fb":
            rewards.append(FBReward(
                target_video_path=s.get("target_video_path"),
                fb_config=s.get("fb_config"),
                device=device or s.get("device"),
                interval=s.get("interval", 1),
                keep=s.get("keep", 1),
                weight=s.get("weight", 1.0),
            ))
        else:
            raise ValueError(f"Unknown reward type `{typ}` in reward spec.")
    mgr = RewardManager(rewards)
    # breakpoint()
    mgr.setup()
    return mgr


# ========= VIPER (Video Prediction Reward) =========

class VIPERReward(BaseReward):
    """
    VIPER (Video Prediction Reward) from https://arxiv.org/abs/2312.14134

    Uses pre-trained VideoGPT (autoregressive transformer) to compute reward
    based on video prediction likelihood or entropy.

    Two reward types:
    - 'likelihood': log p(x_t | x_{<t}) - measures similarity to expert demonstrations
    - 'entropy': -H(p(x_t | x_{<t})) - measures prediction certainty

    Pre-trained models available at: https://huggingface.co/tauhuang/diffusion_reward
    """

    def __init__(self,
                 viper_checkpoint_path: str,
                 viper_config_path: str,
                 stat_path: Optional[str] = None,
                 reward_type: str = 'likelihood',
                 compute_joint: bool = False,
                 use_std: bool = True,
                 device: str = 'cuda:0',
                 keep: int = 256,
                 interval: int = 256,
                 weight: float = 1.0):
        """
        Args:
            viper_checkpoint_path: Path to VideoGPT checkpoint (videogpt.pt)
            viper_config_path: Path to VideoGPT config (.hydra/config.yaml)
            stat_path: Path to reward statistics YAML (for normalization)
            reward_type: 'likelihood' or 'entropy'
            compute_joint: If True, compute joint log-likelihood over all frames
            use_std: If True, normalize reward using mean/std from stat_path
            device: Device to run model on
            keep: Number of frames to buffer
            interval: Compute reward every N steps
            weight: Reward weight in RewardManager
        """
        super().__init__("viper", weight=weight, interval=interval)

        self.device = device
        self.reward_type = reward_type
        self.compute_joint = compute_joint
        self.use_std = use_std
        self.checkpoint_path = viper_checkpoint_path
        self.config_path = viper_config_path
        self.stat_path = stat_path

        # VideoGPT model (lazy load)
        self.model = None
        self.model_cfg = None
        self.stat = None  # (mean, std) for normalization

        # Frame buffer
        self._buf = FrameBuffer(keep=keep)

    def setup(self):
        """Load VideoGPT model and statistics"""
        # Import dependencies (lazy import to avoid conflicts)
        try:
            import sys
            diffusion_reward_path = '/nfs/kun2/users/mianw/CVPR/baseline/diffusion_reward'
            if diffusion_reward_path not in sys.path:
                sys.path.insert(0, diffusion_reward_path)

            from omegaconf import OmegaConf
            from diffusion_reward.models.video_models.videogpt.transformer import VideoGPTTransformer

            # Load config
            self.model_cfg = OmegaConf.load(self.config_path)

            # Load model
            self.model = VideoGPTTransformer(self.model_cfg).to(self.device)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()

            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Load normalization statistics
            if self.use_std and self.stat_path:
                import yaml
                with open(self.stat_path, 'r') as f:
                    stats = yaml.safe_load(f)
                    # stats should be {task_name: [mean, std]}
                    # Use first entry if available
                    if stats:
                        first_key = list(stats.keys())[0]
                        self.stat = stats[first_key]
                    else:
                        self.stat = [0.0, 1.0]

            print(f"[VIPERReward] Loaded VideoGPT from {self.checkpoint_path}")
            print(f"[VIPERReward] Reward type: {self.reward_type}, use_std: {self.use_std}")

        except Exception as e:
            print(f"[VIPERReward] Failed to load model: {e}")
            print("[VIPERReward] Make sure diffusion_reward is installed and models are downloaded")
            raise

    def reset(self):
        super().reset()
        self._buf = FrameBuffer(keep=self._buf.keep)

    def on_step(self, ctx: Optional[RewardContext] = None) -> RewardContext:
        ctx = super().on_step(ctx)
        if ctx.frame is not None:
            self._buf.push(ctx.frame)
        return ctx

    def ready(self, ctx: Optional[RewardContext] = None) -> bool:
        """Check if we have enough frames to compute reward"""
        if self.model is None:
            return False

        # Need at least (num_frames * frame_skip + 1) frames
        min_frames = self.model_cfg.num_frames * self.model_cfg.frame_skip + 1
        has_enough_frames = len(self._buf) >= min_frames

        return super().ready(ctx) and has_enough_frames

    def compute(self, ctx: Optional[RewardContext] = None) -> Tuple[float, Dict[str, Any]]:
        if not self.ready(ctx):
            return 0.0, {}

        frames = self._buf.frames
        if not frames:
            return 0.0, {}

        # Ensure uint8
        frames = ensure_uint8_frames(frames)

        # Compute VIPER reward
        reward = self._calc_viper_reward(frames)

        return reward, {
            "viper_reward": reward,
            "viper_type": self.reward_type,
            "viper_num_frames": len(frames),
        }

    @torch.no_grad()
    def _calc_viper_reward(self, frames: List[np.ndarray]) -> float:
        """
        Core VIPER reward computation
        Ported from diffusion_reward/models/reward_models/viper.py
        """
        # Convert to tensor: 1 × T × H × W × C
        frames_np = np.stack(frames, axis=0)
        imgs = torch.from_numpy(frames_np).float().unsqueeze(0)
        imgs = imgs.to(self.device)

        # Encode to tokens
        batch_embs, batch_indices = self._imgs_to_batch(imgs, self.reward_type)

        # Compute SOS (start-of-sequence) tokens
        sos_tokens = self.model.calc_sos_tokens(imgs, batch_embs)
        sos_tokens = sos_tokens.tile((batch_embs.shape[0], 1, 1))

        # Compute log probability
        rewards = self._cal_log_prob(
            batch_embs, batch_indices, sos_tokens,
            target_indices=batch_indices,
            reward_type=self.reward_type
        )

        # Normalize
        if self.use_std and self.stat is not None:
            rewards = (rewards - self.stat[0]) / self.stat[1]

        # Return mean reward
        return float(rewards.mean().item())

    @torch.no_grad()
    def _imgs_to_batch(self, x, reward_type='likelihood'):
        """
        Encode video frames to token batches
        Ported from viper.py:44-98
        """
        seq_len = x.shape[1]
        num_frames = self.model_cfg.num_frames + 1
        n_skip = self.model_cfg.frame_skip
        subseq_len = self.model_cfg.num_frames * n_skip

        # Encode with VQGAN: B × T × H × W × C -> B × T × spatial_tokens × emb_dim
        x = x.permute(0, 1, 4, 2, 3)
        embs, indices = self.model.encode_to_z(x)
        indices = indices.reshape(indices.shape[0], seq_len, -1)
        embs = embs.reshape(embs.shape[0], seq_len, indices.shape[-1], -1)

        if reward_type == 'likelihood':
            # Build batches for likelihood computation
            post_idxes = list(range(seq_len - subseq_len))
            batch_indices = [
                indices[:, idx:idx+subseq_len+n_skip:n_skip]
                for idx in post_idxes
            ]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)

            batch_embs = [
                embs[:, idx:idx+subseq_len+n_skip:n_skip]
                for idx in post_idxes
            ]
            batch_embs = torch.stack(batch_embs, dim=0)
            batch_embs = batch_embs.squeeze(1).reshape(
                batch_embs.shape[0], -1, batch_embs.shape[-1]
            )

            # Prepend context frames
            pre_batch_indices = [
                indices[:, idx].tile((1, num_frames))
                for idx in range(subseq_len)
            ]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)

            pre_batch_embs = [
                embs[:, idx].tile((1, num_frames, 1))
                for idx in range(subseq_len)
            ]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)

        elif reward_type == 'entropy':
            # Build batches for entropy computation
            post_idxes = list(range(seq_len - subseq_len + n_skip))
            batch_indices = [
                indices[:, idx:idx+subseq_len:n_skip]
                for idx in post_idxes
            ]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)

            batch_embs = [
                embs[:, idx:idx+subseq_len:n_skip]
                for idx in post_idxes
            ]
            batch_embs = torch.stack(batch_embs, dim=0)
            batch_embs = batch_embs.squeeze(1).reshape(
                batch_embs.shape[0], -1, batch_embs.shape[-1]
            )

            # Prepend context frames
            pre_batch_indices = [
                indices[:, idx].tile((1, num_frames - 1))
                for idx in range(subseq_len - n_skip)
            ]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)

            pre_batch_embs = [
                embs[:, idx].tile((1, num_frames - 1, 1))
                for idx in range(subseq_len - n_skip)
            ]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

        return batch_embs, batch_indices

    @torch.no_grad()
    def _cal_log_prob(self, embs, x, c, target_indices=None, reward_type='likelihood'):
        """
        Compute log probability using VideoGPT transformer
        Ported from viper.py:109-136
        """
        import torch.nn.functional as F

        self.model.eval()

        # Concatenate context and sequence
        if not self.model.use_vqemb:
            x = torch.cat((c, x), dim=1) if x is not None else c
        else:
            x = torch.cat((c, embs), dim=1) if x is not None else c

        # Forward through transformer
        logits, _ = self.model.transformer(x[:, :-1])
        probs = F.log_softmax(logits, dim=-1)

        if reward_type == 'likelihood':
            # Compute log p(x_t | x_{<t})
            target = F.one_hot(
                target_indices,
                num_classes=self.model_cfg.codec.num_codebook_vectors
            )

            if self.compute_joint:
                # Joint log-likelihood over all frames
                rewards = (probs * target).sum(-1).sum(-1, keepdim=True)
            else:
                # Single-step log-likelihood
                num_valid_logits = int(logits.shape[1] // (self.model_cfg.num_frames + 1))
                rewards = (probs * target).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)

        elif reward_type == 'entropy':
            # Compute negative entropy: -H(p) = -Σ p log p
            num_valid_logits = int(logits.shape[1] // self.model_cfg.num_frames)
            entropy = (- probs * probs.exp()).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)
            rewards = - entropy
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

        return rewards