from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time
from networks import ForwardMap, BackwardMap
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
prefix_path = '/code/models/models--facebook--dinov3-vits16plus-pretrain-lvd1689m'
suffix = './snapshots/c93d816fc9e567563bc068f01475bec89cc634a6/'

class FBNetworkManager:
    
    def __init__(self, config, dreamer_encoder=None, dreamer_preprocess=None):
        self.config = config
        self.device = getattr(config, 'device', 'cuda')

        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.z_dim = config.z_dim
        self.goal_dim = getattr(config, 'goal_dim', config.obs_dim)
        self.hidden_dim = config.hidden_dim
        self.feature_dim = config.feature_dim

        self.fb_lr = getattr(config, 'fb_lr', 1e-4)
        self.ortho_coef = getattr(config, 'ortho_coef', 1.0)
        self.target_tau = getattr(config, 'target_tau', 0.01)

        # Image Encoders
        self.image_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path= os.path.join(prefix_path, suffix),
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.image_processor = AutoImageProcessor.from_pretrained(os.path.join(prefix_path, suffix))

        self.dreamer_encoder = dreamer_encoder
        self.dreamer_preprocess = dreamer_preprocess

        if dreamer_encoder is not None:
            self.dreamer_projection = None
        else:
            self.dreamer_projection = None

        self.cached_goal_feature = None

        self._init_networks()

        fb_params = list(self.forward_net.parameters()) + list(self.backward_net.parameters())
        self.fb_optimizer = optim.Adam(fb_params, lr=self.fb_lr)

    def encode_images_with_dino(self, images):
        images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)) for img in images]
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        patch_size = self.image_model.config.patch_size
        batch_size, _, img_height, img_width = inputs.pixel_values.shape
        num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
        num_patches_flat = num_patches_height * num_patches_width

        with torch.inference_mode():
            outputs = self.image_model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        assert last_hidden_states.shape == (batch_size, 1 + self.image_model.config.num_register_tokens + num_patches_flat, self.image_model.config.hidden_size)

        cls_token = last_hidden_states[:, 0, :]
        return cls_token.float()

    def setup_projection(self, dreamer_output_dim):
        if self.dreamer_projection is None:
            # print(f"[FB] Creating projection layer: {dreamer_output_dim} -> {self.obs_dim}")
            self.dreamer_projection = nn.Linear(dreamer_output_dim, self.obs_dim).to(self.device)

            fb_params = (
                list(self.forward_net.parameters()) +
                list(self.backward_net.parameters()) +
                list(self.dreamer_projection.parameters())
            )
            self.fb_optimizer = optim.Adam(fb_params, lr=self.fb_lr)

    def encode_obs_with_dreamer(self, obs_tensor):
        # timings = {}
        # t_start = time.time()
        
        # breakpoint()

        if self.dreamer_encoder is None:
            print("[FB] Warning: dreamer_encoder not provided, falling back to DinoV3 (slow!)")
            return self.encode_images_with_dino(obs_tensor.permute(0, 3, 1, 2))

        with torch.no_grad():
            # t0 = time.time()
            batch_size = obs_tensor.shape[0]
            obs_dict = {
                'image': obs_tensor,  # (B, H, W, C)
                'is_first': torch.zeros(batch_size, dtype=torch.bool, device=self.device),
                'is_terminal': torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            }
            # timings['wrap_dict'] = time.time() - t0

            if self.dreamer_preprocess is not None:
                # t1 = time.time()
                obs_preprocessed = self.dreamer_preprocess(obs_dict)
                # timings['preprocess'] = time.time() - t1
            else:
                print("[FB] Warning: dreamer_preprocess not provided, using manual normalization!")
                obs_preprocessed = {'image': obs_tensor / 255.0}

            # t2 = time.time()
            features = self.dreamer_encoder(obs_preprocessed)  # (B, dreamer_dim)
            # timings['encoder'] = time.time() - t2

        # t3 = time.time()
        if self.dreamer_projection is not None:
            features = self.dreamer_projection(features)  # (B, obs_dim=384)
        else:
            self.setup_projection(features.shape[-1])
            features = self.dreamer_projection(features)
        # timings['projection'] = time.time() - t3

        # timings['total_encode'] = time.time() - t_start
        # print('totaol_encode_time（ms）:', timings['total_encode']*1000)
        return features

    def set_goal_feature(self, goal_image_tensor):
        if goal_image_tensor.dim() == 3:
            goal_image_tensor = goal_image_tensor.unsqueeze(0)

        # print(f"[FB] Pre-encoding goal frame with DinoV3 (shape={goal_image_tensor.shape})...")
        goal_chw = goal_image_tensor.permute(0, 3, 1, 2)
        with torch.no_grad():
            self.cached_goal_feature = self.encode_images_with_dino(goal_chw).squeeze(0)
        # print(f"[FB] Goal feature cached! (shape={self.cached_goal_feature.shape})")

    def encode_images(self, images):
        return self.encode_images_with_dino(images)

    def compute_reward_from_img_pair(self, obs, action, target_obs):
        obs_features = self.encode_images(obs)
        target_features = self.encode_images(target_obs)
        reward = self.compute_reward(obs_features, action, target_features)
        reward = float(reward.item())
        return reward

    def _init_networks(self):
        self.forward_net = ForwardMap(
            self.obs_dim, self.z_dim, self.action_dim,
            self.feature_dim, self.hidden_dim,
            preprocess=getattr(self.config, 'preprocess', False),
            add_trunk=getattr(self.config, 'add_trunk', True)
        ).to(self.device)
        
        self.backward_net = BackwardMap(
            self.goal_dim, self.z_dim, self.hidden_dim,
            norm_z=getattr(self.config, 'norm_z', True)
        ).to(self.device)
        
        self.forward_target_net = ForwardMap(
            self.obs_dim, self.z_dim, self.action_dim,
            self.feature_dim, self.hidden_dim,
            preprocess=getattr(self.config, 'preprocess', False),
            add_trunk=getattr(self.config, 'add_trunk', True)
        ).to(self.device)
        
        self.backward_target_net = BackwardMap(
            self.goal_dim, self.z_dim, self.hidden_dim,
            norm_z=getattr(self.config, 'norm_z', True)
        ).to(self.device)
        
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        
    def _to_tensor(self, x):
        """Convert input to tensor if it's a numpy array"""
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        return x

    def sample_z(self, batch_size):
        gaussian_rdv = torch.randn((batch_size, self.z_dim),
                                   dtype=torch.float32, device=self.device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)

        norm_z = getattr(self.config, 'norm_z', True)
        if norm_z:
            z = math.sqrt(self.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand((batch_size, self.z_dim),
                                    dtype=torch.float32, device=self.device)
            z = math.sqrt(self.z_dim) * uniform_rdv * gaussian_rdv

        return z

    def update(self, batch_data):
        # t_update_start = time.time()
        # timings = {}

        # t0 = time.time()
        obs = self._to_tensor(batch_data['obs'])
        action = self._to_tensor(batch_data['action'])
        next_obs = self._to_tensor(batch_data['next_obs'])
        next_action = self._to_tensor(batch_data['next_action'])
        # timings['data_transfer'] = time.time() - t0

        if 'discount' in batch_data:
            discount = self._to_tensor(batch_data['discount'])
        else:
            discount = torch.ones(obs.size(0), device=self.device)

        if discount.dim() == 0:
            discount = discount.unsqueeze(0).expand(obs.size(0))
        elif discount.dim() == 2:  # (batch_size, 1)
            discount = discount.squeeze(1)

        # t1 = time.time()
        z = self.sample_z(obs.size(0))
        # timings['sample_z'] = time.time() - t1

        # print("[FB] Encoding obs/next_obs with Dreamer encoder (fast)...")
        # t2 = time.time()
        obs_features = self.encode_obs_with_dreamer(obs)  # (B, H, W, C) → (B, feature_dim)
        next_obs_features = self.encode_obs_with_dreamer(next_obs)
        # timings['encode_obs'] = time.time() - t2
        # timings['obs_encode_details'] = obs_timings
        # timings['next_obs_encode_details'] = next_obs_timings

        # t3 = time.time()
        if self.cached_goal_feature is not None:
            # print("[FB] Using cached goal feature (instant!)")
            batch_size = obs.size(0)
            goal_image_features = self.cached_goal_feature.unsqueeze(0).expand(batch_size, -1)
        else:
            # print("[FB] Warning: No cached goal feature, encoding goal_image with DinoV3 (slow!)")
            goal_image = self._to_tensor(batch_data['goal_image'])
            goal_image_features = self.encode_images_with_dino(goal_image.permute(0, 3, 1, 2))
        # timings['get_goal_feature'] = time.time() - t3

        obs = obs_features
        next_obs = next_obs_features

        # t4 = time.time()
        with torch.no_grad():
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)
            target_B = self.backward_target_net(goal_image_features)
            target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)
            target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)
            target_M = torch.min(target_M1, target_M2)
        # timings['target_forward'] = time.time() - t4

        # t5 = time.time()
        F1, F2 = self.forward_net(obs, z, action)
        B = self.backward_net(goal_image_features)

        M1 = torch.einsum('sd, td -> st', F1, B)
        M2 = torch.einsum('sd, td -> st', F2, B)
        # timings['main_forward'] = time.time() - t5

        # t6 = time.time()
        I = torch.eye(M1.size(0), device=self.device)
        off_diag = ~I.bool()

        discount_expanded = discount.view(-1, 1)  # (batch_size, 1)

        fb_offdiag = 0.5 * sum((M - discount_expanded * target_M)[off_diag].pow(2).mean()
                              for M in [M1, M2])
        fb_diag = -sum(M.diag().mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        q_loss_enabled = getattr(self.config, 'q_loss', False)
        # breakpoint()
        if q_loss_enabled:
            with torch.no_grad():
                next_Q1 = torch.einsum('sd, sd -> s', target_F1, z)
                next_Q2 = torch.einsum('sd, sd -> s', target_F2, z)
                next_Q = torch.min(next_Q1, next_Q2)

                cov = torch.matmul(B.T, B) / B.shape[0]
                cov = cov + 1e-4 * torch.eye(cov.shape[0], device=self.device)
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)

                target_Q = implicit_reward.detach() + discount * next_Q

            Q1 = torch.einsum('sd, sd -> s', F1, z)
            Q2 = torch.einsum('sd, sd -> s', F2, z)
            q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

            q_loss_coef = getattr(self.config, 'q_loss_coef', 0.01)
            fb_loss = fb_loss + q_loss_coef * q_loss

        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag

        total_loss = fb_loss + self.ortho_coef * orth_loss
        # timings['loss_compute'] = time.time() - t6

        # t7 = time.time()
        self.fb_optimizer.zero_grad()
        total_loss.backward()
        self.fb_optimizer.step()
        # timings['backward_optimize'] = time.time() - t7

        # t8 = time.time()
        self._soft_update_targets()
        # timings['update_target'] = time.time() - t8

        # timings['total_update'] = time.time() - t_update_start
        # print('total_update_time:', timings['total_update']*1000)

        # print(f"time={timings['total_update']*1000:.2f}ms")
        # print(f"[FB] update complete! loss={fb_loss.item():.4f}, orth_loss={orth_loss.item():.4f}, time={timings['total_update']*1000:.2f}ms")

        metrics = {
            'loss': fb_loss.item(),
            'fb_loss': fb_loss.item(),
            'fb_offdiag': fb_offdiag.item(),
            'fb_diag': fb_diag.item(),
            'orth_loss': orth_loss.item(),
            'orth_loss_diag': orth_loss_diag.item(),
            'orth_loss_offdiag': orth_loss_offdiag.item(),
            'total_fb_loss': total_loss.item(),
            'B_norm': torch.norm(B, dim=-1).mean().item(),
            'z_norm': torch.norm(z, dim=-1).mean().item(),
            'target_M': target_M.mean().item(),
            'M1': M1.mean().item(),
            # 'time_total_ms': timings['total_update'] * 1000,
            # 'time_data_transfer_ms': timings['data_transfer'] * 1000,
            # 'time_sample_z_ms': timings['sample_z'] * 1000,
            # 'time_encode_obs_ms': timings['encode_obs'] * 1000,
            # 'time_get_goal_ms': timings['get_goal_feature'] * 1000,
            # 'time_target_forward_ms': timings['target_forward'] * 1000,
            # 'time_main_forward_ms': timings['main_forward'] * 1000,
            # 'time_loss_compute_ms': timings['loss_compute'] * 1000,
            # 'time_backward_opt_ms': timings['backward_optimize'] * 1000,
            # 'time_update_target_ms': timings['update_target'] * 1000,
        }

        if q_loss_enabled:
            metrics['q_loss'] = q_loss.item()
            metrics['Q1'] = Q1.mean().item()
            metrics['target_Q'] = target_Q.mean().item()

        return metrics
    
    def compute_reward(self, obs, action, target):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            if isinstance(action, np.ndarray):
                action = torch.FloatTensor(action)
            if isinstance(target, np.ndarray):
                target = torch.FloatTensor(target)

            #move to device
            obs = obs.to(self.device)
            action = action.to(self.device)
            target = target.to(self.device)

            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            if len(target.shape) == 1:
                target = target.unsqueeze(0)

            z = self.backward_net(target)
            norm_z = getattr(self.config, 'norm_z', True)
            if norm_z:
                z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)

            F1, F2 = self.forward_net(obs, z, action)

            Q1 = torch.einsum('sd, sd -> s', F1, z)
            Q2 = torch.einsum('sd, sd -> s', F2, z)
            reward = torch.min(Q1, Q2)

            return reward.cpu().numpy()
    
    def _soft_update_targets(self):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        soft_update(self.forward_target_net, self.forward_net, self.target_tau)
        soft_update(self.backward_target_net, self.backward_net, self.target_tau)
    
    def sync_from(self, other_manager):
        self.forward_net.load_state_dict(other_manager.forward_net.state_dict())
        self.backward_net.load_state_dict(other_manager.backward_net.state_dict())
        self.forward_target_net.load_state_dict(other_manager.forward_target_net.state_dict())
        self.backward_target_net.load_state_dict(other_manager.backward_target_net.state_dict())
        self.fb_optimizer.load_state_dict(other_manager.fb_optimizer.state_dict())
    
    def save(self, path):
        torch.save({
            'forward_net': self.forward_net.state_dict(),
            'backward_net': self.backward_net.state_dict(),
            'forward_target_net': self.forward_target_net.state_dict(),
            'backward_target_net': self.backward_target_net.state_dict(),
            'fb_optimizer': self.fb_optimizer.state_dict(),
            'config': self.config
        }, path)

    def state_dict(self):
        return {
            'forward_net': self.forward_net.state_dict(),
            'backward_net': self.backward_net.state_dict(),
            'forward_target_net': self.forward_target_net.state_dict(),
            'backward_target_net': self.backward_target_net.state_dict(),
            'fb_optimizer': self.fb_optimizer.state_dict(),
            'config': self.config
        }

    def load_state_dict(self, state_dict):
        self.forward_net.load_state_dict(state_dict['forward_net'])
        self.backward_net.load_state_dict(state_dict['backward_net'])
        self.forward_target_net.load_state_dict(state_dict['forward_target_net'])
        self.backward_target_net.load_state_dict(state_dict['backward_target_net'])
        self.fb_optimizer.load_state_dict(state_dict['fb_optimizer'])
        self.config = state_dict['config']

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.forward_net.load_state_dict(checkpoint['forward_net'])
        self.backward_net.load_state_dict(checkpoint['backward_net'])
        self.forward_target_net.load_state_dict(checkpoint['forward_target_net'])
        self.backward_target_net.load_state_dict(checkpoint['backward_target_net'])
        self.fb_optimizer.load_state_dict(checkpoint['fb_optimizer'])