from pyexpat import model
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from networks import ForwardMap, BackwardMap
from transformers import AutoImageProcessor, AutoModel
from PIL import Image, ImageSequence
from diffusers import CogVideoXImageToVideoPipeline
import os
import argparse
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKLCogVideoX 
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
prefix_path = '/code/models/models--facebook--dinov3-vits16plus-pretrain-lvd1689m'
suffix = './snapshots/c93d816fc9e567563bc068f01475bec89cc634a6/'

class FBNetworkManager:
    """Lightweight Forward-Backward network manager"""
    
    def __init__(self, config):
        """
        Initialize FB network manager

        Args:
            config: Dictionary or object containing network configuration, must include:
                - obs_dim: Observation dimension
                - action_dim: Action dimension
                - z_dim: Latent feature dimension
                - goal_dim: Goal dimension
                - hidden_dim: Hidden layer dimension
                - feature_dim: Feature dimension
                - device: Device
                - fb_lr: Learning rate (default 1e-4)
                - ortho_coef: Orthogonal loss coefficient (default 1.0)
                - target_tau: Target network update rate (default 0.01)
        """
        self.config = config
        self.device = getattr(config, 'device', 'cuda')
        # breakpoint()
        
        # Network parameters
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.z_dim = config.z_dim
        self.goal_dim = getattr(config, 'goal_dim', config.obs_dim)
        self.hidden_dim = config.hidden_dim
        self.feature_dim = config.feature_dim 
        
        # Training parameters
        self.fb_lr = getattr(config, 'fb_lr', 1e-4)
        self.ortho_coef = getattr(config, 'ortho_coef', 1.0)
        self.target_tau = getattr(config, 'target_tau', 0.01)

        #vae
        self.use_vae_encoder = getattr(config, 'use_vae_encoder', False)
        self.vae_path = getattr(config, 'vae_path', "?")

        # Recommended default: 480x480 (multiple of 8), T=1
        self.vae_resize_hw = tuple(getattr(config, 'vae_resize_hw', (480, 480)))   # (H, W)
        self.vae_T = 1                             
        self.vae_input_unit_range = bool(getattr(config, 'vae_input_unit_range', True))  # True: [0,1], False: [-1,1]

        if not self.use_vae_encoder:
            # === Original DINO model ===

            # Image Encoders
            self.image_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path= os.path.join(prefix_path, suffix),
                dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa"
            )
            self.image_processor = AutoImageProcessor.from_pretrained(os.path.join(prefix_path, suffix))
            print("[FB] Using DINOv3 encoder")
        else:
             # === Use CogVideoX VAE instead of DINO ===
            print(f"[FB] Using VAE encoder from: {self.vae_path}")
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                "/code/cogkit/models/CogVideoX-5b-I2V",
                torch_dtype=torch.float16
            ).to(self.device)

            self.vae = pipe.vae
            self.vae.eval()

            for p in self.vae.parameters():
                p.requires_grad_(False)

            self.vae_projection = torch.nn.Conv3d(
                in_channels=16,           # VAE latent channels
                out_channels=self.obs_dim, # Target feature dimension
                kernel_size=1,            # 1x1x1 convolution (pointwise operation)
                bias=False                # No bias needed
            ).to(self.device).to(torch.float16)
            
            # ✅ Create projection layer (first on CPU float32)
            self.vae_projection = torch.nn.Conv3d(
                in_channels=16,
                out_channels=self.obs_dim,
                kernel_size=1,
                bias=False
            )
            
            # ✅ Orthogonal initialization
            print("[FB] Initializing projection with orthogonal matrix...")
            with torch.no_grad():
                w = self.vae_projection.weight
                w_2d = w.squeeze(-1).squeeze(-1).squeeze(-1)  # [obs_dim, 16]
                torch.nn.init.orthogonal_(w_2d)
                self.vae_projection.weight.data = w_2d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # ✅ Move to target device and precision
            self.vae_projection = self.vae_projection.to(self.device).to(torch.float16)
            
            # Freeze projection layer
            for p in self.vae_projection.parameters():
                p.requires_grad_(False)

        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizer
        fb_params = list(self.forward_net.parameters()) + list(self.backward_net.parameters())
        self.fb_optimizer = optim.Adam(fb_params, lr=self.fb_lr)

    def encode_images(self, images):
        """
        Extract image features using pretrained image encoder

        Args:
            images: Input images (batch_size, 3, H, W)

        Returns:
            features: CLS token (batch_size, obs_dim)
        """
        ##vae
        if self.use_vae_encoder:
            # 1) Unify to Tensor [B,3,H,W], value range [0,1] (or switch to [-1,1] based on config)
            if isinstance(images, torch.Tensor):
                print("image\n")
                x = images
                if x.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
                    x = x.float() / 255.0
                if x.dim() == 3:  # [3,H,W] -> [1,3,H,W]
                    x = x.unsqueeze(0)
            else:
                # List case: convert PIL -> numpy -> tensor
                pil_list = []
                for im in images:
                    if isinstance(im, Image.Image):
                        pil_list.append(im.convert("RGB"))
                    else:
                        arr = np.asarray(im).astype(np.uint8)
                        pil_list.append(Image.fromarray(arr).convert("RGB"))
                Wt, Ht = self.vae_resize_hw[1], self.vae_resize_hw[0]
                arrs = []
                for im in pil_list:
                    im = im.resize((Wt, Ht), Image.LANCZOS)
                    arr = np.asarray(im).astype(np.float32) / 255.0  # [H,W,3] in [0,1]
                    arr = torch.from_numpy(arr).permute(2, 0, 1)     # -> [3,H,W]
                    arrs.append(arr)
                x = torch.stack(arrs, dim=0)                        # [B,3,H,W]

            # Resize to (Ht, Wt) (ensure multiple of 8; recommended=480x480 or 480x720)
            Ht, Wt = self.vae_resize_hw
            if (x.shape[-2], x.shape[-1]) != (Ht, Wt):
                x = F.interpolate(x, size=(Ht, Wt), mode="bicubic", align_corners=False)

            # Whether to switch to [-1,1]
            if not self.vae_input_unit_range:
                x = (x - 0.5) / 0.5

            x = x.to(self.device, dtype=torch.float16)              # [B,3,H,W]

            # 2) Expand temporal dimension [B,3,H,W] -> [B,3,T,H,W]
            T = max(int(self.vae_T), 1)
            x = x.unsqueeze(2).repeat(1, 1, T, 1, 1)

            # 3) VAE encoding: latent ~ [B, C_l, T/4, H/8, W/8]
            with torch.no_grad():
                enc_out = self.vae.encode(x)
                latent = enc_out.latent_dist.sample() if hasattr(enc_out, "latent_dist") else enc_out[0].sample()

            # ✅ Channel projection + global pooling
                latent = self.vae_projection(latent)  # [B, obs_dim, 1, 60, 60]
                feats = latent.mean(dim=[2, 3, 4])    # [B, obs_dim]
            
            return feats
    
        # Convert images to PIL format and extract features
        images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)) for img in images]
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        patch_size = self.image_model.config.patch_size
        batch_size, _, img_height, img_width = inputs.pixel_values.shape
        num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
        num_patches_flat = num_patches_height * num_patches_width

        with torch.inference_mode():
            outputs = self.image_model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        # print(last_hidden_states.shape)  # [batch_size, 1 + 4 + 256, 384]
        assert last_hidden_states.shape == (batch_size, 1 + self.image_model.config.num_register_tokens + num_patches_flat, self.image_model.config.hidden_size)

        cls_token = last_hidden_states[:, 0, :]
        patch_features_flat = last_hidden_states[:, 1 + self.image_model.config.num_register_tokens:, :]
        patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
        return cls_token.float()

    def compute_reward_from_img_pair(self, obs, action, target_obs):
        """
        Compute reward from image pair

        Args:
            obs: Current observation (batch_size, 3, H, W)
            action: Current action (batch_size, action_dim)
            target_obs: Target observation (batch_size, 3, H, W)

        Returns:
            reward: Computed reward
        """
        obs_features = self.encode_images(obs)
        target_features = self.encode_images(target_obs)
        reward = self.compute_reward(obs_features, action, target_features)
        reward = float(reward.item())
        return reward

    def _init_networks(self):
        """Initialize all networks"""
        # Main networks
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
        
        # Target networks
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
        
        # Initialize target network weights
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
    
    def update(self, batch_data):
        """
        Update FB network

        Args:
            batch_data: Dictionary containing:
                - obs: Current observation (batch_size, H, W, 3)
                - action: Action (batch_size, action_dim)
                - next_action: Next action (batch_size, action_dim)
                - next_obs: Next observation (batch_size, H, W, 3)
                - discount: Discount factor (batch_size,)

        Returns:
            metrics: Training metrics dictionary
        """
        obs = batch_data['obs'].to(self.device)
        action = batch_data['action'].to(self.device)
        next_obs = batch_data['next_obs'].to(self.device)
        next_action = batch_data['next_action'].to(self.device)
        discount = batch_data.get('discount', torch.ones(obs.size(0), device=self.device))
        # For now, z is set to all zeros
        # Original implementation used sampled Gaussian noise, not sure of its purpose.
        z = torch.zeros(next_action.size(0), self.z_dim, device=self.device)
        # Encode obs and next_obs to required features
        obs_features = self.encode_images(obs.transpose(0, 3, 1, 2))
        next_obs_features = self.encode_images(next_obs.transpose(0, 3, 1, 2))
        obs = obs_features
        next_obs = next_obs_features
        # Target network forward pass
        with torch.no_grad():
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)
            target_B = self.backward_target_net(next_obs)
            target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)
            target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)
            target_M = torch.min(target_M1, target_M2)
        
        # Main network forward pass
        F1, F2 = self.forward_net(obs, z, action)
        B = self.backward_net(next_obs)
        
        # Compute success metric matrix
        M1 = torch.einsum('sd, td -> st', F1, B)
        M2 = torch.einsum('sd, td -> st', F2, B)
        
        # FB loss
        I = torch.eye(M1.size(0), device=self.device)
        off_diag = ~I.bool()
        
        fb_offdiag = 0.5 * sum((M - discount.unsqueeze(1) * target_M)[off_diag].pow(2).mean() 
                              for M in [M1, M2])
        fb_diag = -sum(M.diag().mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag
        
        # Orthogonality loss
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        
        # Total loss
        total_loss = fb_loss + self.ortho_coef * orth_loss
        
        # Optimization
        self.fb_optimizer.zero_grad()
        total_loss.backward()
        self.fb_optimizer.step()
        
        # Update target networks
        self._soft_update_targets()
        
        # Return metrics
        return {
            'fb_loss': fb_loss.item(),
            'fb_offdiag': fb_offdiag.item(),
            'fb_diag': fb_diag.item(),
            'orth_loss': orth_loss.item(),
            'total_fb_loss': total_loss.item(),
            'B_norm': torch.norm(B, dim=-1).mean().item(),
            'z_norm': torch.norm(z, dim=-1).mean().item()
        }
    
    def compute_reward(self, obs, action, target):
        """
        Compute FB intrinsic reward

        Args:
            obs: Observation (can be single or batch)
            action: Action
            target: Target observation

        Returns:
            reward: Intrinsic reward
        """
        with torch.no_grad():
            # Convert to tensor
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
            
            # Ensure batch dimension
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            if len(target.shape) == 1:
                target = target.unsqueeze(0)
            
            # Compute features
            z = torch.zeros(action.size(0), self.z_dim, device=self.device)
            F1, F2 = self.forward_net(obs, z, action)
            B = self.backward_net(target)

            # Compute reward (dot product of F and B, take diagonal)
            M1 = torch.einsum('sd, td -> st', F1, B)
            M2 = torch.einsum('sd, td -> st', F2, B)
            reward = torch.min(M1.diag(), M2.diag())
            # breakpoint()
            return reward.cpu().numpy()
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        soft_update(self.forward_target_net, self.forward_net, self.target_tau)
        soft_update(self.backward_target_net, self.backward_net, self.target_tau)
    
    def sync_from(self, other_manager):
        """Sync network weights from another manager"""
        self.forward_net.load_state_dict(other_manager.forward_net.state_dict())
        self.backward_net.load_state_dict(other_manager.backward_net.state_dict())
        self.forward_target_net.load_state_dict(other_manager.forward_target_net.state_dict())
        self.backward_target_net.load_state_dict(other_manager.backward_target_net.state_dict())
        self.fb_optimizer.load_state_dict(other_manager.fb_optimizer.state_dict())
    
    def save(self, path):
        """Save network state"""
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
        """Load network state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.forward_net.load_state_dict(checkpoint['forward_net'])
        self.backward_net.load_state_dict(checkpoint['backward_net'])
        self.forward_target_net.load_state_dict(checkpoint['forward_target_net'])
        self.backward_target_net.load_state_dict(checkpoint['backward_target_net'])
        self.fb_optimizer.load_state_dict(checkpoint['fb_optimizer'])