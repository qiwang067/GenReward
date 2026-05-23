import atexit
import sys
import threading
import traceback

import cloudpickle
import time
from functools import partial
import gym
import numpy as np
import datetime
import uuid
import torch
from tools import AttrDict

from s3dg import S3D

import vae3d
from vae3d import read_video_frames

from pathlib import Path
from gym import spaces
import hydra
# from rewards import build_reward_manager_for_calvin
from omegaconf import OmegaConf
from typing import Optional, Dict, Any



# import IV2
# from IV2 import video2feature

# import cogvideox_vae
# from diffusers import CogVideoXImageToVideoPipeline
# from huggingface_hub import snapshot_download

# from transformers import pipeline
from fb_net import FBNetworkManager

import swanlab as wandb
from kitchen_env_wrappers import readGif
from ts_video import read_tesseract_video

from PIL import Image
import cv2
import numpy as np

class DeepMindLabyrinth(object):
    ACTION_SET_DEFAULT = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (0, 0, 0, -1, 0, 0, 0),  # Backward
        (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
        (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
        (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
        (0, 0, 0, 0, 1, 0, 0),  # Fire
    )

    ACTION_SET_MEDIUM = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (0, 0, 0, -1, 0, 0, 0),  # Backward
        (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
        (0, 0, 0, 0, 0, 0, 0),  # Idle.
    )

    ACTION_SET_SMALL = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
    )

    def __init__(
            self, level, mode, action_repeat=4, render_size=(64, 64),
            action_set=ACTION_SET_DEFAULT, level_cache=None, seed=None,
            runfiles_path=None):
        assert mode in ('train', 'test')
        import deepmind_lab
        if runfiles_path:
            print('Setting DMLab runfiles path:', runfiles_path)
            deepmind_lab.set_runfiles_path(runfiles_path)
        self._config = {}
        self._config['width'] = render_size[0]
        self._config['height'] = render_size[1]
        self._config['logLevel'] = 'WARN'
        if mode == 'test':
            self._config['allowHoldOutLevels'] = 'true'
            self._config['mixerSeed'] = 0x600D5EED
        self._action_repeat = action_repeat
        self._random = np.random.RandomState(seed)
        self._env = deepmind_lab.Lab(
            level='contributed/dmlab30/' + level,
            observations=['RGB_INTERLEAVED'],
            config={k: str(v) for k, v in self._config.items()},
            level_cache=level_cache)
        self._action_set = action_set
        self._last_image = None
        self._done = True

    @property
    def observation_space(self):
        shape = (self._config['height'], self._config['width'], 3)
        space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return gym.spaces.Dict({'image': space})

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._action_set))

    def reset(self):
        self._done = False
        self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
        obs = self._get_obs()
        return obs

    def step(self, action):
        raw_action = np.array(self._action_set[action], np.intc)
        reward = self._env.step(raw_action, num_steps=self._action_repeat)
        self._done = not self._env.is_running()
        obs = self._get_obs()
        return obs, reward, self._done, {}

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        del args  # Unused
        del kwargs  # Unused
        return self._last_image

    def close(self):
        self._env.close()

    def _get_obs(self):
        if self._done:
            image = 0 * self._last_image
        else:
            image = self._env.observations()['RGB_INTERLEAVED']
        self._last_image = image
        return {'image': image}


class GymWrapper:

    def __init__(self, task, obs_key='image', act_key='action'):
        self._env = gym.make(task)
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name)

    @property
    def obs_space(self):
        #if self._obs_is_dict:
        #    spaces = self._env.observation_space.spaces.copy()
        #else:
        #    spaces = {self._obs_key: self._env.observation_space}
        return {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "state": self._env.observation_space,
            "achieved_goal": self._env.observation_space,
            "desired_goal": self._env.observation_space,
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            "success": gym.spaces.Box(0, 1, (), dtype=np.bool)
        }

    @property
    def action_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()[self._act_key]
        else:
            return self._env.action_space

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        obs = {self._obs_key: self._env.render(mode='rgb_array', width=64, height=64),
               'reward': float(reward),
               'is_first': False,
               'is_last': done,
               'is_terminal': info.get('is_terminal', done),
               'state': np.array(state['observation']),
               'achieved_goal': np.array(state['achieved_goal']),
               'desired_goal': np.array(state['desired_goal'])}
        info['success'] = int(obs['is_terminal'])
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        obs = {self._obs_key: self._env.render(mode='rgb_array', width=64, height=64),
               'reward': 0.0,
               'is_first': True,
               'is_last': False,
               'is_terminal': False,
               'state': np.array(state['observation']),
               'achieved_goal': np.array(state['achieved_goal']),
               'desired_goal': np.array(state['desired_goal'])}
        return obs


class RoboDeskGC:

    def __init__(self, task, obs_key='image', act_key='action',reward_case=None, reward_type=None, alpha=1, device=None, video_path=None, human=True, ts=False, fb_config=None, text_string=None, viper_config=None):
        import robodesk
        self._env = robodesk.RoboDesk(task=task, reward='dense', action_repeat=1, episode_length=500, image_size=64)
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = (64,64)
        self.counter = 0
        self.past_observations = []
        self.reward_type = reward_type
        self.reward_case = reward_case
        self.alpha = alpha
        self.device = device
        
        if reward_case == 'vae' or reward_case == 'vaefb':
            vae_path = "/code/taskpipeline/vae"
            vae_model_path="/code/cogkit/models/CogVideoX-5b-I2V" 
            vae_lora_path="/code/tesseract/tesseract_v01e_rgb_lora"
            # breakpoint()
            self.vae = vae3d.VAETester(vae_model_path, device=device, lora_path=vae_lora_path, vae_path=vae_path) 
            if video_path:
                frames = read_video_frames(video_path)
                video_tensor = self.vae.preprocess_video_for_vae(frames, (480,480), 16)
                latents = self.vae.encode_video(video_tensor)
                self.target_embedding = latents
        # breakpoint()
        # results = self.vae.compare_video_latents(videos_list, (480,480), 16, 'cosine_flatten')
        
        
    #     lora_path = "/code/tesseract/tesseract_v01e_rgb_lora"
    #     model_path = "/code/cogkit/models/CogVideoX-5b-I2V" 
    #     tester = cogvideox_vae.VAETester(model_path, "cuda", 'float32', lora_path=lora_path)
        
    #     video1="/code/taskpipeline/demos/pick_place.mp4"
    #     video2="/code/taskpipeline/demos/success_pick_place.mp4"
        
    #     results = tester.compare_video_latents(
    #     video_paths=[video2, video1],
    #     target_size=(480, 480),
    #     target_frames=16,
    # )
        # breakpoint()
        
        if reward_case == 'iv2':
            iv2_config = IV2.read_config_from_file()
            self.net_iv2 = IV2.build_from_config(iv2_config)
            self.tokenizer_iv2 = self.net_iv2[1]
            self.net_iv2 = self.net_iv2[0]
            self.target_embedding_iv2 = None
        # Load the model weights
        if reward_case == 's3d':
            self.net = S3D('s3d/s3d_dict.npy', 512)
            self.net.load_state_dict(torch.load('s3d/s3d_howto100m.pth'))
            self.net = self.net.eval()
            self.target_embedding = None
        
        
        if reward_case == 'fb' or reward_case == 'vaefb':
            # breakpoint()
            fb_config = AttrDict(fb_config)
            # self.forward_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
            #                           cfg.feature_dim, cfg.hidden_dim,
            #                           preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
            fb_config.device = self.device
            self.fb_manager = FBNetworkManager(fb_config, dreamer_encoder=None)
            self.fb_reward_start = fb_config.fb_train_until
            self.use_fb_reward = False
            # self.use_fb_reward = True
            if hasattr(fb_config, "fb_ckpt") and fb_config.fb_ckpt is not None:
                self.load_fb_networks(fb_config.fb_ckpt)
            # breakpoint()
        if text_string:
            text_output = self.net.text_module([text_string])
            self.target_embedding = text_output['text_embedding']
        if video_path:
            if ts:
                frames = read_tesseract_video(video_path, asNumpy=True)
            else:
                frames = readGif(video_path)

            if reward_case == 'iv2':
                video_output_iv2 = IV2.video2feature(frames, self.net_iv2, config={})
                self.target_embedding_iv2 = video_output_iv2[0]# (1,C) -> (C)
                assert self.target_embedding_iv2 is not None

            if reward_case == 'fb' or reward_case == 'vaefb':
                from frame_selector import get_max_similarity_frame
                text_prompts = [
                "robot arm grasping an object",
                "robotic gripper holding something", 
                "robot hand with object holded",
                "robot arm gripping item",
                "robot manipulator with grasped object"
                ]
                idx= get_max_similarity_frame(video_path, text_prompts, device=self.device)
                # breakpoint()
                self.last_frame = frames[idx] # np array HW3
            
            if reward_case == 's3d':
                if human or ts:
                    frames = self.preprocess_human_demo(frames)
                else:
                    frames = self.preprocess_metaworld(frames)
                if frames.shape[1]>3:
                    frames = frames[:,:3]
                video = torch.from_numpy(frames)
                video_output = self.net(video.float())
                self.target_embedding = video_output['video_embedding']
                assert self.target_embedding is not None
    
    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        # time, chanel,
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames 
    
    def frames_to_mp4(self, frames, output_path, fps=10):
        # breakpoint()
        if len(frames) == 0:
            raise ValueError("Frames list cannot be empty")
        
        # Ensure output path has .mp4 extension
        if not output_path.endswith('.mp4'):
            output_path = output_path.replace('.gif', '.mp4') if output_path.endswith('.gif') else output_path + '.mp4'
        
        # Get frame dimensions
        first_frame = frames[0]
        if first_frame.dtype != np.uint8:
            # Normalize if values are in [0,1] range
            if first_frame.max() <= 1.0:
                first_frame = (first_frame * 255).astype(np.uint8)
            else:
                first_frame = first_frame.astype(np.uint8)
        
        height, width = first_frame.shape[:2]
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # Normalize if values are in [0,1] range
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Grayscale - convert to 3 channels
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            out.write(frame_bgr)
        
        # Release everything
        out.release()
        
        print(f"MP4 saved to: {output_path}")
    
    def frames_to_gif(self, frames, output_path, duration=100, loop=0):
        # breakpoint()
        if len(frames) == 0:
        # if not frames:
            raise ValueError("Frames list cannot be empty")
        
        # Convert numpy arrays to PIL Images
        pil_images = []
        for frame in frames:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # Normalize if values are in [0,1] range
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert to PIL Image
            if len(frame.shape) == 3:
                pil_image = Image.fromarray(frame, 'RGB')
            else:
                pil_image = Image.fromarray(frame, 'L')  # Grayscale
            
            pil_images.append(pil_image)
    
            # Save as GIF
        pil_images[0].save(
                output_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=loop
            )
            
        print(f"GIF saved to: {output_path}")
    
    def preprocess_metaworld_crop_only(self, frames, shorten=True):
        # self.frames_to_gif(frames, 'metaworld.gif', duration=100, loop=0)
        # self.frames_to_mp4(frames, 'metaworld.mp4')
        # breakpoint()
        if self.crop:
            # print('111111')
            # breakpoint()
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            frames = [frame[y:y+h, x:x+w] for frame in frames]
        else:
            return frames
        return frames
    
    def preprocess_metaworld(self, frames, shorten=True):
        # self.frames_to_gif(frames, 'metaworld.gif', duration=100, loop=0)
        # self.frames_to_mp4(frames, 'metaworld.mp4')
        # breakpoint()
        frames = np.array(frames)
        # print('22222')
        # breakpoint()
        # frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        
        # self.frames_to_mp4(frames, 'crop_metaworld.mp4')
        # breakpoint()
        # self.frames_to_gif(frames, 'crop_metaworld.gif', duration=100, loop=0)
        
        a = frames
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name) 

    @property
    def obs_space(self):
        #if self._obs_is_dict:
        #    spaces = self._env.observation_space.spaces.copy()
        #else:
        #    spaces = {self._obs_key: self._env.observation_space}
        return {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            "success": gym.spaces.Box(0, 1, (), dtype=np.bool)
        }
    
    def eval_fb_as_reward(self,):
        self.use_fb_reward = True

    def load_fb_networks(self, fb_ckpt_path):
        self.fb_manager.load_state_dict(torch.load(fb_ckpt_path))

    def setup_fb_encoder(self, dreamer_encoder, dreamer_preprocess=None):
        """
        agentDreamer encoderpreprocessgoal feature

        Args:
            dreamer_encoder: agent._wm.encoder
            dreamer_preprocess: agent._wm.preprocess (obs)
        """
        if not hasattr(self, 'fb_manager'):
            return

        print("[MetaWorldGC] Setting up FB encoder...")
        self.fb_manager.dreamer_encoder = dreamer_encoder
        self.fb_manager.dreamer_preprocess = dreamer_preprocess

        if dreamer_preprocess is None:
            print("[MetaWorldGC] Warning: dreamer_preprocess not provided, FB may not work correctly!")

        if hasattr(self, 'last_frame') and self.last_frame is not None:
            goal_tensor = torch.from_numpy(self.last_frame).to(self.device)
            self.fb_manager.set_goal_feature(goal_tensor)
        print("[MetaWorldGC] FB encoder setup complete!")

    def train_fb_networks(self, fb_train):
        """
        Use in training case, train Forward & Backward feature Network.
        Compute contrastive loss and optimize network.
        """
        data = next(fb_train)
        obs = data['image']  # (B, 2, H, W, C) numpy array
        batch_size = obs.shape[0]

        obs_t0 = torch.from_numpy(obs[:,0]).to(self.device)  # (B, H, W, C)
        obs_t1 = torch.from_numpy(obs[:,1]).to(self.device)
        action_t0 = torch.from_numpy(data["action"][:,0]).to(self.device)
        action_t1 = torch.from_numpy(data["action"][:,1]).to(self.device)
        discount_t0 = torch.from_numpy(data["discount"][:,0]).to(self.device)

        batch_data = {
            "obs": obs_t0,
            "action": action_t0,
            "next_action": action_t1,
            "next_obs": obs_t1,
            "discount": discount_t0,
        }
        if self.reward_case == 'fb' or self.reward_case == 'vaefb':
            return self.fb_manager.update(batch_data)
        return {}

    def sync_fb(self, other_mgc):
        """ FB """
        if (self.reward_case == 'fb' or self.reward_case == 'vaefb') and hasattr(other_mgc, 'fb_manager'):
            self.fb_manager.sync_from(other_mgc.fb_manager)
    
    def get_fb_network(self):
        """ FB """
        if self.reward_case == 'fb' or self.reward_case == 'vaefb':
            return self.fb_manager.state_dict()
        return None
    
    def compute_fb_reward_from_img_pair(self, obs, action, target_obs):
        if self.reward_case == 'fb' or self.reward_case == 'vaefb':
            return self.fb_manager.compute_reward_from_img_pair(obs, action, target_obs)
        return 0.0


    @property
    def action_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()[self._act_key]
        else:
            return self._env.action_space

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        obs = {'reward': float(reward),
               'is_first': False,
               'is_last': done,
               'is_terminal': info.get('is_terminal', done)}
        obs.update(state)
        info['success'] = int(obs['is_terminal'])
        self.past_observations.append(obs['image'])
        self.counter += 1
        
        if self.reward_type == 'dense':
            # reward += rew or 0.0
            info['extrinsic_reward'] = reward
            if self.counter%128==0:
                if self.reward_case == 'iv2':
                    frames = self.preprocess_metaworld(self.past_observations)
                    frames = frames.transpose(0,2,3,4,1)[0]
                    frames = frames.tolist()
                    frames = [np.array(frame).astype(np.uint8) for frame in frames]
                    video_embedding_iv2 = video2feature(frames, self.net_iv2, config={})
                    video_embedding_iv2 = video_embedding_iv2[0] # (1,C) -> (C)
                    similarity_matrix_iv2 = torch.matmul(self.target_embedding_iv2, video_embedding_iv2.t())
                    iv2_reward = similarity_matrix_iv2.detach().cpu().numpy()
                    reward = reward + iv2_reward
                    info['iv2_reward'] = iv2_reward
                if self.reward_case == 's3d':
                    frames = self.preprocess_metaworld(self.past_observations)
                    video = torch.from_numpy(frames)
                    video_output = self.net(video.float())
                    video_embedding = video_output['video_embedding']
                    similarity_matrix = torch.matmul(self.target_embedding, video_embedding.t())
                    s3d_reward = similarity_matrix.detach().numpy()[0][0]
                    reward = reward + s3d_reward
                    info['s3d_reward'] = s3d_reward
                if self.reward_case == 'vae' or self.reward_case == 'vaefb':
                    frames = self.past_observations
                    # breakpoint()
                    video_tensor = self.vae.preprocess_video_for_vae(frames, (480,480), 16)
                    # breakpoint()
                    latents = self.vae.encode_video(video_tensor)
                    similarity = self.vae.compute_vae_similarity(
                    self.target_embedding, 
                    latents, 
                    method='cosine_flatten'
                )
                    
                    instrinsic_reward = self.alpha * similarity
                    reward = reward +  instrinsic_reward
                    
                    info['vae_reward'] = instrinsic_reward
                    # breakpoint()
                if (self.reward_case == 'fb' or self.reward_case == 'vaefb') and self.use_fb_reward:
                    
                    frames = self.preprocess_metaworld(self.past_observations)
                    frames = frames.transpose(0,2,3,4,1)[0] # T,C,H,W
                    target_obs = torch.tensor(self.last_frame.transpose(2,0,1)).unsqueeze(0)
                    current_obs = torch.tensor(self.past_observations[-1].transpose(2,0,1)).unsqueeze(0)
                    fb_reward = self.compute_fb_reward_from_img_pair(current_obs, action, target_obs)
                    
                    # breakpoint()
                    if hasattr(self, 'use_vae_fb') and self.use_vae_fb:
                        print('Using VAE features to compute FB reward')
                        # breakpoint()
                        current_frame = self.past_observations[-1]
                        target_frame = self.last_frame
                        
                        current_vae_features = self.extract_vae_features_from_single_frame(current_frame)
                        target_vae_features = self.extract_vae_features_from_single_frame(target_frame)
                        
                        vae_fb_reward = self.compute_fb_reward_from_vae_features(current_vae_features, action, target_vae_features)
                        info['vae_fb_reward'] = vae_fb_reward
                        fb_reward = vae_fb_reward
                    
                    scale_fb_reward = 0.00001 * fb_reward
                    reward += scale_fb_reward
                    info['fb_reward'] = scale_fb_reward
                        
                if self.reward_case == 'vaefb':
                    if self.use_fb_reward:
                        # breakpoint()
                        info['vaefb_reward'] = scale_fb_reward + instrinsic_reward
                    else:
                        info['vaefb_reward'] = instrinsic_reward
                        
                    # breakpoint()
        
                    # info['vaefb_reward'] = fb_reward + instrinsic_reward
                        
                        
                self.past_observations=[]
                    
        if self.reward_type == 'sparse':
            if self.counter >= 128:
                frames = self.preprocess_metaworld(self.past_observations)
                video = torch.from_numpy(frames)
                video_output = self.net(video.float())
                video_embedding = video_output['video_embedding']
                similarity_matrix = torch.matmul(self.target_embedding, video_embedding.t())
                reward = similarity_matrix.detach().numpy()[0][0]
                info['s3d_reward'] = s3d_reward
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        obs = {'reward': 0.0,
               'is_first': True,
               'is_last': False,
               'is_terminal': False}
        obs.update(state)
        return obs

class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()

class DeepMindControl:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:
    LOCK = threading.Lock()

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky_actions=True, all_actions=False):
        assert size[0] == size[1]
        import gym.wrappers
        import gym.envs.atari
        if name == 'james_bond':
            name = 'jamesbond'
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name, obs_type='image', frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                full_action_space=all_actions)
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale)
        self._env = env
        self._grayscale = grayscale

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'image': self._env.observation_space,
            'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        })

    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.env._get_ram()}
        return obs

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.env._get_ram()}
        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)


class CollectDataset:

    def __init__(self, env, callbacks=None, precision=32):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        if isinstance(action, dict):
            transition.update(action)
        else:
            transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount', np.array(1 - float(done)))
        self._episode.append(transition)
        if done:
            for key, value in self._episode[1].items():
                if key not in self._episode[0]:
                    self._episode[0][key] = 0 * value
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            #print(info['success'])
            episode = {k: self._convert(v) for k, v in episode.items()}
            info['episode'] = episode
            for callback in self._callbacks:
                callback(episode)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        # Missing keys will be filled with a zeroed out version of the first
        # transition, because we do not know what action information the agent will
        # pass yet.
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self._episode = [transition]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        elif np.issubdtype(value.dtype, np.bool_):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)



class MetaWorld:
    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera=None, gpu="cuda:0", high_res_size=(1024, 1024)):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )
        import os
        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        print(task)
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._high_res_size = high_res_size
        self._action_repeat = action_repeat
        self._camera = camera
        self._gpu_id = int(gpu.split(':')[1])
        self._count = 0


    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        return spaces

    @property
    def action_space(self):
        action = self._env.action_space
        return action

    def render_high_res(self):
        """
        
        : numpy (height, width, 3)dtypeuint8
        """
        return self._env.sim.render(
            *self._high_res_size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
        )

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        done = None
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action)
            # print('state is',state)
            # assert 1==2
            success += float(info["success"])
            reward += rew or 0.0
            if done:
                break
        success = min(success, 1.0)
        assert success in [0.0, 1.0]

        obs = {
            "reward": reward,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            ),
            "state": state,
            "is_first": False,
            "is_terminal": success
        }
        info['success'] = success
        if success!=0.0:
            print('succ is',success)
        # print('obs reward is',obs['reward'])
        # assert 1==2
        return obs, reward, done, info

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            ),
            "state": state,
            "is_first": True,
            "is_terminal": False
        }
        return obs





class CalvinPlayTable:
    """
    DreamerCALVIN wrappers.MetaWorld
    -  PlayTableSimEnv get_env(...) 
    -  task_name =10
    - imagestaticstaterobot_obs
    - success  info['success']
    """
    def __init__(self, dataset_path: Optional[str],
                 task_name: str,
                 ep_len: int = 256,
                 size=(64, 64),
                 camera: str = "rgb_static",
                 show_gui: bool = False,
                 scene: Optional[str] = None):
        conf_dir = Path(__file__).parent / "third_party/calvin/calvin_env/conf"
        task_cfg = OmegaConf.load(conf_dir / "tasks/new_playtable_tasks.yaml")
        self._tasks = hydra.utils.instantiate(task_cfg)
        conf_dir = Path(__file__).parent / "third_party/calvin/calvin_models/conf"
        self._val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

        self._task = task_name
        self._lang = self._val_annotations[self._task][0]

        if dataset_path:
            env_dataset_path = Path(dataset_path) / "validation"
            self._env = get_env(env_dataset_path, show_gui=show_gui, scene=scene)
        else:
            self._env = get_env(None, show_gui=show_gui, scene=scene)

        self._size = size
        self._ep_len = ep_len
        self._t = 0
        self._start_info = None

        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        obs0 = self._env.reset()
        img0 = self._render_rgb(obs0)
        state0 = self._vectorize_robot_state(obs0)
        self._obs_space = {
            "image": spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": spaces.Box(0, 1, (), dtype=np.bool_),
            "is_last": spaces.Box(0, 1, (), dtype=np.bool_),
            "is_terminal": spaces.Box(0, 1, (), dtype=np.bool_),
            "state": spaces.Box(-np.inf, np.inf, shape=state0.shape, dtype=np.float32),
        }

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, options=None):
        """
        options {"robot_obs": np.ndarray, "scene_obs": np.ndarray} 
        """
        if options and "robot_obs" in options and "scene_obs" in options:
            obs = self._env.reset(robot_obs=options["robot_obs"], scene_obs=options["scene_obs"])
        else:
            obs = self._env.reset()
        self._start_info = self._env.get_info()
        self._t = 0
        img = self._render_rgb(obs)
        state = self._vectorize_robot_state(obs)
        out = {
            "image": img,
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "state": state,
        }
        return out

    def step(self, action: np.ndarray):
        assert np.isfinite(action).all(), action
        action = np.clip(action, self._action_space.low, self._action_space.high)
        # map gripper action to binary -1,1
        action[-1] = 1 if action[-1] > 0 else -1
        obs, _, _, info = self._env.step(action)
        self._t += 1

        success = len(self._tasks.get_task_info_for_set(self._start_info, info, {self._task})) > 0
        reward = float(success)
        done = bool(success)
        truncated = self._t >= self._ep_len
        img = self._render_rgb(obs)
        state = self._vectorize_robot_state(obs)

        out = {
            "image": img,
            "reward": reward,
            "is_first": False,
            "is_last": done or truncated,
            "is_terminal": bool(success),
            "state": state,
        }
        info = dict(info)
        info.setdefault("start_info", self._start_info)
        info.setdefault("task_oracle", self._tasks)
        info["success"] = bool(success)
        return out, reward, (done or truncated), info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError("Only 'rgb_array' is supported.")
        return self._env.render(mode="rgb_array")

    def close(self):
        self._env.close()

    def set_task(self, task_name: str):
        """"""
        self._task = task_name
        self._lang = self._val_annotations[self._task][0]

    def _render_rgb(self, obs):
        img = self._env.render(mode="rgb_array")
        if img.shape[:2] != self._size:
            import cv2
            img = cv2.resize(img, self._size[::-1])
        return img

    def _vectorize_robot_state(self, obs) -> np.ndarray:
        """
         CALVIN  robot_obs(dict)  Dreamer
         'robot_state_full'
        - robot_state: ndarray (16,) By default we use euler convention(the latter, 7-dim include gripper)
            - tcp_pos: robot_state[:3]
            - tcp_orn: robot_state[3:7] (quat) / [3:6] (euler)
            - gripper_opening_width: robot_state[7:8] (quat) / [6:7] (euler)
            - arm_joint_states: robot_state[8:15] (quat) / [7:14] (euler)
            - gripper_action: robot_state[15:] (quat) / [14:] (euler)
        - robot_info: Dict
        """
        rob = obs.get("robot_obs", {})
        return rob[:7]

class CalvinGC(CalvinPlayTable):
    """
    CalvinGCCalvinPlayTable RewardManager
    - reward_specs: list of dict rewards.build_reward_manager_for_calvin  success reward
    - device:  GPU/
    
      - reset()  reset  reward manager start_info
      - step()  env.step(),  (frame, action, env_info)  RewardManager
      - episode  task oracle RewardManager  success reward reward  success 
    """
    def __init__(self,
                 dataset_path: Optional[str],
                 task_name: str,
                 reward_specs: list = None,
                 reward_case: Optional[str] = None,
                 device: str = None,
                 ep_len: int = 240,
                 size=(64, 64),
                 camera: str = "rgb_static",
                 show_gui: bool = False,
                 scene: Optional[str] = None):
        super().__init__(dataset_path, task_name, ep_len=ep_len, size=size, camera=camera, show_gui=show_gui, scene=scene)
        self._task_name = task_name
        self._reward_manager = build_reward_manager_for_calvin(reward_specs, device=device)
        self._reward_device = device
        self._fb_train_iter = None
        self._global_step = 0
        self.reward_case = reward_case or self._infer_reward_case(reward_specs)
        self._fb_eval_enabled = False

    def _infer_reward_case(self, reward_specs: Optional[list]) -> Optional[str]:
        if not reward_specs:
            return None
        types = {spec.get("type") for spec in reward_specs if spec.get("type")}
        if {"fb", "vae"}.issubset(types):
            return "vaefb"
        for candidate in ("fb", "vae", "s3d", "iv2"):
            if candidate in types:
                return candidate
        return next(iter(types), None)

    def reset(self, options=None):
        obs = super().reset(options=options)
        self._global_step = 0
        self._fb_eval_enabled = False
        if self._reward_manager:
            self._reward_manager.reset()
        return obs

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        # breakpoint()
        extra_reward = 0.0
        reward_info: Dict[str, Any] = {}
        if self._reward_manager:
            frame = obs.get("image")
            env_info = dict(info)
            env_info.setdefault("start_info", getattr(self, "_start_info", None))
            env_info.setdefault("current_info", dict(info))
            env_info.setdefault("task_oracle", getattr(self, "_tasks", None))
            env_info["task_name"] = self._task_name
            extra_reward, reward_info = self._reward_manager.on_step(
                frame, action, env_info, global_step=self._global_step
            )
            if self._fb_train_iter is not None:
                self._reward_manager.train({"fb": self._fb_train_iter}, global_step=self._global_step)
        # breakpoint()
        extra_reward = 0
        reward = reward + extra_reward
        obs["reward"] = reward
        info.update(reward_info)
        if self._reward_manager:
            info["reward_manager_reward"] = extra_reward
        self._global_step += 1
        return obs, reward, done, info

    def attach_fb_train(self, fb_train):
        self._fb_train_iter = fb_train

    def train_fb_networks(self, fb_train=None):
        if fb_train is not None:
            self._fb_train_iter = fb_train
        if self._reward_manager and self._fb_train_iter is not None:
            results = self._reward_manager.train({"fb": self._fb_train_iter}, global_step=self._global_step)
            if isinstance(results, dict):
                return results.get("fb", results)
            return results
        return {}

    def eval_fb_as_reward(self):
        if not self._reward_manager or self._fb_eval_enabled:
            return
        fb_reward = self._reward_manager.get("fb")
        if fb_reward:
            fb_reward.enable()
            self._fb_eval_enabled = True

    def sync_fb(self, other_gc):
        if not self._reward_manager:
            return
        peer_mgr = getattr(other_gc, "_reward_manager", None)
        if peer_mgr:
            self._reward_manager.sync_from(peer_mgr)

    def get_fb_network(self):
        if not self._reward_manager:
            return None
        fb_reward = self._reward_manager.get("fb")
        return fb_reward.state_dict() if fb_reward else None

    def load_fb_network(self, state_dict: Optional[Dict[str, Any]]):
        if not self._reward_manager or not state_dict:
            return
        fb_reward = self._reward_manager.get("fb")
        if fb_reward:
            fb_reward.load_state_dict(state_dict)


class MetaWorldGC:
    def __init__(self, name, sparse_interval=None, alpha=1, device=None, crop=None, seed=None, action_repeat=1, size=(64, 64), camera=None, text_string=None, video_path=None, 
                 human=True, ts=False, reward_type=None, reward_case=None, fb_config=None, gpu="cuda:0", use_vae_fb=False, viper_config=None, tadpole_config=None):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )
        import os
        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        print(task)
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self._gpu_id = int(gpu.split(':')[1])
        # self._count = 0
        self.counter = 0
        self.past_observations = []
        self.reward_type = reward_type
        self.reward_case = reward_case
        self.sparse_interval = sparse_interval
        self.alpha = alpha
        self.device = device
        self.fb_reward_scaled = True
        self.beta_scale = 0.0001
        self.use_vae_fb = use_vae_fb
        # breakpoint()
        if crop == 'false':
            self.crop = False
        else:
            self.crop = True

        if reward_case == 'vae' or reward_case == 'vaefb':
            vae_path = "/code/taskpipeline/vae"
            vae_model_path="/code/cogkit/models/CogVideoX-5b-I2V" 
            vae_lora_path="/code/tesseract/tesseract_v01e_rgb_lora"
            # breakpoint()
            self.vae = vae3d.VAETester(vae_model_path, device=device, lora_path=vae_lora_path, vae_path=vae_path) 
            if video_path:
                frames = read_video_frames(video_path)
                video_tensor = self.vae.preprocess_video_for_vae(frames, (480,480), 16)
                latents = self.vae.encode_video(video_tensor)
                self.target_embedding = latents
        # breakpoint()
        # results = self.vae.compare_video_latents(videos_list, (480,480), 16, 'cosine_flatten')
        
        
    #     lora_path = "/code/tesseract/tesseract_v01e_rgb_lora"
    #     model_path = "/code/cogkit/models/CogVideoX-5b-I2V" 
    #     tester = cogvideox_vae.VAETester(model_path, "cuda", 'float32', lora_path=lora_path)
        
    #     video1="/code/taskpipeline/demos/pick_place.mp4"
    #     video2="/code/taskpipeline/demos/success_pick_place.mp4"
        
    #     results = tester.compare_video_latents(
    #     video_paths=[video2, video1],
    #     target_size=(480, 480),
    #     target_frames=16,
    # )
        # breakpoint()
        
        if reward_case == 'iv2':
            iv2_config = IV2.read_config_from_file()
            self.net_iv2 = IV2.build_from_config(iv2_config)
            self.tokenizer_iv2 = self.net_iv2[1]
            self.net_iv2 = self.net_iv2[0]
            self.target_embedding_iv2 = None
        # Load the model weights
        
        if reward_case == 's3d':
            self.net = S3D('s3d/s3d_dict.npy', 512)
            self.net.load_state_dict(torch.load('s3d/s3d_howto100m.pth'))
            self.net = self.net.eval()
            self.target_embedding = None
        
        
        if reward_case == 'fb' or reward_case == 'vaefb':
            # breakpoint()
            fb_config = AttrDict(fb_config)
            # self.forward_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
            #                           cfg.feature_dim, cfg.hidden_dim,
            #                           preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
            fb_config.device = self.device
            self.fb_manager = FBNetworkManager(fb_config, dreamer_encoder=None)
            self.fb_reward_start = fb_config.fb_train_until
            self.use_fb_reward = False
            # self.use_fb_reward = True
            if hasattr(fb_config, "fb_ckpt") and fb_config.fb_ckpt is not None:
                self.load_fb_networks(fb_config.fb_ckpt)
            # breakpoint()

        # Initialize Diffusion Reward (VQ-Diffusion baseline)
        if reward_case == 'diffusion':
            from rewards import DiffusionReward
            # Ensure fb_config is an AttrDict
            if fb_config is None:
                fb_config = {}
            fb_config = AttrDict(fb_config) if not isinstance(fb_config, AttrDict) else fb_config

            # Get checkpoint and config paths with defaults
            vqgan_checkpoint = fb_config.get('vqgan_checkpoint_path',
                '/code/baseline1/diffusion_reward/exp_local/codec_models/vqgan/metaworld/results/checkpoints/vqgan.pt')
            vqgan_config = fb_config.get('vqgan_config_path',
                '/code/baseline1/diffusion_reward/exp_local/codec_models/vqgan/metaworld/.hydra/config.yaml')
            vqdiffusion_checkpoint = fb_config.get('vqdiffusion_checkpoint_path',
                '/code/baseline1/diffusion_reward/exp_local/video_models/vqdiffusion/metaworld/results/checkpoints/best.pth')
            vqdiffusion_config = fb_config.get('vqdiffusion_config_path',
                '/code/baseline1/diffusion_reward/exp_local/video_models/vqdiffusion/metaworld/.hydra/config.yaml')
            stat_path = fb_config.get('diffusion_stat_path',
                '/code/baseline1/diffusion_reward/diffusion_reward/models/reward_models/statistics/diffusion_reward/entropy/metaworld.yaml')

            # Get other parameters
            diffusion_reward_type = fb_config.get('diffusion_reward_type', 'entropy')
            diffusion_keep = fb_config.get('diffusion_keep', 256)
            diffusion_interval = fb_config.get('diffusion_interval', 256)
            diffusion_weight = fb_config.get('diffusion_weight', 1.0)
            diffusion_use_std = fb_config.get('diffusion_use_std', True)
            diffusion_skip_step = fb_config.get('diffusion_skip_step', 0)
            diffusion_num_sample = fb_config.get('diffusion_num_sample', 1)
            diffusion_noise = fb_config.get('diffusion_noise', True)
            diffusion_noise_scale = fb_config.get('diffusion_noise_scale', 1e-6)
            diffusion_expl_scale = fb_config.get('diffusion_expl_scale', 0.0)

            # Extract task name from name parameter (e.g., 'pick-place')
            task_name = name

            # Create DiffusionReward instance
            self.diffusion_reward = DiffusionReward(
                vqgan_checkpoint_path=vqgan_checkpoint,
                vqgan_config_path=vqgan_config,
                vqdiffusion_checkpoint_path=vqdiffusion_checkpoint,
                vqdiffusion_config_path=vqdiffusion_config,
                stat_path=stat_path,
                task_name=task_name,
                reward_type=diffusion_reward_type,
                use_std=diffusion_use_std,
                skip_step=diffusion_skip_step,
                num_sample=diffusion_num_sample,
                noise=diffusion_noise,
                noise_scale=diffusion_noise_scale,
                device=self.device,
                keep=diffusion_keep,
                interval=diffusion_interval,
                weight=diffusion_weight,
                expl_scale=diffusion_expl_scale
            )

            # Setup the model (load checkpoints)
            self.diffusion_reward.setup()
            print(f"Diffusion Reward initialized: type={diffusion_reward_type}, keep={diffusion_keep}, interval={diffusion_interval}")

        if video_path:
            if ts:
                frames = read_tesseract_video(video_path, asNumpy=True)
            else:
                frames = readGif(video_path)

            if reward_case == 'iv2':
                video_output_iv2 = IV2.video2feature(frames, self.net_iv2, config={})
                self.target_embedding_iv2 = video_output_iv2[0]# (1,C) -> (C)
                assert self.target_embedding_iv2 is not None

            if reward_case == 'fb' or reward_case == 'vaefb':
                from frame_selector import get_max_similarity_frame
                text_prompts = [
                "robot arm grasping an object",
                "robotic gripper holding something", 
                "robot hand with object holded",
                "robot arm gripping item",
                "robot manipulator with grasped object"
                ]
                idx= get_max_similarity_frame(video_path, text_prompts, device=self.device)
                # breakpoint()
                self.last_frame = frames[idx] # np array HW3
            
            if reward_case == 's3d':
                if text_string:
                    text_output = self.net.text_module([text_string])
                    self.target_embedding = text_output['text_embedding']
                '''
                if human or ts:
                    frames = self.preprocess_human_demo(frames)
                else:
                    frames = self.preprocess_metaworld(frames)
                if frames.shape[1]>3:
                    frames = frames[:,:3]
                video = torch.from_numpy(frames)
                video_output = self.net(video.float())
                self.target_embedding = video_output['video_embedding']
                '''
                assert self.target_embedding is not None

            # Initialize VIPER reward
            if reward_case == 'viper':
                from rewards import VIPERReward
                # Use fb_config as container for viper params (or can be separate viper_config param)
                if viper_config is None:
                    viper_config = {}
                viper_config = AttrDict(viper_config) if not isinstance(viper_config, AttrDict) else viper_config

                viper_checkpoint = viper_config.get('viper_checkpoint_path',
                    '/nfs/kun2/users/mianw/CVPR/baseline/diffusion_reward/exp_local/video_models/videogpt/metaworld/results/checkpoints/videogpt.pt')
                viper_config_path = viper_config.get('viper_config_path',
                    '/nfs/kun2/users/mianw/CVPR/baseline/diffusion_reward/exp_local/video_models/videogpt/metaworld/.hydra/config.yaml')
                viper_stat_path = viper_config.get('viper_stat_path',
                    '/nfs/kun2/users/mianw/CVPR/baseline/diffusion_reward/diffusion_reward/models/reward_models/statistics/viper/likelihood/metaworld.yaml')

                self.viper_reward = VIPERReward(
                    viper_checkpoint_path=viper_checkpoint,
                    viper_config_path=viper_config_path,
                    stat_path=viper_stat_path,
                    reward_type=viper_config.get('viper_reward_type', 'likelihood'),
                    compute_joint=viper_config.get('viper_compute_joint', False),
                    use_std=viper_config.get('viper_use_std', True),
                    device=self.device,
                    keep=256,
                    interval=256,
                    weight=1.0
                )
                self.viper_reward.setup()
                print("[MetaWorldGC] VIPER reward initialized")

            if reward_case == 'tadpole':
                from tadpole import TADPoLe
                print('Running tadpole with ',text_string)
                self.guidance = TADPoLe(device=self.device, text_cond=text_string)
                self.tadpole_config = tadpole_config
                print("[MetaWorldGC] TADPoLe reward initialized")
            
            if reward_case == 'video-tadpole':
                from tadpole import VideoTADPoLe
                print('Running video-tadpole with ',text_string)
                self.guidance = VideoTADPoLe(device=self.device, text_cond=text_string)
                self.tadpole_config = tadpole_config
                print("[MetaWorldGC] Video-TADPoLe reward initialized")


    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name) 
    
    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        # time, chanel,
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames 
    
    def frames_to_mp4(self, frames, output_path, fps=10):
        # breakpoint()
        if len(frames) == 0:
            raise ValueError("Frames list cannot be empty")
        
        # Ensure output path has .mp4 extension
        if not output_path.endswith('.mp4'):
            output_path = output_path.replace('.gif', '.mp4') if output_path.endswith('.gif') else output_path + '.mp4'
        
        # Get frame dimensions
        first_frame = frames[0]
        if first_frame.dtype != np.uint8:
            # Normalize if values are in [0,1] range
            if first_frame.max() <= 1.0:
                first_frame = (first_frame * 255).astype(np.uint8)
            else:
                first_frame = first_frame.astype(np.uint8)
        
        height, width = first_frame.shape[:2]
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # Normalize if values are in [0,1] range
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Grayscale - convert to 3 channels
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            out.write(frame_bgr)
        
        # Release everything
        out.release()
        
        print(f"MP4 saved to: {output_path}")
    
    def frames_to_gif(self, frames, output_path, duration=100, loop=0):
        # breakpoint()
        if len(frames) == 0:
        # if not frames:
            raise ValueError("Frames list cannot be empty")
        
        # Convert numpy arrays to PIL Images
        pil_images = []
        for frame in frames:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # Normalize if values are in [0,1] range
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert to PIL Image
            if len(frame.shape) == 3:
                pil_image = Image.fromarray(frame, 'RGB')
            else:
                pil_image = Image.fromarray(frame, 'L')  # Grayscale
            
            pil_images.append(pil_image)
    
            # Save as GIF
        pil_images[0].save(
                output_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=loop
            )
            
        print(f"GIF saved to: {output_path}")
    
    def preprocess_metaworld_crop_only(self, frames, shorten=True):
        # self.frames_to_gif(frames, 'metaworld.gif', duration=100, loop=0)
        # self.frames_to_mp4(frames, 'metaworld.mp4')
        # breakpoint()
        if self.crop:
            # print('111111')
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            frames = [frame[y:y+h, x:x+w] for frame in frames]
        else:
            return frames
        return frames
    
    def preprocess_metaworld(self, frames, shorten=True):
        # self.frames_to_gif(frames, 'metaworld.gif', duration=100, loop=0)
        # self.frames_to_mp4(frames, 'metaworld.mp4')
        # breakpoint()
        if self.crop:
            # print('111111')
            # breakpoint()
            center = 240, 320
            h, w = (250, 250)
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        else:
            frames = np.array(frames)
        # print('22222')
        # breakpoint()
        # frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        
        # self.frames_to_mp4(frames, 'crop_metaworld.mp4')
        # breakpoint()
        # self.frames_to_gif(frames, 'crop_metaworld.gif', duration=100, loop=0)
        
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames
    
    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        return spaces

    @property
    def action_space(self):
        action = self._env.action_space
        return action

    def render(self):
        frame = self._env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame

    # def load_fb_networks(self, fb_ckpt_path):
    #     """
    #     Load Forward & Backward feature Network from checkpoint.
    #     """
    #     ckpt = torch.load(fb_ckpt_path)
    #     self.forward_net.load_state_dict(ckpt["forward_net"])
    #     self.backward_net.load_state_dict(ckpt["backward_net"])
    #     self.forward_target_net.load_state_dict(ckpt["forward_target_net"])
    #     self.backward_target_net.load_state_dict(ckpt["backward_target_net"])
    def eval_fb_as_reward(self,):
        self.use_fb_reward = True

    def load_fb_networks(self, fb_ckpt_path):
        self.fb_manager.load_state_dict(torch.load(fb_ckpt_path))

    def train_fb_networks(self, fb_train):
        """
        Use in training case, train Forward & Backward feature Network.
        Compute contrastive loss and optimize network.
        """
        data = next(fb_train)
        # breakpoint()
        obs = data['image']
        batch_size = obs.shape[0]

        if hasattr(self, 'last_frame') and self.last_frame is not None:
            goal_image = np.repeat(self.last_frame[np.newaxis, :, :, :], batch_size, axis=0)
        else:
            goal_image = np.repeat(self.last_frame[np.newaxis, :, :, :], batch_size, axis=0)
            #goal_image = obs[:, 1]

        data = {
            "obs": obs[:,0],
            "action": data["action"][:,0],
            "next_action": data["action"][:,1],
            "next_obs": obs[:,1],
            "goal_image": goal_image,
            "discount": data["discount"][:,0],
        }
        if self.reward_case == 'fb' or self.reward_case == 'vaefb':
            return self.fb_manager.update(data)
        return {}

    def sync_fb(self, other_mgc):
        """ FB """
        if (self.reward_case == 'fb' or self.reward_case == 'vaefb') and hasattr(other_mgc, 'fb_manager'):
            self.fb_manager.sync_from(other_mgc.fb_manager)
    
    def get_fb_network(self):
        """ FB """
        if self.reward_case == 'fb' or self.reward_case == 'vaefb':
            return self.fb_manager.state_dict()
        return None
    # def compute_fb_reward(self, state, action, target):
    #     if self.reward_case == 'fb':
    #         if len(state.shape) > 1:
    #             state = state.flatten()
    #         if len(target.shape) > 1:
    #             target = target.flatten()
            
    #         reward = self.fb_manager.compute_reward(state, action, target)
    #         return self.fb_reward_scale * reward[0] if len(reward) == 1 else reward
    #     return 0.0
    
    def extract_vae_features_from_single_frame(self, frame):
        """
        3VAE
        
        Args:
            frame:  numpy array (H, W, 3)
            
        Returns:
            features: VAE
        """
        
        if not hasattr(self, 'vae') or self.vae is None:
            raise AttributeError("VAE not initialized. Make sure reward_case is 'vae' or 'vaefb' to initialize VAE.")
        
        frames = [frame]
        video_tensor = self.vae.preprocess_video_for_vae(frames, (480,480), 1)
        # breakpoint()
        latents = self.vae.encode_video(video_tensor)
        return latents.flatten()

    def compute_fb_reward_from_img_pair(self, obs, action, target_obs):
        if self.reward_case == 'fb' or self.reward_case == 'vaefb':
            return self.fb_manager.compute_reward_from_img_pair(obs, action, target_obs)
        return 0.0

    def compute_fb_reward_from_vae_features(self, current_features, action, target_features):
        """
        4VAEFB
        
        Args:
            current_features: VAE
            action: 
            target_features: VAE
            
        Returns:
            reward: FB
        """
        if self.reward_case == 'fb' or self.reward_case == 'vaefb':
            # breakpoint()
            if isinstance(current_features, np.ndarray):
                current_features = torch.FloatTensor(current_features).to(self.device)
            if isinstance(target_features, np.ndarray):
                target_features = torch.FloatTensor(target_features).to(self.device)
            if isinstance(action, np.ndarray):
                action = torch.FloatTensor(action).to(self.device)
            
            if len(current_features.shape) == 1:
                current_features = current_features.unsqueeze(0)
            if len(target_features.shape) == 1:
                target_features = target_features.unsqueeze(0)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            
            reward = self.fb_manager.compute_reward(current_features, action, target_features)
            return float(reward.item()) if hasattr(reward, 'item') else float(reward[0])
        return 0.0

    def setup_fb_encoder(self, dreamer_encoder, dreamer_preprocess=None):
        """
        agentDreamer encoderpreprocessgoal feature

        Args:
            dreamer_encoder: agent._wm.encoder
            dreamer_preprocess: agent._wm.preprocess (obs)
        """
        if not hasattr(self, 'fb_manager'):
            return

        print("[MetaWorldGC] Setting up FB encoder...")
        self.fb_manager.dreamer_encoder = dreamer_encoder
        self.fb_manager.dreamer_preprocess = dreamer_preprocess

        if dreamer_preprocess is None:
            print("[MetaWorldGC] Warning: dreamer_preprocess not provided, FB may not work correctly!")

        if hasattr(self, 'last_frame') and self.last_frame is not None:
            goal_tensor = torch.from_numpy(self.last_frame).to(self.device)
            self.fb_manager.set_goal_feature(goal_tensor)
        print("[MetaWorldGC] FB encoder setup complete!")

 

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        done = None
        for _ in range(self._action_repeat):
            self.counter += 1
            state, rew, done, info = self._env.step(action)
            # print('done is',done)
            # self.past_observations.append(self._env.sim.render(
            #     *self._size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            # ))
            # Use appropriate size based on reward_case
            # Diffusion/VIPER rewards need 64x64 (trained on this size)
            # Other rewards can use 480x480 for better quality
            if self.reward_case in ['diffusion', 'viper']:
                render_size = (64, 64)
            else:
                render_size = (480, 480)

            self.past_observations.append(self._env.sim.render(
                *render_size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            ))
            # breakpoint()
            # self.past_observations.append(self._env.render())
            # breakpoint()
            # print('state is',state)
            # assert 1==2
            success += float(info["success"])
            # reward += rew or 0.0
            # if done:
            # breakpoint()
            if self.reward_type == 'dense':
                reward += rew or 0.0
                info['extrinsic_reward'] = reward
            elif self.reward_type == 'sparse':
                if self.counter%self.sparse_interval==0:
                    reward += rew
                    # breakpoint()
                else:
                    reward = 0.0
                info['extrinsic_reward'] = reward
            
            elif self.reward_type == 'intrinsic_only':
                reward = 0.0
                info['extrinsic_reward'] = reward
            else:
                raise ValueError(f"Unknown reward_type: {self.reward_type}")
            
            if self.counter%128==0:
                if self.reward_case == 'iv2':
                    frames = self.preprocess_metaworld(self.past_observations)
                    frames = frames.transpose(0,2,3,4,1)[0]
                    frames = frames.tolist()
                    frames = [np.array(frame).astype(np.uint8) for frame in frames]
                    video_embedding_iv2 = video2feature(frames, self.net_iv2, config={})
                    video_embedding_iv2 = video_embedding_iv2[0] # (1,C) -> (C)
                    similarity_matrix_iv2 = torch.matmul(self.target_embedding_iv2, video_embedding_iv2.t())
                    iv2_reward = similarity_matrix_iv2.detach().cpu().numpy()
                    reward = reward + iv2_reward
                    info['iv2_reward'] = iv2_reward
                if self.reward_case == 's3d':
                    frames = self.preprocess_metaworld(self.past_observations)
                    video = torch.from_numpy(frames)
                    video_output = self.net(video.float())
                    video_embedding = video_output['video_embedding']
                    similarity_matrix = torch.matmul(self.target_embedding, video_embedding.t())
                    s3d_reward = similarity_matrix.detach().numpy()[0][0]
                    reward = reward + 0.01*s3d_reward
                    info['s3d_reward'] = s3d_reward
                if self.reward_case == 'vae' or self.reward_case == 'vaefb':
                    frames = self.preprocess_metaworld_crop_only(self.past_observations)
                    # breakpoint()
                    video_tensor = self.vae.preprocess_video_for_vae(frames, (480,480), 16)
                    # breakpoint()
                    latents = self.vae.encode_video(video_tensor)
                    similarity = self.vae.compute_vae_similarity(
                    self.target_embedding, 
                    latents, 
                    method='cosine_flatten'
                )
                    
                    instrinsic_reward = self.alpha * similarity
                    reward = reward +  instrinsic_reward
                    
                    info['vae_reward'] = instrinsic_reward
                    # breakpoint()
                if (self.reward_case == 'fb' or self.reward_case == 'vaefb') and self.use_fb_reward:
                    
                    frames = self.preprocess_metaworld(self.past_observations)
                    frames = frames.transpose(0,2,3,4,1)[0] # T,C,H,W
                    target_obs = torch.tensor(self.last_frame.transpose(2,0,1)).unsqueeze(0)
                    current_obs = torch.tensor(self.past_observations[-1].transpose(2,0,1)).unsqueeze(0)
                    fb_reward = self.compute_fb_reward_from_img_pair(current_obs, action, target_obs)
                    
                    # breakpoint()
                    '''
                    if hasattr(self, 'use_vae_fb') and self.use_vae_fb:
                        print('Using VAE features to compute FB reward')
                        # breakpoint()
                        current_frame = self.past_observations[-1]
                        target_frame = self.last_frame
                        
                        current_vae_features = self.extract_vae_features_from_single_frame(current_frame)
                        target_vae_features = self.extract_vae_features_from_single_frame(target_frame)
                        
                        vae_fb_reward = self.compute_fb_reward_from_vae_features(current_vae_features, action, target_vae_features)
                        info['vae_fb_reward'] = vae_fb_reward
                        fb_reward = vae_fb_reward
                    '''
                    
                    
                    if self.fb_reward_scaled and 100 <= fb_reward <= 1000:
                        self.beta_scale = 0.00001
                        self.fb_reward_scaled = False
                        # print('hhhhhhhhhh')
                        # breakpoint()
                    
                    # breakpoint()
                    scale_fb_reward = self.beta_scale * fb_reward
                    info['fb_reward'] = scale_fb_reward
                    reward += scale_fb_reward
                    
                    
                if self.reward_case == 'vaefb':
                    if self.use_fb_reward:
                        info['vaefb_reward'] = scale_fb_reward + instrinsic_reward
                    else:
                        info['vaefb_reward'] = instrinsic_reward

               # breakpoint()
               # NOTE: Diffusion and VIPER reward computation moved outside counter%128 block

                # info['vaefb_reward'] = fb_reward + instrinsic_reward


                # self.past_observations=[]  # MOVED BELOW - don't clear yet!

                    # breakpoint()

            # Compute Diffusion Reward (moved outside counter%128 block, before clearing observations)
            if self.reward_case == 'diffusion':
                from rewards import RewardContext
                # Prepare context with current frame
                current_frame = self.past_observations[-1] if self.past_observations else None
                if current_frame is not None:
                    ctx = RewardContext(
                        frame=current_frame,
                        action=action,
                        global_step=self.counter
                    )
                    # Update Diffusion reward with the frame
                    ctx = self.diffusion_reward.on_step(ctx)

                    # Check if ready to compute reward
                    if self.diffusion_reward.ready(ctx):
                        diffusion_r, diffusion_info = self.diffusion_reward.compute(ctx)
                        reward += diffusion_r
                        info.update(diffusion_info)
                        # info['intrinsic_reward'] = self.alpha * diffusion_r

            # Compute VIPER Reward (moved outside counter%128 block)
            if self.reward_case == 'viper':
                from rewards import RewardContext
                # Prepare context with current frame
                current_frame = self.past_observations[-1] if self.past_observations else None
                if current_frame is not None:
                    ctx = RewardContext(
                        frame=current_frame,
                        action=action,
                        global_step=self.counter
                    )
                    # Update VIPER reward with the frame
                    ctx = self.viper_reward.on_step(ctx)

                    # Check if ready to compute reward
                    if self.viper_reward.ready(ctx):
                        viper_r, viper_info = self.viper_reward.compute(ctx)
                        reward += self.alpha * viper_r
                        info.update(viper_info)
                        info['intrinsic_reward'] = self.alpha * viper_r
            if self.reward_case in ['tadpole', 'video-tadpole']:
                # time1 = time.time()
                noise_level_base = self.tadpole_config.get('noise_level_base')
                noise_level_range = self.tadpole_config.get('noise_level_range')
                context_len = self.tadpole_config.get('context_len') if self.reward_case == 'video-tadpole' else 1
                align_scale = self.tadpole_config.get('align_scale')
                recon_scale = self.tadpole_config.get('recon_scale')
                imgs = self.preprocess_metaworld(self.past_observations[-context_len:])[0].transpose(1,0,2,3)/255.0 # (T,C,H,W)
                latent = self.guidance.encode_imgs(torch.tensor(imgs).to(self.device).float())# (1, C, H, W) or (T, C, H, W)
                timestep = torch.randint(noise_level_base, noise_level_base + noise_level_range, [1], dtype=torch.long, device=latent.device)
                tadpole_reward = self.guidance.get_reward(latent, timestep, align_scale, recon_scale).float().cpu().numpy().item()
                reward = reward + tadpole_reward
                # time2 = time.time()
                # print("[TADPoLe] Reward computation time: {:.4f} seconds".format(time2 - time1))
                info.update({'tadpole_reward': tadpole_reward})
            # Clear observations after computing all rewards
            if self.counter % 128 == 0:
                self.past_observations = []

        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        # breakpoint()
        obs = {
            "reward": reward,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            ),
            "state": state,
            "is_first": False,
            "is_terminal": success
        }
        
        info['success'] = success
        # print('obs reward is',obs['reward']) 
        # assert 1==2
        return obs, reward, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            ),
            "state": state,
            "is_first": True,
            "is_terminal": False
        }
        return obs

class MetaWorld2:
    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera=None, gpu="cuda:0"):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )
        import os
        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        print(task)
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self._gpu_id = int(gpu.split(':')[1])
        self._count = 0

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        return spaces

    @property
    def action_space(self):
        action = self._env.action_space
        return action
    
    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        done = None
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action + np.random.randn(action.shape[0],) * 5)
            success += float(info["success"])
            reward += rew or 0.0
            if done:
                break
        success = min(success, 1.0)
        assert success in [0.0, 1.0]

        obs = {
            "reward": reward,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            ),
            "state": state,
        }
        info['success'] = success
        return obs, reward, done, info

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=self._gpu_id
            ),
            "state": state,
        }
        return obs


class RLBench:
    def __init__(
            self,
            name,
            size=(64, 64),
            action_repeat=1,
    ):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.tasks import ReachTarget

        # we only support reach_target in this codebase
        obs_config = ObservationConfig()
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)
        obs_config.wrist_camera.set_all(False)
        obs_config.front_camera.image_size = size
        obs_config.front_camera.depth = False
        obs_config.front_camera.point_cloud = False
        obs_config.front_camera.mask = False

        action_mode = partial(JointPosition, absolute_mode=False)

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=action_mode(), gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=True,
            shaped_rewards=True,
        )
        env.launch()

        if name == "reach_target":
            task = ReachTarget
        else:
            raise ValueError(name)
        self._env = env
        self._task = env.get_task(task)

        _, obs = self._task.reset()
        self._prev_obs = None

        self._size = size
        self._action_repeat = action_repeat

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def action_space(self):
        action = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_shape, dtype=np.float32
        )
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        try:
            reward = 0.0
            for i in range(self._action_repeat):
                obs, reward_, terminal = self._task.step(action["action"])
                success, _ = self._task._task.success()
                reward += reward_
                if terminal:
                    break
            self._prev_obs = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            success = False
            reward = 0.0
            obs = self._prev_obs

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": terminal,
            "is_terminal": terminal,
            "image": obs.front_rgb,
            "success": success,
        }
        return obs

    def reset(self):
        _, obs = self._task.reset()
        self._prev_obs = obs
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": obs.front_rgb,
            "success": False,
        }
        return obs


'''
 TimeLimit wrapper
episode
episodedone = True)
duration
'''
class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            # breakpoint()
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeActions:

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class TargetNormalizeActions:

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.action_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise AttributeError(name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.action_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class OneHotAction:

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.discrete = True
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class SelectAction:

    def __init__(self, env, key):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return self._env.step(action[self._key])


class Async:
    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy="thread"):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == "process":
            import multiprocessing as mp

            context = mp.get_context("spawn")
        elif strategy == "thread":
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn,))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access("obs_space")()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access("act_space")()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError("Lost connection to environment worker.")
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.
