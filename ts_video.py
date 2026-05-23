from gym import Env, spaces
import numpy as np
# from stable_baselines3 import PPO
# from d4rl_alt.kitchen.kitchen_envs import KitchenMicrowaveHingeSlideV0, KitchenKettleV0, KitchenSlideV0, KitchenHingeV0, KitchenMicrowaveV0, KitchenLightV0
import torch as th
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
# from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
# from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image, ImageSequence
import torch as th
from s3dg import S3D
import numpy as np
from PIL import Image, ImageSequence
import cv2
import PIL
import os
# import seaborn as sns
import matplotlib.pylab as plt

from typing import Any, Dict

import gym
from gym.spaces import Box
import torch as th

def read_tesseract_video(video_path, asNumpy=True):
    """
    Read a video with opencv, return PIL Images or numpy arrays (H,W,3) if asNumpy.
    """
    cv2_video = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cv2_video.read()
            if not ret:
                break
            if asNumpy:
                frames.append(frame)
            else:
                frames.append(Image.fromarray(frame))
    finally:
        cv2_video.release()

    return frames[:-1]

if __name__ == "__main__":
    video_path = "/code/goal_dream/videos/seed1_val_0_pick_up_the_blue_fork_Trossen_WidowX_250_robot_arm_0.mp4"
    frames = read_tesseract_video(video_path, asNumpy=True)
    print(f"Read {len(frames)} frames from the video.")
    # You can visualize the first frame if needed
    if frames:
        plt.imshow(frames[0])
        plt.axis('off')
        plt.show()