import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import cv2


def load_video_frames(video_path: str, max_frames: int = None) -> torch.Tensor:
    """
    Load frames from video file and convert to required format

    Args:
        video_path: Video file path
        max_frames: Maximum frame count limit (optional)

    Returns:
        Tensor with shape (1, T, 3, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # OpenCV reads in BGR format, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor and adjust dimension order (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame).float() / 255.0  # Normalize to 0-1
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
        frames.append(frame_tensor)
        
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")
    
    # Stack to (T, C, H, W) then add batch dimension
    video_tensor = torch.stack(frames).unsqueeze(0)  # (1, T, C, H, W)
    
    return video_tensor

def compute_frame_similarities(video_frames: torch.Tensor, 
                             text_prompts: List[str],
                             model_name: str = "ViT-B-32",
                             pretrained: str = "laion2b_s34b_b79k",
                             device: str = "cuda"):
    """
    Compute similarity scores between video frames and text prompts

    Args:
        frames: Video frame list, each frame is (H, W, C) numpy array
        text_prompts: Text prompt list
        model_name: OpenCLIP model name
        pretrained: Pretrained weight name
        device: Compute device

    Returns:
        Similarity score array with shape (T,)
    """
    # Initialize OpenCLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # Pre-compute text embeddings
    with torch.no_grad():
        text_embeddings = []
        for prompt in text_prompts:
            text_tokens = tokenizer([prompt]).to(device)
            text_embedding = model.encode_text(text_tokens)
            text_embedding = F.normalize(text_embedding, dim=-1)
            text_embeddings.append(text_embedding)
        
        # Average embeddings from multiple prompts
        text_embedding = torch.stack(text_embeddings).mean(dim=0)
        text_embedding = F.normalize(text_embedding, dim=-1)
    
    # Compute similarity for each frame
    similarities = []
    
    B, T, C, H, W = video_frames.shape
    frames = []
    video_frames = video_frames.view(B * T, C, H, W)
    for i in range(B * T):
        frame = video_frames[i]
        if frame.dtype == torch.float32:
            frame_np = (frame.clamp(0, 1) * 255).byte().cpu().numpy()
        else:
            frame_np = frame.cpu().numpy()
        frame_np = np.transpose(frame_np, (1, 2, 0))
        pil_image = Image.fromarray(frame_np)
        preprocessed = preprocess(pil_image)
        frames.append(preprocessed)
    frames_tensor = torch.stack(frames).to(device)
    with torch.no_grad():
        image_features = model.encode_image(frames_tensor)
        image_features = F.normalize(image_features, dim=-1)
        similarities = torch.matmul(image_features, text_embedding.T).squeeze(-1)
    similarities = similarities.view(B, T)
    max_similarity, max_idx = torch.max(similarities, dim=1)  # Get maximum over time dimension

    print(f"Maximum similarity index: {max_idx.item()}")
    print(f"Maximum similarity score: {max_similarity.item():.4f}")
    print(f"Maximum similarity score: {max_similarity.item()}")
    
    return max_idx.item()
    
    # breakpoint()


def get_max_similarity_frame(video_path: str, 
                           text_prompts: List[str],
                           model_name: str = "ViT-B-32",
                           pretrained: str = "laion2b_s34b_b79k", 
                           device: str = "cuda",
                           max_frames: Optional[int] = None) -> Tuple[int, float, np.ndarray]:
    """
    Find the frame with highest similarity to text prompts in the video
    Args:
        video_path: Video file path
        text_prompts: Text prompt list
        model_name: OpenCLIP model name
        pretrained: Pretrained weight name
        device: Compute device
        max_frames: Maximum frame count limit (optional)

    Returns:
        (max_idx, max_similarity, similarities)
        max_idx: Index of frame with highest similarity
        max_similarity: Highest similarity score
        similarities: Similarity score array for all frames
    """
    # Load video frames
    frames = load_video_frames(video_path, max_frames)

    # Compute similarity
    max_idx = compute_frame_similarities(
        frames, text_prompts, model_name, pretrained, device
    )
    
    return max_idx


def get_max_similarity_frame_wrapper(video_path: str, 
                                   text_prompt: str,
                                   device: str = "cuda") -> int:
    """
    Simplified version: Given video path and single text prompt, return index of frame with highest similarity

    Args:
        video_path: Video file path
        text_prompt: Text prompt
        device: Compute device

    Returns:
        Index of frame with highest similarity
    """
    text_prompts = [text_prompt]
    max_idx, _, _ = get_max_similarity_frame(video_path, text_prompts, device=device)
    return max_idx


# Usage example
if __name__ == "__main__":
    # Example usage
    # video_path = "/code/TesserAct-main/results/seed1_val_0_pick_apple_google_robot_0.mp4"
    video_path = "/code/goal_dream/videos/seed1_val_0_pick_up_the_blue_fork_Trossen_WidowX_250_robot_arm_0.mp4"
    text_prompts = [
        "robot arm grasping an object",
        "robotic gripper holding something", 
        "robot hand with object holded",
        "robot arm gripping item",
        "robot manipulator with grasped object"
        # "robot arm grasping an object",
        # "robotic gripper holding something",
        # "robot hand with object holded"
    ]
    
    try:
        max_idx, max_similarity, similarities = get_max_similarity_frame(
            video_path, text_prompts, device="cuda"
        )
        
        print(f"Highest similarity frame index: {max_idx}")
        print(f"Highest similarity score: {max_similarity:.4f}")
        print(f"Highest similarity score: {max_similarity}")
        print(f"Total frames: {len(similarities)}")

        # Simplified version example
        simple_idx = get_max_similarity_frame_wrapper(
            video_path, "robot arm grasping an object", device="cuda"
        )
        print(f"Simplified version result: {simple_idx}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please check if video path exists and CUDA is available")