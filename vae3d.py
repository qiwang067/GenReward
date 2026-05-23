import torch
import torch.nn.functional as F
import numpy as np
from diffusers import CogVideoXImageToVideoPipeline
import os
import argparse
from PIL import Image, ImageSequence
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKLCogVideoX 
try:
    from kitchen_env_wrappers import readGif
except (ImportError, AttributeError):
    def readGif(filename):
        from PIL import Image, ImageSequence
        pilIm = Image.open(filename)
        frames = []
        for frame in ImageSequence.Iterator(pilIm):
            frame = frame.convert('RGB')
            frames.append(np.array(frame))
        return frames

try:
    import cv2
    USE_CV2 = True
except ImportError:
    print("⚠️  OpenCV not found, using PIL for image resizing")
    USE_CV2 = False

def read_video_frames(video_path):
    """
    Read frames from video file (supports MP4, AVI, MOV, GIF, etc.)
    
    Args:
        video_path: Path to video file
        
    Returns:
        List of frames as numpy arrays
    """
    file_ext = os.path.splitext(video_path)[1].lower()
    
    if file_ext == '.gif':
        # Use the existing GIF reader
        return readGif(video_path)
    
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Use OpenCV for video files
        if not USE_CV2:
            raise ImportError("OpenCV is required to read video files. Please install: pip install opencv-python")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames could be read from video: {video_path}")
        
        print(f"Successfully read {len(frames)} frames from {video_path}")
        return frames
    
    else:
        # Try PIL for other image formats
        try:
            return readGif(video_path)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: .gif, .mp4, .avi, .mov, .mkv, .webm") from e

def resize_frame(frame, target_size):
    """
    Resize frame using available methods
    
    Args:
        frame: numpy array or PIL Image
        target_size: (width, height) tuple
        
    Returns:
        Resized frame as numpy array
    """
    if USE_CV2:
        # Use OpenCV if available
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
    else:
        # Use PIL as fallback
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame.astype(np.uint8))
        # PIL resize expects (width, height), same as target_size
        resized_pil = frame.resize(target_size, Image.LANCZOS)
        return np.array(resized_pil)


class VAETester:
    def __init__(self, model_path="THUDM/CogVideoX-5b-I2V", device="cuda", dtype=torch.float16, lora_path=None, vae_save_path=None, vae_path=None):
        """
        Initialize CogVideoX VAE for testing
        
        Args:
            model_path: Path to pretrained model or HuggingFace model ID (full pipeline path)
            device: Device to load model on
            dtype: Data type for model
            lora_path: Optional path to LoRA weights (local or HF repo/subfolder)
        """
        self.device = device
        self.dtype = dtype
        
        print(f"Loading pipeline from: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")
        if vae_path:
            self.vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, torch_dtype=torch.float16 ).to(device)
            # breakpoint()
        else:
            print(f"Loading pipeline from: {model_path}")
            print(f"Device: {device}, Dtype: {dtype}")
            # Load the full pipeline
            self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                model_path, 
                torch_dtype=dtype
            ).to(device)
            
            print("✅ Pipeline loaded successfully!")
            
            # Load LoRA weights if provided
            if lora_path:
                print(f"Loading LoRA from: {lora_path}")
                if lora_path.count("/") == 2:
                    list_of_files = lora_path.split("/")
                    repo_id = "/".join(list_of_files[:2])
                    subfolder = list_of_files[2]
                    lora_path = snapshot_download(
                        repo_id=repo_id,
                        local_dir_use_symlinks=False,
                    )
                    lora_path = os.path.join(lora_path, subfolder)
                self.pipe.load_lora_weights(lora_path, adapter_name="cogvideox-lora")
                self.pipe.set_adapters(["cogvideox-lora"], [1.0])
                print("✅ LoRA weights loaded successfully!")
            
            # Extract the VAE from the pipeline
            self.vae = self.pipe.vae
        
            if vae_save_path:
                print(f"💾 Saving VAE (with LoRA) to: {vae_save_path}")
                os.makedirs(vae_save_path, exist_ok=True)
                self.vae.save_pretrained(vae_save_path)
                print(f"✅ VAE with LoRA saved to: {vae_save_path}")

            # Optional: Delete unnecessary components to save memory
            del self.pipe.transformer
            del self.pipe.text_encoder
            del self.pipe.scheduler
            torch.cuda.empty_cache()
            
        # Set to evaluation mode
        self.vae.eval()
        
        print("VAE model loaded successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in self.vae.parameters())
        print(f"Total parameters: {total_params:,}")
    
    def preprocess_video_for_vae(self, frames, target_size=(480, 480), target_frames=16):
        """
        Preprocess video frames for VAE input
        
        Args:
            frames: List of frames (PIL Images or numpy arrays)
            target_size: Target spatial resolution (H, W) - should be multiple of 8
            target_frames: Target number of frames - should be multiple of 4
            
        Returns:
            Preprocessed video tensor: [1, C, T, H, W] normalized to [-1, 1]
        """
        # print(f"Original frames: {len(frames)}")
        
        # Validation checks based on CogVideoX requirements
        if target_size[0] % 8 != 0 or target_size[1] % 8 != 0:
            print(f"⚠️  Warning: Target size {target_size} should be multiple of 8 for optimal compression.")
        if target_frames % 4 != 0:
            print(f"⚠️  Warning: Target frames {target_frames} should be multiple of 4 for temporal compression.")
        
        if len(frames) < target_frames:
            frames += [frames[-1]] * (target_frames - len(frames))
            print(f"Padded to {len(frames)} frames")
        elif len(frames) > target_frames:
            indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
            # print(f"Downsampled to {len(frames)} frames")
        
        processed_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            
            # Resize frame using available method
            frame_resized = resize_frame(frame, target_size)
            processed_frames.append(frame_resized)
        
        video_array = np.array(processed_frames)  # [T, H, W, C]
        video_array = video_array.astype(np.float32) / 255.0  # [0, 1]
        video_array = (video_array - 0.5) / 0.5  # [-1, 1] normalization for VAE
        
        video_array = video_array.transpose(3, 0, 1, 2)  # [C, T, H, W]
        video_tensor = torch.from_numpy(video_array).unsqueeze(0)  # [1, C, T, H, W]
        
        # print(f"Processed video shape: {video_tensor.shape}")
        # print(f"Value range: [{video_tensor.min().item():.3f}, {video_tensor.max().item():.3f}]")
        
        return video_tensor.to(self.device, dtype=self.dtype)
    
    def encode_video(self, video_tensor):
        """
        Encode video to latent space
        
        Args:
            video_tensor: Input video tensor [B, C, T, H, W]
            
        Returns:
            latents: Encoded latent representation
        """
        # print(f"Encoding video with shape: {video_tensor.shape}")
        
        with torch.no_grad():
            # Encode to latent space
            latent_dist = self.vae.encode(video_tensor).latent_dist
            latents = latent_dist.sample()
            
            # print(f"Latent shape: {latents.shape}")
            # print(f"Latent mean: {latents.mean().item():.4f}")
            # print(f"Latent std: {latents.std().item():.4f}")
            # print(f"Latent range: [{latents.min().item():.3f}, {latents.max().item():.3f}]")
            
            return latents
    
    def decode_latents(self, latents):
        """
        Decode latents back to video
        
        Args:
            latents: Latent representation
            
        Returns:
            reconstructed_video: Decoded video tensor
        """
        print(f"Decoding latents with shape: {latents.shape}")
        
        with torch.no_grad():
            # Decode from latent space
            reconstructed = self.vae.decode(latents).sample
            
            print(f"Reconstructed video shape: {reconstructed.shape}")
            print(f"Reconstructed value range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
            
            return reconstructed
    
    def compute_reconstruction_metrics(self, original, reconstructed):
        """
        Compute reconstruction quality metrics
        
        Args:
            original: Original video tensor
            reconstructed: Reconstructed video tensor
            
        Returns:
            metrics: Dictionary of metrics
        """
        with torch.no_grad():
            # Ensure same device and dtype
            original = original.to(reconstructed.device, dtype=reconstructed.dtype)
            
            # MSE Loss
            mse = F.mse_loss(original, reconstructed)
            
            # PSNR (Peak Signal-to-Noise Ratio)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Assuming [-1,1] range
            
            # MAE (Mean Absolute Error)
            mae = F.l1_loss(original, reconstructed)
            
            # Cosine Similarity (treating as flattened vectors)
            original_flat = original.flatten()
            reconstructed_flat = reconstructed.flatten()
            cosine_sim = F.cosine_similarity(
                original_flat.unsqueeze(0), 
                reconstructed_flat.unsqueeze(0),
                dim=1  # Fixed: use dim=1 for [1, N] shaped tensors
            )
            
            metrics = {
                'mse': mse.item(),
                'psnr': psnr.item(),
                'mae': mae.item(),
                'cosine_similarity': cosine_sim.item()
            }
            
            print("\nReconstruction Metrics:")
            print(f"MSE: {metrics['mse']:.6f}")
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            print(f"MAE: {metrics['mae']:.6f}")
            print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
            
            return metrics
    
    def test_video_reconstruction(self, video_path, target_size=(256, 256), target_frames=16):
        """
        Test complete encode-decode cycle for a video
        
        Args:
            video_path: Path to video file
            target_size: Target spatial resolution
            target_frames: Target number of frames
            
        Returns:
            Dictionary with results
        """
        print("\n" + "="*60)
        print("TESTING VIDEO RECONSTRUCTION")
        print("="*60)
        print(f"Video: {video_path}")
        
        try:
            # Read and preprocess video
            frames = readGif(video_path)
            video_tensor = self.preprocess_video_for_vae(frames, target_size, target_frames)
            
            # Encode
            latents = self.encode_video(video_tensor)
            
            # Decode
            reconstructed = self.decode_latents(latents)
            
            # Compute metrics
            metrics = self.compute_reconstruction_metrics(video_tensor, reconstructed)
            
            # Compression ratio
            original_size = video_tensor.numel() * video_tensor.element_size()
            latent_size = latents.numel() * latents.element_size()
            compression_ratio = original_size / latent_size
            
            print(f"\nCompression Analysis:")
            print(f"Original size: {original_size:,} bytes")
            print(f"Latent size: {latent_size:,} bytes")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            
            results = {
                'video_path': video_path,
                'original_shape': tuple(video_tensor.shape),
                'latent_shape': tuple(latents.shape),
                'reconstructed_shape': tuple(reconstructed.shape),
                'compression_ratio': compression_ratio,
                'metrics': metrics,
                'latents': latents,
                'original': video_tensor,
                'reconstructed': reconstructed
            }
            
            return results
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            raise
    
    def compare_video_latents(self, video_paths, target_size=(480, 480), target_frames=16, similarity_method='cosine_mean'):
        """
        Compare latent representations of multiple videos
        
        Args:
            video_paths: List of video file paths
            target_size: Target spatial resolution
            target_frames: Target number of frames
            similarity_method: Similarity computation method:
                - 'cosine_mean': Cosine similarity on spatial-temporal averaged features
                - 'cosine_flatten': Cosine similarity on flattened features  
                - 'dot_mean': Dot product on spatial-temporal averaged features
                - 'dot_flatten': Dot product on flattened features
            
        Returns:
            Comparison results
        """
        print("\n" + "="*60)
        print("COMPARING VIDEO LATENTS")
        print("="*60)
        print(f"Similarity method: {similarity_method}")
        
        all_latents = []
        valid_paths = []
        
        # Encode all videos
        for i, video_path in enumerate(video_paths):
            print(f"\n[{i+1}/{len(video_paths)}] Processing: {video_path}")
            
            if not os.path.exists(video_path):
                print(f"WARNING: Video file not found: {video_path}")
                continue
            
            try:
                frames = read_video_frames(video_path)  # Fixed: use read_video_frames instead of readGif
                
                # Check if frames were successfully read
                if not frames:
                    print(f"ERROR: No frames could be read from {video_path}")
                    continue
                    
                video_tensor = self.preprocess_video_for_vae(frames, target_size, target_frames)
                latents = self.encode_video(video_tensor)
                
                all_latents.append(latents)
                valid_paths.append(video_path)
                
            except Exception as e:
                print(f"ERROR processing {video_path}: {e}")
                continue
        
        if len(all_latents) < 2:
            print("Need at least 2 valid videos for comparison")
            return None
        
        # Compute pairwise similarities using the selected method
        print(f"\nComputing pairwise latent similarities with {similarity_method}...")
        n = len(all_latents)
        similarity_matrix = np.zeros((n, n))
        
        # Print header
        labels = [os.path.basename(path)[:12] for path in valid_paths]
        header = "    " + "".join(f"{label:>12}" for label in labels)
        print(header)
        
        for i in range(n):
            row_str = f"{labels[i]:>3} "
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                    row_str += f"{'1.0000':>12}"
                else:
                    # Use the selected similarity computation method
                    similarity = self.compute_vae_similarity(
                        all_latents[i], 
                        all_latents[j], 
                        method=similarity_method
                    )
                    similarity_matrix[i][j] = similarity
                    row_str += f"{similarity:>12.4f}"
            print(row_str)
        
        return {
            'video_paths': valid_paths,
            'latents': all_latents,
            'similarity_matrix': similarity_matrix,
            'similarity_method': similarity_method
        }
        
    def compute_vae_similarity(self, latents1, latents2, method='cosine_mean'):
        """
        Compute similarity between VAE latents with different methods
        
        Args:
            latents1: First latent tensor [B, C, T, H, W]
            latents2: Second latent tensor [B, C, T, H, W]  
            method: Similarity computation method:
                - 'cosine_mean': Cosine similarity on spatial-temporal averaged features (recommended)
                - 'cosine_flatten': Cosine similarity on flattened features
                - 'dot_mean': Dot product on spatial-temporal averaged features  
                - 'dot_flatten': Dot product on flattened features
            
        Returns:
            Similarity score (float)
        """
        
        # Parse method
        if method.startswith('cosine_'):
            use_cosine = True
            reduction = method.split('_')[1]  # 'mean' or 'flatten'
        elif method.startswith('dot_'):
            use_cosine = False
            reduction = method.split('_')[1]  # 'mean' or 'flatten'
        else:
            raise ValueError(f"Unknown method: {method}. Use 'cosine_mean', 'cosine_flatten', 'dot_mean', or 'dot_flatten'")
        
        # Prepare vectors based on reduction method
        if reduction == 'mean':
            # Average over spatial-temporal dimensions, keep channel dim
            # [B, C, T, H, W] -> [B, C] -> [C] (remove batch)
            vec1 = latents1.mean(dim=[2, 3, 4]).squeeze(0)  # [C]
            vec2 = latents2.mean(dim=[2, 3, 4]).squeeze(0)  # [C] 
        elif reduction == 'flatten':
            # Flatten everything
            vec1 = latents1.flatten()
            vec2 = latents2.flatten()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        # Compute similarity
        if use_cosine:
            # Cosine similarity (normalized dot product)
            similarity = F.cosine_similarity(
                vec1.unsqueeze(0), 
                vec2.unsqueeze(0), 
                dim=1
            ).item()
        else:
            # Dot product - fix infinity issue by normalizing for large vectors
            # Convert to float32 to avoid overflow
            vec1 = vec1.float()
            vec2 = vec2.float()
            
            # Normalize vectors to avoid overflow in dot product
            norm1 = vec1.norm(p=2)
            norm2 = vec2.norm(p=2)
            
            if norm1 > 0 and norm2 > 0:
                # Normalize to unit vectors (makes it equivalent to cosine but more stable)
                vec1_normalized = vec1 / norm1
                vec2_normalized = vec2 / norm2
                similarity = torch.dot(vec1_normalized, vec2_normalized).item()
                # Scale back by norms to preserve magnitude information
                similarity = similarity * norm1.item() * norm2.item() / (norm1.item() * norm2.item())
                # Actually, let's just keep the normalized version to avoid overflow
                similarity = torch.dot(vec1_normalized, vec2_normalized).item()
            else:
                similarity = 0.0
        
        return similarity
    
    def compare_all_similarity_methods(self, video_paths, target_size=(480, 480), target_frames=16):
        """
        Compare all similarity methods side by side for the same videos
        
        Args:
            video_paths: List of video file paths
            target_size: Target spatial resolution
            target_frames: Target number of frames
            
        Returns:
            Dictionary with results from all methods
        """
        print("\n" + "="*70)
        print("COMPARING ALL SIMILARITY METHODS")
        print("="*70)
        
        # Encode all videos once
        all_latents = []
        valid_paths = []
        
        for i, video_path in enumerate(video_paths):
            print(f"\n[{i+1}/{len(video_paths)}] Processing: {video_path}")
            
            if not os.path.exists(video_path):
                print(f"WARNING: Video file not found: {video_path}")
                continue
            
            try:
                frames = read_video_frames(video_path)
                video_tensor = self.preprocess_video_for_vae(frames, target_size, target_frames)
                latents = self.encode_video(video_tensor)
                
                all_latents.append(latents)
                valid_paths.append(video_path)
                
            except Exception as e:
                print(f"ERROR processing {video_path}: {e}")
                continue
        
        if len(all_latents) < 2:
            print("Need at least 2 valid videos for comparison")
            return None
        
        # Test all methods
        methods = ['cosine_mean', 'cosine_flatten', 'dot_mean', 'dot_flatten']
        all_results = {}
        
        for method in methods:
            print(f"\n📊 Method: {method}")
            print("-" * 50)
            
            n = len(all_latents)
            similarity_matrix = np.zeros((n, n))
            
            # Compute similarities
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        similarity = self.compute_vae_similarity(
                            all_latents[i], 
                            all_latents[j], 
                            method=method
                        )
                        similarity_matrix[i][j] = similarity
            
            # Print matrix
            labels = [os.path.basename(path)[:12] for path in valid_paths]
            header = "    " + "".join(f"{label:>12}" for label in labels)
            print(header)
            
            for i in range(n):
                row_str = f"{labels[i]:>3} "
                for j in range(n):
                    if i == j:
                        row_str += f"{'1.0000':>12}"
                    else:
                        row_str += f"{similarity_matrix[i][j]:>12.4f}"
                print(row_str)
            
            all_results[method] = {
                'similarity_matrix': similarity_matrix,
                'method': method
            }
        
        # Summary comparison with better handling of inf/large values
        print(f"\n📈 SUMMARY COMPARISON")
        print("=" * 70)
        print("Method".ljust(15) + "Range".ljust(25) + "Average".ljust(15) + "Description")
        print("-" * 70)
        
        for method in methods:
            matrix = all_results[method]['similarity_matrix']
            # Get off-diagonal elements (exclude diagonal 1.0s)
            off_diag = matrix[np.triu_indices_from(matrix, k=1)]
            
            if len(off_diag) > 0:
                # Handle inf values
                finite_vals = off_diag[np.isfinite(off_diag)]
                
                if len(finite_vals) > 0:
                    min_val = finite_vals.min()
                    max_val = finite_vals.max()
                    avg_val = finite_vals.mean()
                    
                    # Format range and average
                    if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                        range_str = f"[{min_val:>6.2e},{max_val:>6.2e}]"
                        avg_str = f"{avg_val:>12.2e}"
                    else:
                        range_str = f"[{min_val:>6.3f},{max_val:>6.3f}]"
                        avg_str = f"{avg_val:>12.4f}"
                else:
                    range_str = "[inf, inf]"
                    avg_str = "inf"
                
                # Description
                if method == 'cosine_mean':
                    desc = "Semantic similarity (recommended)"
                elif method == 'cosine_flatten':
                    desc = "Pixel-level similarity"
                elif method == 'dot_mean':
                    desc = "Semantic similarity (S3D-like, normalized)"
                else:
                    desc = "Pixel-level similarity (S3D-like, normalized)"
                
                print(f"{method:<15}{range_str:<25}{avg_str:<15}    {desc}")
        
        all_results['video_paths'] = valid_paths
        all_results['latents'] = all_latents
        
        return all_results
    
    def save_reconstruction_frames(self, original, reconstructed, output_dir, prefix=""):
        """
        Save original and reconstructed frames for visual comparison
        
        Args:
            original: Original video tensor [1, C, T, H, W]
            reconstructed: Reconstructed video tensor [1, C, T, H, W]
            output_dir: Directory to save frames
            prefix: Prefix for saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert from [-1, 1] to [0, 255]
        def tensor_to_frames(tensor):
            tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
            tensor = tensor.squeeze(0)  # Remove batch dimension: [C, T, H, W]
            tensor = tensor.permute(1, 2, 3, 0)  # [T, H, W, C]
            frames = (tensor.cpu().numpy() * 255).astype(np.uint8)
            return frames
        
        original_frames = tensor_to_frames(original)
        reconstructed_frames = tensor_to_frames(reconstructed)
        
        # Save frames
        for t in range(original_frames.shape[0]):
            orig_frame = Image.fromarray(original_frames[t])
            recon_frame = Image.fromarray(reconstructed_frames[t])
            
            orig_frame.save(os.path.join(output_dir, f"{prefix}original_frame_{t:03d}.png"))
            recon_frame.save(os.path.join(output_dir, f"{prefix}reconstructed_frame_{t:03d}.png"))
        
        print(f"Saved frames to: {output_dir}")


def get_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='CogVideoX VAE Testing Tool')
    
    # Main mode selection
    parser.add_argument('--mode', type=str, choices=['encode', 'reconstruct', 'compare', 'compare-all'], 
                       default='reconstruct', help='Testing mode')
    
    # Video-related arguments  
    parser.add_argument('--video', type=str, nargs='+',
                       default=["./gifs/human_opening_door.gif"],
                       help='Video file path(s) to test (supports .mp4, .avi, .mov, .gif, etc.)')
    
    # Model and processing parameters
    parser.add_argument('--model-path', type=str, 
                       default="/code/cogkit/CogKit/models/CogVideoX1.5-5B-I2V/vae",
                       help='Path to VAE model')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'],
                       help='Data type for model (float32 recommended to avoid overflow)')
    
    parser.add_argument('--target-size', type=int, nargs=2, default=[480, 480],
                       help='Target spatial resolution [H, W] (should be multiple of 8)')
    
    parser.add_argument('--target-frames', type=int, default=16,
                       help='Target number of frames (should be multiple of 4)')
    
    parser.add_argument('--similarity-method', type=str, default='cosine_mean', 
                       choices=['cosine_mean', 'cosine_flatten', 'dot_mean', 'dot_flatten'],
                       help='Similarity computation method:\n'
                            'cosine_mean: Cosine similarity on channel-averaged features (recommended)\n'
                            'cosine_flatten: Cosine similarity on all pixels\n'
                            'dot_mean: Dot product on channel-averaged features (like S3D)\n'
                            'dot_flatten: Dot product on all pixels')
    
    parser.add_argument('--save-frames', action='store_true', default=False,
                       help='Save reconstruction frames for visual inspection')
    
    parser.add_argument('--output-dir', type=str, default='./vae_output',
                       help='Directory to save output frames')
    parser.add_argument('--vae_save_path', type=str, help='Save VAE separately to this path')
    parser.add_argument('--vae_path', type=str, default=None, help='Saved VAE')

    # In get_args():
    parser.add_argument('--lora-path', type=str, default=None, help='Path to LoRA weights (local or HF repo/subfolder)')
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    if args.dtype == 'float16':
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32
    
    return args


def main():
    """
    Main function with command line argument support
    """
    args = get_args()
    
    print("CogVideoX VAE Testing Tool")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Target size: {args.target_size}")
    print(f"Target frames: {args.target_frames}")
    
    # Initialize tester
    try:
        tester = VAETester(args.model_path, args.device, args.dtype, lora_path=args.lora_path, vae_save_path=args.vae_save_path, vae_path=args.vae_path)
    except Exception as e:
        print(f"Error initializing VAE: {e}")
        return
    
    # Execute based on mode
    if args.mode == 'encode':
        print(f"Video inputs: {args.video}")
        
        for video_path in args.video:
            if os.path.exists(video_path):
                try:
                    frames = read_video_frames(video_path)
                    video_tensor = tester.preprocess_video_for_vae(
                        frames, tuple(args.target_size), args.target_frames
                    )
                    latents = tester.encode_video(video_tensor)
                    print(f"Encoding completed for: {video_path}")
                except Exception as e:
                    print(f"Error encoding {video_path}: {e}")
            else:
                print(f"Video file not found: {video_path}")
    
    elif args.mode == 'reconstruct':
        print(f"Video inputs: {args.video}")
        
        for video_path in args.video:
            if os.path.exists(video_path):
                try:
                    results = tester.test_video_reconstruction(
                        video_path, tuple(args.target_size), args.target_frames
                    )
                    
                    if args.save_frames:
                        filename = os.path.splitext(os.path.basename(video_path))[0]
                        tester.save_reconstruction_frames(
                            results['original'], 
                            results['reconstructed'],
                            args.output_dir,
                            f"{filename}_"
                        )
                    
                    print(f"Reconstruction test completed for: {video_path}")
                except Exception as e:
                    print(f"Error testing {video_path}: {e}")
            else:
                print(f"Video file not found: {video_path}")
    
    elif args.mode == 'compare':
        print(f"Comparing videos: {args.video}")
        
        try:
            results = tester.compare_video_latents(
                args.video, tuple(args.target_size), args.target_frames, args.similarity_method
            )
            
            if results:
                print(f"Comparison completed for {len(results['video_paths'])} videos using {args.similarity_method}")
            
        except Exception as e:
            print(f"Error comparing videos: {e}")
    
    elif args.mode == 'compare-all':
        print(f"Comparing videos with all similarity methods: {args.video}")
        
        try:
            results = tester.compare_all_similarity_methods(
                args.video, tuple(args.target_size), args.target_frames
            )
            
            if results:
                print(f"All-method comparison completed for {len(results['video_paths'])} videos")
            
        except Exception as e:
            print(f"Error comparing videos: {e}")
    
    print("\nTesting completed!")


if __name__ == "__main__":
    main()