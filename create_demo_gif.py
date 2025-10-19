import cv2
import numpy as np
from PIL import Image
import os

def video_to_gif(video_path, output_path, max_frames=100, fps=10, scale=0.5):
    """
    Convert video to GIF
    
    Args:
        video_path: Path to input video
        output_path: Path to output GIF
        max_frames: Maximum number of frames to extract
        fps: GIF frames per second
        scale: Scale factor for resizing (0.5 = half size)
    """
    print(f"Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}")
    print(f"Video FPS: {video_fps}")
    
    # Calculate frame skip to get max_frames
    frame_skip = max(1, total_frames // max_frames)
    print(f"Frame skip: {frame_skip}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if scale != 1.0:
            height, width = frame_rgb.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        
        frame_count += 1
        
        if len(frames) >= max_frames:
            break
    
    cap.release()
    
    print(f"Extracted {len(frames)} frames")
    print(f"Creating GIF...")
    
    # Create GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000/fps,  # milliseconds per frame
        loop=0  # infinite loop
    )
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"GIF created: {output_path}")
    print(f"File size: {file_size:.2f} MB")

if __name__ == "__main__":
    # Input video
    video_path = "data/output/tracked_output.mp4"
    
    # Output GIF
    output_path = "docs/demo.gif"
    
    # Create docs directory if it doesn't exist
    os.makedirs("docs", exist_ok=True)
    
    # Convert video to GIF
    # Parameters: max 100 frames, 10 fps, 50% scale
    video_to_gif(video_path, output_path, max_frames=100, fps=10, scale=0.5)

