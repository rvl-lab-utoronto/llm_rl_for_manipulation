import numpy as np
import cv2
from pathlib import Path
import os
import subprocess
import tempfile
import shutil

def save_numpy_array_as_video(array, output_path, fps=30):
    """
    Save a sequence of images as a video file by first saving individual frames
    and then potentially using ffmpeg if available.
    
    Parameters:
    -----------
    array : numpy.ndarray or list
        Array of shape (T, n, n, 3) or list of arrays with shape (n, n, 3) where:
        - T is the number of frames (time dimension)
        - n is the height and width of each frame (assuming square frames)
        - 3 represents the RGB color channels
        
    output_path : str
        Path where the video will be saved
        
    fps : int, default=30
        Frames per second for the output video
    
    Returns:
    --------
    bool
        True if saving was successful, False otherwise
    """
    
    # Ensure array is uint8
    if isinstance(array, list):
        # Convert list to numpy array
        array = np.array(array)
    
    # Make sure we have uint8 data
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    
    # Create a temporary directory to store frames
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Saving frames to temporary directory: {frames_dir}")
    
    try:
        # Save each frame as a PNG image
        for i in range(array.shape[0]):
            data = array[i, :, :, :].copy()
            # Convert RGB to BGR (OpenCV uses BGR)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, data)
        
        print(f"Saved {array.shape[0]} frames as PNG images")
        
        # Try to find ffmpeg
        ffmpeg_available = False
        try:
            # Check if ffmpeg is available
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            ffmpeg_path = result.stdout.strip()
            ffmpeg_available = bool(ffmpeg_path)
        except:
            ffmpeg_available = False
            
        if ffmpeg_available:
            print(f"ffmpeg found at {ffmpeg_path}, using it to create video")
            frame_pattern = os.path.join(frames_dir, "frame_%04d.png")
            
            # Try to use ffmpeg to create the video
            cmd = [
                ffmpeg_path,
                '-y',  # Overwrite output file if it exists
                '-framerate', str(fps),
                '-i', frame_pattern,
                '-c:v', 'png',  # Use lossless codec
                '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
                output_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Successfully created video with ffmpeg: {output_path}")
                    return True
                else:
                    print(f"ffmpeg failed with error: {result.stderr}")
            except Exception as e:
                print(f"Error running ffmpeg: {e}")
        
        # If ffmpeg is not available or failed, fall back to a simple OpenCV solution
        print("Trying OpenCV VideoWriter...")
        
        # Try different codecs and file extensions
        codecs_and_extensions = [
            ('DIVX', '.avi'),
            ('MJPG', '.avi'),
            ('XVID', '.avi'),
            ('mp4v', '.mp4'),
            ('X264', '.mp4')
        ]
        
        success = False
        
        for codec, ext in codecs_and_extensions:
            # Create a new output path with the right extension
            base_path = os.path.splitext(output_path)[0]
            current_output = f"{base_path}{ext}"
            
            print(f"Trying codec {codec} with output {current_output}")
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            frame_size = (array.shape[2], array.shape[1])  # Width, Height
            
            out = cv2.VideoWriter(current_output, fourcc, fps, frame_size)
            
            if not out.isOpened():
                print(f"Could not open VideoWriter with codec {codec}")
                out.release()
                continue
            
            # Read back the saved frames and write to video
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                out.write(frame)
            
            out.release()
            
            # Check if the file was created successfully
            if os.path.exists(current_output) and os.path.getsize(current_output) > 0:
                print(f"Successfully created video with codec {codec}: {current_output}")
                
                # Copy to original output path if different
                if current_output != output_path:
                    shutil.copy2(current_output, output_path)
                    print(f"Copied to requested output path: {output_path}")
                
                success = True
                break
        
        if not success:
            print("All video codec attempts failed. Keeping the frames as images.")
            
            # Create a zip file of all frames as a fallback
            frames_zip = f"{os.path.splitext(output_path)[0]}_frames.zip"
            print(f"Creating zip archive of frames: {frames_zip}")
            
            shutil.make_archive(
                os.path.splitext(frames_zip)[0],  # Remove .zip as make_archive adds it
                'zip',
                frames_dir
            )
            
            print(f"Created zip archive: {frames_zip}")
            return False
            
        return success
            
    except Exception as e:
        print(f"Error during video creation: {e}")
        return False
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")
   
# Example usage
if __name__ == "__main__":
    # Create a simple example array (10 frames of a 64x64 image)
    T, n = 10, 64
    # Create a gradient animation
    example_array = np.zeros((T, n, n, 3), dtype=np.uint8)
    for t in range(T):
        # Create a different color gradient for each frame
        for i in range(n):
            for j in range(n):
                r = int(255 * i / n)
                g = int(255 * j / n)
                b = int(255 * t / T)
                example_array[t, i, j] = [r, g, b]
    
    # Save the example as a video
    output_file = "gradient_animation.mp4"
    success = save_numpy_array_as_video(example_array, output_file, fps=5)
    
    if success:
        print(f"Video saved successfully to {output_file}")
        print(f"Full path: {os.path.abspath(output_file)}")
    else:
        print(f"Video creation failed, check for a zip file of frames: {os.path.splitext(output_file)[0]}_frames.zip")