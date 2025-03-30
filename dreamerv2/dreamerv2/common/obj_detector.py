import cv2
import numpy as np
import os
import logging
from pathlib import Path
import tensorflow as tf

class PacmanDetector:
    """
    A simplified class for detecting PacMan in Atari game frames using template matching.
    Returns both the original frame and a binary mask with PacMan's location.
    Also handles resizing frames from detection size to model processing size.
    """
    
    def __init__(self, template_path=None, threshold=0.7, detection_size=(128, 128), process_size=(64, 64)):
        """
        Initialize the PacMan detector.
        
        Args:
            template_path: Path to the PacMan template image
            threshold: Threshold for template matching (0.0-1.0)
            detection_size: Size of frames used for detection (height, width)
            process_size: Size of frames after resizing for model processing (height, width)
        """
        self.template = None
        self.threshold = threshold
        self.detection_size = detection_size
        self.process_size = process_size
        self.logger = logging.getLogger('PacmanDetector')

        # Set up default template path if none provided
        if template_path is None:
            # Use a single default path relative to the module
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'pacman.png')
            self.logger.info(f"Using default template path: {template_path}")
        
        self.load_template(template_path)
    
    def load_template(self, template_path):
        """Load PacMan template from the given path."""
        print(f"PacmanDetector: Attempting to load template from {template_path}")
        
        if not template_path:
            print(f"PacmanDetector: Template path is empty")
            return False
            
        if not os.path.exists(template_path):
            print(f"PacmanDetector: Template file doesn't exist at {template_path}")
            print(f"PacmanDetector: Current working directory: {os.getcwd()}")
            print(f"PacmanDetector: Absolute template path: {os.path.abspath(template_path)}")
            return False
        
        print(f"PacmanDetector: Template file exists with size: {os.path.getsize(template_path)} bytes")
        print(f"PacmanDetector: File readable: {os.access(template_path, os.R_OK)}")
            
        self.template = cv2.imread(template_path, 0)
        if self.template is None:
            print(f"PacmanDetector: Failed to load template with cv2.imread")
            color_template = cv2.imread(template_path, 1)
            if color_template is None:
                print(f"PacmanDetector: Failed to load in color mode too")
                try:
                    with open(template_path, 'rb') as f:
                        content = f.read(20)
                    print(f"PacmanDetector: File header bytes: {content}")
                except Exception as e:
                    print(f"PacmanDetector: Error reading file: {str(e)}")
            else:
                print(f"PacmanDetector: Loaded in color mode: {color_template.shape}")
                self.template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)
                print(f"PacmanDetector: Converted to grayscale: {self.template.shape}")
        else:
            print(f"PacmanDetector: Template loaded successfully: shape={self.template.shape}, dtype={self.template.dtype}")
            
        self.logger.info(f"Loaded PacMan template from {template_path}")
        return self.template is not None
    
    @property
    def templates(self):
        """Return a list of templates for compatibility with the wrapper."""
        if self.template is not None:
            return [self.template]
        return []
    
    def process_frame(self, frame):
        """
        Process a frame to detect PacMan and generate a binary mask.
        
        Args:
            frame: A numpy array containing the game frame
            
        Returns:
            tuple: (original_frame, binary_mask)
                original_frame: The input frame unchanged
                binary_mask: A binary frame with 1s where PacMan is detected, 0s elsewhere
        """
        if self.template is None:
            self.logger.warning("No template loaded, cannot detect PacMan")
            # Return original frame and empty mask
            mask = np.zeros_like(frame, dtype=np.uint8)
            return frame, mask
        
        # Create a copy of the frame to avoid modifying the original
        frame_copy = frame.copy()
        
        # Convert frame to grayscale if needed
        if len(frame.shape) > 2 and frame.shape[2] >= 3:
            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame_copy
            
        # Create empty binary mask the same size as the frame
        binary_mask = np.zeros_like(gray_frame, dtype=np.uint8)
        
        # Perform template matching
        res = cv2.matchTemplate(gray_frame, self.template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.threshold)
        
        # Create binary mask with detected regions
        for pt in zip(*loc[::-1]):  # x, y coordinates
            w, h = self.template.shape[1], self.template.shape[0]
            x1, y1 = pt[0], pt[1]
            x2, y2 = x1 + w, y1 + h
            
            # Set the detected region to 1 in the binary mask
            binary_mask[y1:y2, x1:x2] = 1
        
        return frame, binary_mask
    
    def process_and_resize(self, frame):
        """
        Process a frame for detection at detection_size and then resize to process_size.
        
        Args:
            frame: A numpy array containing the game frame
            
        Returns:
            tuple: (detection_frame, model_frame, resized_mask, combined_output)
                detection_frame: Frame at detection_size 
                model_frame: Frame resized to process_size for model input
                resized_mask: A binary mask with PacMan detection resized to process_size
                combined_output: A combined frame+mask tensor with shape (H, W, 2) for model input
        """
        # Ensure the frame is at detection size
        frame_h, frame_w = frame.shape[:2] if len(frame.shape) > 2 else frame.shape
        if (frame_h, frame_w) != self.detection_size:
            self.logger.debug(f"Resizing input frame from {(frame_h, frame_w)} to {self.detection_size}")
            # Use cv2 for numpy arrays instead of tf
            frame = cv2.resize(frame, (self.detection_size[1], self.detection_size[0]))
        
        # Run detection on the properly sized frame
        detection_frame, binary_mask = self.process_frame(frame)
        
        # Resize for model processing using cv2 for numpy arrays
        model_frame = cv2.resize(
            detection_frame, 
            (self.process_size[1], self.process_size[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Also resize the binary mask if needed
        resized_mask = cv2.resize(
            binary_mask,
            (self.process_size[1], self.process_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create combined output (frame + mask) with shape (H, W, 2)
        # First normalize the model frame to 0-1 range if it's not already
        if model_frame.dtype == np.uint8:
            normalized_frame = model_frame.astype(np.float32) / 255.0
        else:
            normalized_frame = model_frame.astype(np.float32)
        
        # Convert mask to float32 for consistency
        mask_float = resized_mask.astype(np.float32)
        
        # Create combined output with shape (H, W, 2)
        if len(normalized_frame.shape) == 3:  # Color image case
            # Use just the first channel or grayscale conversion for consistent shape
            if normalized_frame.shape[2] >= 3:
                frame_gray = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2GRAY)
            else:
                frame_gray = normalized_frame[:,:,0]
            combined_output = np.stack([frame_gray, mask_float], axis=-1)
        else:  # Already grayscale
            combined_output = np.stack([normalized_frame, mask_float], axis=-1)
            
        return detection_frame, model_frame, resized_mask, combined_output
    
    def process_batch(self, frames):
        """
        Process a batch of frames, detecting PacMan and resizing for model input.
        
        Args:
            frames: Batch of frames as numpy array or tensor
            
        Returns:
            tuple: (detection_frames, model_frames, binary_masks, combined_outputs)
                combined_outputs: Tensor/array with shape (batch, H, W, 2) containing frame+mask
        """
        # Convert tensor to numpy if needed
        if tf.is_tensor(frames):
            frames_np = frames.numpy()
        else:
            frames_np = frames
            
        # Process each frame
        detection_frames = []
        model_frames = []
        binary_masks = []
        combined_outputs = []
        
        for frame in frames_np:
            det_frame, mod_frame, mask, combined = self.process_and_resize(frame)
            detection_frames.append(det_frame)
            model_frames.append(mod_frame)
            binary_masks.append(mask)
            combined_outputs.append(combined)
            
        # Convert back to numpy arrays
        detection_frames = np.array(detection_frames)
        model_frames = np.array(model_frames)
        binary_masks = np.array(binary_masks)
        combined_outputs = np.array(combined_outputs)
        
        # Convert back to tensors if input was a tensor
        if tf.is_tensor(frames):
            detection_frames = tf.convert_to_tensor(detection_frames)
            model_frames = tf.convert_to_tensor(model_frames)
            binary_masks = tf.convert_to_tensor(binary_masks)
            combined_outputs = tf.convert_to_tensor(combined_outputs)
            
        return detection_frames, model_frames, binary_masks, combined_outputs

# Simplified function to just get the combined output (frame+mask) for the model
def process_for_model(frames, template_path=None, threshold=0.7, detection_size=(128, 128), process_size=(64, 64)):
    """Process frames and return combined output (frame+mask) for model input."""
    detector = create_detector(template_path, threshold, detection_size, process_size)
    _, _, _, combined_outputs = detector.process_batch(frames)
    return combined_outputs

# Simple function to create a detector instance
def create_detector(template_path=None, threshold=0.7, detection_size=(128, 128), process_size=(64, 64)):
    """Create and return a PacmanDetector instance."""
    return PacmanDetector(template_path, threshold, detection_size, process_size)