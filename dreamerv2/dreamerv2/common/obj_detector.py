import cv2
import numpy as np
import os
import glob
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
            template_path: Path to a single template image or directory with multiple templates
            threshold: Threshold for template matching (0.0-1.0)
            detection_size: Size of frames used for detection (height, width)
            process_size: Size of frames after resizing for model processing (height, width)
        """
        self.templates = []  # Will store multiple templates
        self.threshold = threshold
        self.detection_size = detection_size
        self.process_size = process_size
        self.logger = logging.getLogger('PacmanDetector')

        # Set up default template path if none provided
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'pacman.png')
            self.logger.info(f"Using default template path: {template_path}")
        
        self.load_templates(template_path)
    
    def load_templates(self, template_path):
        """Load one or more templates for pacman detection."""
        print(f"PacmanDetector: Loading templates from {template_path}")
        
        if not template_path:
            print(f"PacmanDetector: Template path is empty")
            return False
        
        # Check if path is a directory
        if os.path.isdir(template_path):
            template_files = glob.glob(os.path.join(template_path, "*.png"))
            
            if not template_files:
                return False
                
            for file_path in template_files:
                self._load_single_template(file_path)
            
            return len(self.templates) > 0
        else:
            return self._load_single_template(template_path)
    
    def _load_single_template(self, file_path):
        """Helper method to load a single template file."""
        if not os.path.exists(file_path):
            return False
        
        template = cv2.imread(file_path, 0)  # Load in grayscale
        if template is None:
            color_template = cv2.imread(file_path, 1)  # Try loading in color
            if color_template is None:
                return False
            else:
                template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)
        
        self.templates.append(template)
        return True
    
    @property
    def template(self):
        """Return the first template for backward compatibility."""
        return self.templates[0] if self.templates else None
    
    def try_templates(self, frame):
        """
        Try detection with each template, finding ALL matches.
        
        Args:
            frame: A numpy array containing the game frame
            
        Returns:
            tuple: Same as process_and_resize but using all successful templates
        """
        if not self.templates:
            # Return empty results
            binary_mask = np.zeros_like(frame, dtype=np.uint8)
            if len(frame.shape) > 2:
                binary_mask = binary_mask[:,:,0]  # Ensure mask is 2D
            
            # Create empty combined output
            model_frame = cv2.resize(frame, (self.process_size[1], self.process_size[0]))
            resized_mask = np.zeros(self.process_size, dtype=np.uint8)
            
            if model_frame.dtype == np.uint8:
                normalized_frame = model_frame.astype(np.float32) / 255.0
            else:
                normalized_frame = model_frame.astype(np.float32)
                
            if len(normalized_frame.shape) == 3:
                frame_gray = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2GRAY)
            else:
                frame_gray = normalized_frame
                
            combined_output = np.stack([frame_gray, np.zeros_like(frame_gray)], axis=-1)
            return frame, model_frame, binary_mask, combined_output
        
        # Ensure the frame is at detection size
        frame_h, frame_w = frame.shape[:2] if len(frame.shape) > 2 else frame.shape
        if (frame_h, frame_w) != self.detection_size:
            frame = cv2.resize(frame, (self.detection_size[1], self.detection_size[0]))
        
        # Convert frame to grayscale if needed
        if len(frame.shape) > 2 and frame.shape[2] >= 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame.copy()
        
        # Create empty binary mask
        binary_mask = np.zeros_like(gray_frame, dtype=np.uint8)
        detection_found = False
        
        # Try each template and find ALL matches
        for template in self.templates:
            # Perform template matching
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            
            # Find ALL locations where the match exceeds the threshold
            locations = np.where(res >= self.threshold)
            
            # Process all matching locations
            for y, x in zip(locations[0], locations[1]):
                detection_found = True
                
                # Get template dimensions
                h, w = template.shape
                
                # Add this detection to the mask without resetting it
                binary_mask[y:y+h, x:x+w] = 1
        
        # DEBUG: Print detection statistics
        detection_pixels = np.sum(binary_mask)
        print(f"🔍 Detection stats: {detection_pixels} pixels detected as Pacman")
        if detection_pixels > 0:
            # Make sure the binary_mask is 2D for np.where()
            if len(binary_mask.shape) > 2:
                # If mask has more than 2 dimensions, use only the first channel
                flat_mask = binary_mask[:,:,0]
            else:
                flat_mask = binary_mask
                
            # Find positions where mask is 1 (detections)
            positions = np.where(flat_mask == 1)
            y_coords, x_coords = positions[0], positions[1]
            
            # Get unique coordinates (top-left corners of detections)
            unique_coords = []
            i = 0
            seen = set()
            while i < len(y_coords) and len(unique_coords) < 5:
                # Check for new regions by looking at larger gaps
                if i == 0 or (abs(y_coords[i] - y_coords[i-1]) > 5 or abs(x_coords[i] - x_coords[i-1]) > 5):
                    coord = (y_coords[i], x_coords[i])
                    if coord not in seen:
                        unique_coords.append(coord)
                        seen.add(coord)
                i += 1
            
            print("📌 Sample detection positions (potential objects):")
            for y, x in unique_coords[:5]:  # Show at most 5 positions
                print(f"  Position: ({x}, {y})")
        
        # Crop the mask below line 103 to avoid false detections in score bar
        if binary_mask.shape[0] > 103:
            binary_mask[103:, :] = 0
        
        # Resize for model processing
        model_frame = cv2.resize(
            frame, 
            (self.process_size[1], self.process_size[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Also resize the binary mask
        resized_mask = cv2.resize(
            binary_mask,
            (self.process_size[1], self.process_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create combined output (frame + mask)
        if model_frame.dtype == np.uint8:
            normalized_frame = model_frame.astype(np.float32) / 255.0
        else:
            normalized_frame = model_frame.astype(np.float32)
        
        mask_float = resized_mask.astype(np.float32)
        
        # Create combined output with shape (H, W, 2)
        if len(normalized_frame.shape) == 3:
            if normalized_frame.shape[2] >= 3:
                frame_gray = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2GRAY)
            else:
                frame_gray = normalized_frame[:,:,0]
            combined_output = np.stack([frame_gray, mask_float], axis=-1)
        else:
            combined_output = np.stack([normalized_frame, mask_float], axis=-1)
            
        return frame, model_frame, resized_mask, combined_output
    
    def process_frame(self, frame):
        """
        Process a frame to detect PacMan and generate a binary mask with ALL detections.
        
        Args:
            frame: A numpy array containing the game frame
            
        Returns:
            tuple: (original_frame, binary_mask)
        """
        if not self.templates:
            # Return original frame and empty mask
            mask = np.zeros_like(frame, dtype=np.uint8)
            if len(mask.shape) > 2 and mask.shape[2] >= 3:
                mask = mask[:,:,0]  # Use just one channel for the mask
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
        
        # Try each template and find ALL matches
        for template in self.templates:
            # Perform template matching
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            
            # Find ALL locations where the match exceeds the threshold
            locations = np.where(res >= self.threshold)
            
            # Process all matching locations
            for y, x in zip(locations[0], locations[1]):
                # Get template dimensions
                h, w = template.shape
                
                # Add this detection to the mask without resetting it
                binary_mask[y:y+h, x:x+w] = 1
        
        # Crop the mask below line 103 to avoid false detections in score bar
        if binary_mask.shape[0] > 103:
            binary_mask[103:, :] = 0
            
        return frame, binary_mask
    
    def process_and_resize(self, frame):
        """
        Process a frame for detection at detection_size and then resize to process_size.
        Uses all templates and returns all successful matches.
        """
        # For multiple templates, use the try_templates method 
        if len(self.templates) > 1:
            return self.try_templates(frame)
        
        # Otherwise use the original single template logic
        # Ensure the frame is at detection size
        frame_h, frame_w = frame.shape[:2] if len(frame.shape) > 2 else frame.shape
        if (frame_h, frame_w) != self.detection_size:
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
        if tf.is_tensor(frames):
            frames_np = frames.numpy()
        else:
            frames_np = frames
            
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
            
        detection_frames = np.array(detection_frames)
        model_frames = np.array(model_frames)
        binary_masks = np.array(binary_masks)
        combined_outputs = np.array(combined_outputs)
        
        if tf.is_tensor(frames):
            detection_frames = tf.convert_to_tensor(detection_frames)
            model_frames = tf.convert_to_tensor(model_frames)
            binary_masks = tf.convert_to_tensor(binary_masks)
            combined_outputs = tf.convert_to_tensor(combined_outputs)
            
        return detection_frames, model_frames, binary_masks, combined_outputs

def process_for_model(frames, template_path=None, threshold=0.7, detection_size=(128, 128), process_size=(64, 64)):
    """Process frames and return combined output (frame+mask) for model input."""
    detector = create_detector(template_path, threshold, detection_size, process_size)
    _, _, _, combined_outputs = detector.process_batch(frames)
    return combined_outputs

def create_detector(template_path=None, threshold=0.7, detection_size=(128, 128), process_size=(64, 64)):
    """Create and return a PacmanDetector instance."""
    return PacmanDetector(template_path, threshold, detection_size, process_size)