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
        self.template_names = []  # Store template names for debugging
        self.threshold = threshold
        self.detection_size = detection_size
        self.process_size = process_size
        self.logger = logging.getLogger('PacmanDetector')
        self.debug_enabled = True  # Enable/disable detailed debug output

        # Set up default template path if none provided
        if template_path is None:
            # Use a single default path relative to the module
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
            print(f"PacmanDetector: Loading templates from directory: {template_path}")
            template_files = glob.glob(os.path.join(template_path, "*.png"))
            
            if not template_files:
                print(f"PacmanDetector: No template files found in directory")
                return False
                
            print(f"PacmanDetector: Found {len(template_files)} template files")
            
            # Load each template
            for file_path in template_files:
                success = self._load_single_template(file_path)
                if success:
                    print(f"PacmanDetector: Successfully loaded template: {os.path.basename(file_path)}")
                else:
                    print(f"PacmanDetector: Failed to load template: {os.path.basename(file_path)}")
            
            return len(self.templates) > 0
        else:
            # Single file path
            return self._load_single_template(template_path)
    
    def _load_single_template(self, file_path):
        """Helper method to load a single template file."""
        if not os.path.exists(file_path):
            print(f"PacmanDetector: Template file doesn't exist at {file_path}")
            return False
        
        template = cv2.imread(file_path, 0)  # Load in grayscale
        if template is None:
            print(f"PacmanDetector: Failed to load template with cv2.imread: {file_path}")
            color_template = cv2.imread(file_path, 1)  # Try loading in color
            if color_template is None:
                print(f"PacmanDetector: Failed to load in color mode too")
                return False
            else:
                print(f"PacmanDetector: Loaded in color mode: {color_template.shape}")
                template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)
        
        print(f"PacmanDetector: Template loaded: shape={template.shape}, dtype={template.dtype}")
        self.templates.append(template)
        self.template_names.append(os.path.basename(file_path))
        return True
    
    @property
    def template(self):
        """Return the first template for backward compatibility."""
        return self.templates[0] if self.templates else None
    
    def _generate_detection_debug_info(self, detected_template_name, binary_mask, best_confidence=None):
        """
        Generate structured debug information for frame processing.
        
        Args:
            detected_template_name: Name of the detected template, or None if none was detected
            binary_mask: The binary mask with 1s at detected positions
            best_confidence: The best confidence score (if no detection, otherwise None)
        
        Returns:
            str: Formatted debug string
        """
        # Count pixels with value 1
        pixels_count = np.sum(binary_mask) if binary_mask is not None else 0
        
        # Get coordinates of pixels with value 1
        coords_list = []
        if binary_mask is not None and pixels_count > 0:
            coords = np.where(binary_mask == 1)
            coords_list = list(zip(coords[0].tolist(), coords[1].tolist()))  # List of (y, x) coordinates
        
        # Limit the number of coordinates to avoid excessively large strings
        max_coords_to_show = 5
        coords_truncated = coords_list[:max_coords_to_show]
        coords_str = str(coords_truncated)
        if len(coords_list) > max_coords_to_show:
            coords_str = coords_str[:-1] + ", ...]"  # Replace the closing bracket with ", ...]"
        
        # Format the debug string
        debug_str = "$$Frame Procs$$\n"
        debug_str += f"Plantilla detectada: {detected_template_name}\n"
        debug_str += f"Pixels con 1's: Pasados al segundo canal\n"
        debug_str += f" - Cantidad (suma): {pixels_count}\n"
        debug_str += f" - Ubicacion (lista de coords): {coords_str}\n"
        debug_str += f"Mejor similitud no detectada: {best_confidence if detected_template_name is None else 'None'}\n"
        debug_str += "$$End$$"
        
        return debug_str
    
    def try_templates(self, frame):
        """
        Try detection with each template, stopping at the first match.
        
        Args:
            frame: A numpy array containing the game frame
            
        Returns:
            tuple: Same as process_and_resize but using the first successful template
        """
        if not self.templates:
            if self.debug_enabled:
                debug_info = self._generate_detection_debug_info(None, None, None)
                print(debug_info)
                
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
        best_confidence = 0
        detection_found = False
        detected_template_index = -1
        all_confidences = []
        
        # Try each template until a match is found
        templates_checked = 0
        for i, template in enumerate(self.templates):
            templates_checked += 1
            
            # Perform template matching
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_confidence, _, max_loc = cv2.minMaxLoc(res)
            all_confidences.append((i, max_confidence))
            
            if max_confidence >= self.threshold and max_confidence > best_confidence:
                detection_found = True
                best_confidence = max_confidence
                detected_template_index = i
                
                # Update the mask with this detection
                w, h = template.shape[1], template.shape[0]
                x, y = max_loc
                binary_mask = np.zeros_like(gray_frame, dtype=np.uint8)  # Reset mask
                binary_mask[y:y+h, x:x+w] = 1  # Create new mask for this detection
                
                # Stop searching once we find a match
                break
        
        # NEW: Crop the mask below line 103 to avoid false detections in score bar
        if binary_mask.shape[0] > 103:
            binary_mask[103:, :] = 0
        
        # Generate debug information
        if self.debug_enabled:
            if detection_found:
                detected_name = self.template_names[detected_template_index]
                debug_info = self._generate_detection_debug_info(detected_name, binary_mask)
            else:
                # Find the best non-detection confidence
                all_confidences.sort(key=lambda x: x[1], reverse=True)
                highest_conf = all_confidences[0][1] if all_confidences else 0
                debug_info = self._generate_detection_debug_info(None, binary_mask, highest_conf)
            
            print(debug_info)
        
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
        Process a frame to detect PacMan and generate a binary mask.
        
        Args:
            frame: A numpy array containing the game frame
            
        Returns:
            tuple: (original_frame, binary_mask)
        """
        if not self.templates:
            self.logger.warning("No templates loaded, cannot detect PacMan")
            
            # Generate debug information
            if self.debug_enabled:
                debug_info = self._generate_detection_debug_info(None, None, None)
                print(debug_info)
                
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
        best_confidence = 0
        detected_template_index = -1
        all_confidences = []
        
        # Try each template and use the best match
        for i, template in enumerate(self.templates):
            # Perform template matching
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_confidence, _, max_loc = cv2.minMaxLoc(res)
            all_confidences.append((i, max_confidence))
            
            if max_confidence >= self.threshold and max_confidence > best_confidence:
                best_confidence = max_confidence
                detected_template_index = i
                
                # Create mask with this detection
                w, h = template.shape[1], template.shape[0]
                x, y = max_loc
                binary_mask = np.zeros_like(gray_frame, dtype=np.uint8)  # Reset mask
                binary_mask[y:y+h, x:x+w] = 1  # Create new mask for this detection
                
                # Stop at the first successful match
                break
        
        # NEW: Crop the mask below line 103 to avoid false detections in score bar
        if binary_mask.shape[0] > 103:
            binary_mask[103:, :] = 0
            
        # Generate debug information
        if self.debug_enabled:
            if detected_template_index >= 0:
                detected_name = self.template_names[detected_template_index]
                debug_info = self._generate_detection_debug_info(detected_name, binary_mask)
            else:
                # Find the best non-detection confidence
                all_confidences.sort(key=lambda x: x[1], reverse=True)
                highest_conf = all_confidences[0][1] if all_confidences else 0
                debug_info = self._generate_detection_debug_info(None, binary_mask, highest_conf)
            
            print(debug_info)
            
        return frame, binary_mask
    
    def process_and_resize(self, frame):
        """
        Process a frame for detection at detection_size and then resize to process_size.
        Uses all templates and returns the first successful match.
        """
        # For multiple templates, use the try_templates method which implements
        # the logic of trying each template and stopping at the first match
        if len(self.templates) > 1:
            return self.try_templates(frame)
        
        # Otherwise use the original single template logic
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