import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dreamerv2.dreamerv2.common.obj_detector import PacmanDetector

def main():
    # Load test frame (replace with your own image path)
    test_frame_path = "sample_frame.png"  # Create or add your own test frame
    if not os.path.exists(test_frame_path):
        # Create a dummy test frame if none exists
        test_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        # Add a simple shape that might resemble pacman
        cv2.circle(test_frame, (64, 64), 20, (0, 255, 255), -1)
        cv2.imwrite(test_frame_path, test_frame)
        print(f"Created test frame at {test_frame_path}")
    
    # Load the test frame
    frame = cv2.imread(test_frame_path)
    if frame is None:
        print(f"Failed to load test frame from {test_frame_path}")
        return
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Path to templates directory
    templates_dir = os.path.join("dreamerv2", "dreamerv2", "templates")
    if not os.path.exists(templates_dir):
        print(f"Templates directory not found at {templates_dir}")
        templates_dir = None
    
    # Create detector with debug enabled
    detector = PacmanDetector(template_path=templates_dir, 
                              threshold=0.7, 
                              detection_size=(128, 128),
                              process_size=(64, 64))
    detector.debug_enabled = True
    
    # Process frame
    print("\n==== Processing Frame ====")
    _, _, mask, _ = detector.try_templates(frame_rgb)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title('Original Frame')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Detection Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("detection_debug_result.png")
    plt.show()
    
    print(f"Results saved to detection_debug_result.png")

if __name__ == "__main__":
    main()
