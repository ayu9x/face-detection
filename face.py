import cv2
import numpy as np
import argparse
import os
from datetime import datetime

class FaceDetector:
    def __init__(self, cascade_path=None, age_model_path=None, age_proto_path=None):
        """Initialize the face detector with cascade classifier and age prediction model."""
        if cascade_path is None:
            # Use the default OpenCV haarcascade file
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Error loading cascade classifier from {cascade_path}")
        
        # Set up age prediction model if paths are provided
        self.age_net = None
        self.has_age_model = False
        
        if age_model_path and age_proto_path and os.path.exists(age_model_path) and os.path.exists(age_proto_path):
            self.age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
            self.has_age_model = True
            self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Initialize video capture object to None
        self.cap = None
    
    def predict_age(self, face_img):
        """Predict age from face image."""
        if not self.has_age_model or self.age_net is None:
            return None
        
        # Preprocess face for age prediction
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Set input and get prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age_idx = np.argmax(age_preds)
        
        return self.age_list[age_idx]
        
    def detect_faces_image(self, image_path, scale_factor=1.1, min_neighbors=5, min_size=(30, 30), predict_age=False):
        """Detect faces in a single image and return the processed image."""
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Draw rectangles around faces and predict age
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if predict_age and self.has_age_model:
                # Extract face ROI for age prediction
                face_roi = img[y:y+h, x:x+w].copy()
                age = self.predict_age(face_roi)
                
                if age:
                    # Display age prediction
                    cv2.putText(img, f"Age: {age}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return img, faces
    
    def start_video_detection(self, camera_id=0, scale_factor=1.1, min_neighbors=5, min_size=(30, 30), predict_age=False):
        """Start real-time face detection using webcam."""
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
            
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            
            # Draw rectangles around faces and predict age
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if predict_age and self.has_age_model:
                    # Extract face ROI for age prediction
                    face_roi = frame[y:y+h, x:x+w].copy()
                    age = self.predict_age(face_roi)
                    
                    if age:
                        # Display age prediction
                        cv2.putText(frame, f"Age: {age}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Display the face count
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the result
            cv2.imshow('Face Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        self.stop_video_detection()
    
    def stop_video_detection(self):
        """Stop video detection and release resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
    
    def save_detected_image(self, image, output_path=None):
        """Save the image with detected faces."""
        if output_path is None:
            # Create output directory if it doesn't exist
            os.makedirs('detected_faces', exist_ok=True)
            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"detected_faces/detected_{timestamp}.jpg"
            
        # Save the image
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")
        return output_path

def download_age_model():
    """Download age prediction models if they don't exist."""
    import gdown
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Paths for age model files
    age_model_path = 'models/age_net.caffemodel'
    age_proto_path = 'models/age_deploy.prototxt'
    
    # Download if not exists
    if not os.path.exists(age_model_path):
        print("Downloading age prediction model...")
        # This is the Google Drive ID for the age prediction model
        gdown.download('https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW', age_model_path, quiet=False)
    
    if not os.path.exists(age_proto_path):
        print("Downloading age prediction prototxt...")
        # This is the Google Drive ID for the prototxt file
        gdown.download('https://drive.google.com/uc?id=1kWv0AjScEU5uwQ5R5OV7Ctv4qIYT4zve', age_proto_path, quiet=False)
        
    return age_model_path, age_proto_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Detection using OpenCV')
    parser.add_argument('--image', help='Path to input image file')
    parser.add_argument('--video', action='store_true', help='Enable video mode (webcam)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--cascade', help='Path to Haar cascade XML file')
    parser.add_argument('--scale-factor', type=float, default=1.1, 
                      help='Scale factor for detection (default: 1.1)')
    parser.add_argument('--min-neighbors', type=int, default=5,
                      help='Min neighbors for detection (default: 5)')
    parser.add_argument('--output', help='Path to output image file (image mode only)')
    parser.add_argument('--predict-age', action='store_true', help='Enable age prediction')
    args = parser.parse_args()
    
    # Download age prediction model if needed and age prediction is enabled
    age_model_path = None
    age_proto_path = None
    
    if args.predict_age:
        try:
            age_model_path, age_proto_path = download_age_model()
            print("Age prediction models ready.")
        except ImportError:
            print("Warning: gdown package not found. Please install it using:")
            print("pip install gdown")
            print("Age prediction will be disabled.")
        except Exception as e:
            print(f"Warning: Failed to download age prediction models: {e}")
            print("Age prediction will be disabled.")
    
    # Create face detector
    detector = FaceDetector(args.cascade, age_model_path, age_proto_path)
    
    if args.image:
        # Image mode
        result_img, faces = detector.detect_faces_image(
            args.image, 
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            predict_age=args.predict_age
        )
        
        if result_img is not None:
            print(f"Detected {len(faces)} faces")
            
            # Save or display the result
            if args.output:
                detector.save_detected_image(result_img, args.output)
            else:
                detector.save_detected_image(result_img)
                
            # Display the image
            cv2.imshow('Detected Faces', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.video:
        # Video mode
        print("Starting video detection (press 'q' to quit)...")
        detector.start_video_detection(
            camera_id=args.camera,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            predict_age=args.predict_age
        )
    
    else:
        print("Error: Please specify either --image or --video")
        parser.print_help()

if __name__ == "__main__":
    main()