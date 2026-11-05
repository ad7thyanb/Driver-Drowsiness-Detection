import cv2
import os
import numpy as np
from pygame import mixer
import time
import dlib

# Initialize pygame mixer for sound alerts
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')

# Load face and eye detection models
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Use dlib's face detector and shape predictor for more robust facial landmark detection
dlib_detector = dlib.get_frontal_face_detector()
try:
    # Try to load the shape predictor model file
    # If not available, we'll fall back to Haar cascade only
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    use_dlib = True
    print("Using dlib facial landmarks for enhanced detection")
except:
    use_dlib = False
    print("Dlib shape predictor not found. Using Haar cascade only.")
    # Fallback to Haar cascade for eyes
    left_eye_detector = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right_eye_detector = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# Initialize webcam with DirectShow backend (better for Windows)
print("Opening camera...")
try:
    # Try DirectShow backend first
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # If that fails, try the default backend
    if not capture.isOpened():
        print("DirectShow backend failed, trying default backend...")
        capture = cv2.VideoCapture(0)
    
    # Final check
    if not capture.isOpened():
        raise IOError("Cannot open webcam. Please check camera connections and permissions.")
    
    print("Camera opened successfully!")
except Exception as e:
    print(f"Error opening camera: {e}")
    raise

# Set better resolution if supported
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Constants
EYE_AR_THRESH = 0.2      # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 15  # Number of consecutive frames for alarm
EYE_DETECTION_COOLDOWN = 5  # Frames to wait before redetecting eyes

# Variables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
counter = 0
alarm_on = False
face_frame_count = 0
last_face = None

# For smoothing eye state predictions
left_eye_open_probs = [1.0] * 5
right_eye_open_probs = [1.0] * 5

def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio to detect blinks"""
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_landmarks(shape, left_start, left_end, right_start, right_end):
    """Extract eye landmarks from shape predictor"""
    left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(left_start, left_end)])
    right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(right_start, right_end)])
    return left_eye, right_eye

def detect_eye_state_advanced(eye_frame):
    """Detect eye state using multiple techniques"""
    if eye_frame.size == 0 or eye_frame.shape[0] < 2 or eye_frame.shape[1] < 2:
        return 0.0  # Return closed if invalid frame
    
    # Convert to grayscale
    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use thresholding to identify the sclera (white part of the eye)
    _, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    
    # Count white pixels (potential indicator of open eye)
    white_ratio = np.sum(threshold == 255) / threshold.size
    
    # Calculate edge density using Canny edge detector
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    
    # Calculate average intensity
    avg_intensity = np.mean(gray)
    
    # Combine features
    # Weight each feature based on its reliability
    open_score = (white_ratio * 0.5) + (edge_density * 0.3) + (min(1.0, avg_intensity/100) * 0.2)
    
    # Return probability that eye is open (0.0-1.0)
    return open_score

def draw_eye(frame, eye_points, is_open, color=None):
    """Draw eye contour and state"""
    if color is None:
        color = (0, 255, 0) if is_open else (0, 0, 255)
    
    # Draw eye contour
    hull = cv2.convexHull(eye_points)
    cv2.drawContours(frame, [hull], -1, color, 1)

def smooth_prediction(new_pred, history):
    """Apply temporal smoothing to predictions to reduce flickering"""
    # Add new prediction and remove oldest
    history.pop(0)
    history.append(new_pred)
    
    # Return average of recent predictions
    return sum(history) / len(history)

# Process video feed
try:
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip horizontally for more intuitive mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create a copy for drawing
        display = frame.copy()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast
        
        # Face detection cooldown to improve performance
        face_frame_count += 1
        faces = []
        
        # Only detect faces every few frames
        if face_frame_count >= 3 or last_face is None:
            face_frame_count = 0
            
            # Try to detect faces using dlib first (more accurate but slower)
            dlib_faces = dlib_detector(gray, 0)
            
            if len(dlib_faces) > 0:
                # Convert dlib rectangle to OpenCV format (x,y,w,h)
                for face in dlib_faces:
                    x, y = face.left(), face.top()
                    w, h = face.width(), face.height()
                    faces.append((x, y, w, h))
            else:
                # Fall back to Haar cascade
                faces = face_detector.detectMultiScale(gray, 
                                                       scaleFactor=1.1,
                                                       minNeighbors=5,
                                                       minSize=(30, 30),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
            
            if len(faces) > 0:
                # Keep track of the largest face for tracking
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                last_face = faces[0]
        
        # If we have a face from previous detection, use it
        if last_face is not None:
            faces = [last_face]
        
        # Initialize states
        left_eye_open = False
        right_eye_open = False
        left_eye_prob = 0.0
        right_eye_prob = 0.0
        
        # Process faces
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(display, (x, y), (x+w, y+h), (100, 150, 200), 2)
            
            # Extract face ROI
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]
            
            # Process eyes based on available methods
            if use_dlib:
                # Convert OpenCV rectangle to dlib rectangle
                dlib_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
                
                # Get facial landmarks
                shape = predictor(gray, dlib_rect)
                
                # Get eye landmarks (36-41 for left eye, 42-47 for right eye)
                left_eye, right_eye = get_eye_landmarks(shape, 36, 42, 42, 48)
                
                # Calculate eye aspect ratios
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                
                # Determine if eyes are open based on aspect ratio
                left_eye_prob = max(0, min(1, (left_ear - 0.15) / 0.15))
                right_eye_prob = max(0, min(1, (right_ear - 0.15) / 0.15))
                
                # Draw eyes
                draw_eye(display, left_eye, left_eye_prob > 0.5)
                draw_eye(display, right_eye, right_eye_prob > 0.5)
                
            else:
                # Use Haar cascade method for eyes
                # Define eye search regions to improve detection
                roi_gray = face_gray
                
                # Detect left and right eyes
                left_eye_region = roi_gray[int(h/5):int(h/2), int(w/5):int(w/2)]
                right_eye_region = roi_gray[int(h/5):int(h/2), int(w/2):int(4*w/5)]
                
                # Detect eyes in the defined regions
                left_eyes = left_eye_detector.detectMultiScale(left_eye_region, 
                                                            scaleFactor=1.05,
                                                            minNeighbors=4,
                                                            minSize=(int(w/12), int(h/12)))
                
                right_eyes = right_eye_detector.detectMultiScale(right_eye_region, 
                                                              scaleFactor=1.05,
                                                              minNeighbors=4,
                                                              minSize=(int(w/12), int(h/12)))
                
                # Process detected eyes
                for (ex, ey, ew, eh) in left_eyes:
                    # Adjust coordinates to face coordinates
                    ex += int(w/5)
                    ey += int(h/5)
                    
                    # Draw eye rectangle
                    cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 1)
                    
                    # Extract eye frame
                    eye_frame = face_color[ey:ey+eh, ex:ex+ew]
                    
                    # Determine eye state
                    left_eye_prob = detect_eye_state_advanced(eye_frame)
                    break
                
                for (ex, ey, ew, eh) in right_eyes:
                    # Adjust coordinates to face coordinates
                    ex += int(w/2)
                    ey += int(h/5)
                    
                    # Draw eye rectangle
                    cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 1)
                    
                    # Extract eye frame
                    eye_frame = face_color[ey:ey+eh, ex:ex+ew]
                    
                    # Determine eye state
                    right_eye_prob = detect_eye_state_advanced(eye_frame)
                    break
        
        # Apply temporal smoothing to reduce flickering
        left_eye_prob = smooth_prediction(left_eye_prob, left_eye_open_probs)
        right_eye_prob = smooth_prediction(right_eye_prob, right_eye_open_probs)
        
        # Convert probabilities to binary state
        left_eye_open = left_eye_prob > 0.5
        right_eye_open = right_eye_prob > 0.5
        
        # Update drowsiness counter
        if not left_eye_open and not right_eye_open:
            counter += 1
        else:
            counter = max(0, counter - 1)  # Decrease counter but not below 0
        
        # Draw status background
        cv2.rectangle(display, (0, height-70), (250, height), (0, 0, 0), thickness=cv2.FILLED)
        
        # Draw status information
        cv2.putText(display, f"Left: {int(left_eye_prob*100)}%", (10, height-50), font, 0.6, 
                    (0, 255, 0) if left_eye_open else (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Right: {int(right_eye_prob*100)}%", (10, height-30), font, 0.6, 
                    (0, 255, 0) if right_eye_open else (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Drowsy: {counter}/{EYE_AR_CONSEC_FRAMES}", (10, height-10), font, 0.6, 
                    (0, 255, 0) if counter < EYE_AR_CONSEC_FRAMES else (0, 0, 255), 1, cv2.LINE_AA)
        
        # Alert if drowsy for too long
        if counter >= EYE_AR_CONSEC_FRAMES:
            # Visual alert
            cv2.rectangle(display, (0, 0), (width, height), (0, 0, 255), 4)
            cv2.putText(display, "ALERT! DROWSINESS DETECTED", (int(width/6), int(height/2)), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Play alarm if not already playing
            if not alarm_on:
                alarm_on = True
                try:
                    alarm_sound.play()
                except Exception as e:
                    print(f"Error playing alarm: {e}")
        else:
            alarm_on = False
        
        # Show the frame
        cv2.imshow('Driver Drowsiness Detection', display)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Clean up
    capture.release()
    cv2.destroyAllWindows()
    print("Program terminated.")
