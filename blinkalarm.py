import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any
import simpleaudio as sa # MacOS
import threading

FORWARD_TILT_ADJUSTMENT_MULTIPLIER = 1.0
BACKWARD_TILT_ADJUSTMENT_MULTIPLIER = 1.6

class BlinkDetector:
    def __init__(self, process_every_n_frames: int = 5, blink_threshold: float = 0.3, seconds_between_blinks: float = 2.0):
        """
        Initialize the blink detector.
        
        Args:
            process_every_n_frames: Process only one frame for every n frames
            blink_threshold: EAR (Eye Aspect Ratio) threshold below which eyes are considered closed
            seconds_between_blinks: The interval between blinks for reminder
        """
        self.process_every_n_frames = process_every_n_frames
        self.blink_threshold = blink_threshold
        self.seconds_between_blinks = seconds_between_blinks
        
        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        # Head pose reference points (nose, left and right temple)
        self.NOSE_INDEX = 4
        self.LEFT_TEMPLE_INDEX = 234
        self.RIGHT_TEMPLE_INDEX = 454
        
        # Head tilt reference points (forehead and chin)
        self.FOREHEAD_INDEX = 10  # Top of forehead
        self.CHIN_INDEX = 152     # Bottom of chin
        
        # Variable to track the last detection state
        self.last_detection_state: Optional[str] = None
        self.last_ear_value: Optional[float] = None
        
        # Add blink tracking variables
        self.blink_count: int = 0
        self.total_blinks: int = 0
        self.start_time: float = time.time()
        self.last_minute: int = 0
        self.minutes_elapsed: int = 0
        self.last_blink_time: float = time.time()
        
    def _calculate_ear(self, eye_landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) for a set of 3D eye landmarks.
        
        Args:
            eye_landmarks: A list of (x, y, z) coordinates for the eye landmarks
            
        Returns:
            The EAR value
        """
        # Calculate the vertical distances in 3D
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Calculate the horizontal distance in 3D
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        
        return ear
    
    def _calculate_eye_area(self, eye_landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Calculate the 2D projected area of the eye as a secondary metric.
        
        Args:
            eye_landmarks: A list of (x, y, z) coordinates for the eye landmarks
            
        Returns:
            The area of the eye contour
        """
        # Extract 2D points for area calculation
        points_2d = np.array([(x, y) for x, y, _ in eye_landmarks])
        # Calculate area using the shoelace formula
        area = 0.5 * abs(np.dot(points_2d[:, 0], np.roll(points_2d[:, 1], 1)) - 
                         np.dot(points_2d[:, 1], np.roll(points_2d[:, 0], 1)))
        return area
    
    def _normalize_by_face_size(self, landmarks_3d: List[Tuple[float, float, float]], 
                               face_landmarks: Any) -> List[Tuple[float, float, float]]:
        """
        Normalize landmarks by the face size to make them resilient to distance from camera.
        
        Args:
            landmarks_3d: List of 3D landmarks to normalize
            face_landmarks: Complete face landmarks from MediaPipe
            
        Returns:
            Normalized landmarks
        """
        # Get reference points for normalization
        nose = np.array([face_landmarks.landmark[self.NOSE_INDEX].x,
                         face_landmarks.landmark[self.NOSE_INDEX].y,
                         face_landmarks.landmark[self.NOSE_INDEX].z])
        
        left_temple = np.array([face_landmarks.landmark[self.LEFT_TEMPLE_INDEX].x,
                               face_landmarks.landmark[self.LEFT_TEMPLE_INDEX].y,
                               face_landmarks.landmark[self.LEFT_TEMPLE_INDEX].z])
        
        right_temple = np.array([face_landmarks.landmark[self.RIGHT_TEMPLE_INDEX].x,
                                face_landmarks.landmark[self.RIGHT_TEMPLE_INDEX].y,
                                face_landmarks.landmark[self.RIGHT_TEMPLE_INDEX].z])
        
        # Calculate face scale (distance between temples)
        face_scale = np.linalg.norm(left_temple - right_temple)
        
        # Normalize landmarks relative to nose and face scale
        normalized_landmarks = []
        for landmark in landmarks_3d:
            lm_array = np.array(landmark)
            normalized = (lm_array - nose) / face_scale if face_scale > 0 else lm_array
            normalized_landmarks.append(tuple(normalized))
            
        return normalized_landmarks
    
    def _calculate_head_tilt(self, face_landmarks: Any) -> float:
        """
        Calculate the head tilt (pitch) from face landmarks.
        
        Args:
            face_landmarks: Complete face landmarks from MediaPipe
            
        Returns:
            The head tilt angle in degrees (+ is forward tilt, - is backward tilt)
        """
        forehead = np.array([
            face_landmarks.landmark[self.FOREHEAD_INDEX].x,
            face_landmarks.landmark[self.FOREHEAD_INDEX].y,
            face_landmarks.landmark[self.FOREHEAD_INDEX].z
        ])
        
        chin = np.array([
            face_landmarks.landmark[self.CHIN_INDEX].x,
            face_landmarks.landmark[self.CHIN_INDEX].y,
            face_landmarks.landmark[self.CHIN_INDEX].z
        ])
        
        nose = np.array([
            face_landmarks.landmark[self.NOSE_INDEX].x,
            face_landmarks.landmark[self.NOSE_INDEX].y,
            face_landmarks.landmark[self.NOSE_INDEX].z
        ])
        
        # Calculate vectors
        forehead_to_nose = nose - forehead
        chin_to_nose = nose - chin
        
        # The z-component difference indicates head tilt
        # When head tilts forward, forehead_z becomes more negative compared to chin_z
        # When head tilts backward, forehead_z becomes more positive compared to chin_z
        z_diff = forehead_to_nose[2] - chin_to_nose[2]
        
        # Normalize by the vertical face size to make it scale-invariant
        face_height = np.linalg.norm(forehead - chin)
        normalized_tilt = z_diff / face_height if face_height > 0 else 0
        
        # Scale to a reasonable range (approximately -45 to +45 degrees)
        # This scaling factor can be adjusted based on observations
        return normalized_tilt * 90
    
    def _get_adjusted_threshold(self, head_tilt: float) -> float:
        """
        Get an adjusted EAR threshold based on head tilt.
        
        Args:
            head_tilt: Head tilt in degrees (+ is forward, - is backward)
            
        Returns:
            Adjusted threshold value
        """
        # Base threshold
        base_threshold = self.blink_threshold
        
        # Tilt adjustment factor (these can be tuned based on testing)
        forward_max_adjust = 0.1  # Maximum increase for forward tilt
        backward_max_adjust = 0.1  # Maximum decrease for backward tilt
        
        # Tilt range where adjustment is applied
        max_tilt_angle = 30.0  # Degrees
        
        # Calculate adjustment factor
        if head_tilt > 0:  # Forward tilt
            # Increase threshold for forward tilt (seeing more eye)
            adjustment = min(head_tilt / max_tilt_angle, 1.0) * forward_max_adjust
            return base_threshold + (adjustment * FORWARD_TILT_ADJUSTMENT_MULTIPLIER)
        else:  # Backward tilt
            # Decrease threshold for backward tilt (seeing less eye)
            adjustment = min(abs(head_tilt) / max_tilt_angle, 1.0) * backward_max_adjust
            return base_threshold - (adjustment * BACKWARD_TILT_ADJUSTMENT_MULTIPLIER)

    def get_blink_stats(self) -> Dict[str, float]:
        """
        Get the current blink statistics.
        
        Returns:
            Dictionary containing blink rate statistics
        """
        total_time = time.time() - self.start_time
        minutes = total_time / 60
        avg_rate = self.total_blinks / minutes if minutes > 0 else 0
        return {
            "total_blinks": self.total_blinks,
            "total_minutes": minutes,
            "average_bpm": avg_rate
        }

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], bool]:
        """
        Process a frame to detect if a person is blinking.
        
        Args:
            frame: The video frame to process
            
        Returns:
            The processed frame, detection state, and whether reminder sound should play
        """
        play_reminder = False
        current_time = time.time()
        time_since_blink = current_time - self.last_blink_time
        
        # Calculate seconds until reminder
        seconds_until_reminder = max(0, self.seconds_between_blinks - time_since_blink)
        
        # Check if we need to play reminder - only if face is detected
        if time_since_blink > self.seconds_between_blinks:
            play_reminder = True
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        detection_state = None
        avg_ear = 0.0
        
        if not results.multi_face_landmarks:
            detection_state = "No face detected"
            cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Reset blink count when face is not detected
            self.blink_count = 0
            # Update last blink time to prevent immediate sound when face is re-detected
            self.last_blink_time = current_time
            # Don't play reminder when no face is detected
            play_reminder = False
            return frame, detection_state, play_reminder
        
        for face_landmarks in results.multi_face_landmarks:
            # Get 3D landmark coordinates
            landmarks_3d = [(landmark.x, landmark.y, landmark.z) 
                           for landmark in face_landmarks.landmark]
            
            # Get 2D landmarks for visualization
            landmarks_2d = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) 
                           for landmark in face_landmarks.landmark]
            
            # Calculate head tilt
            head_tilt = self._calculate_head_tilt(face_landmarks)
            
            # Get adjusted blink threshold based on head tilt
            adjusted_threshold = self._get_adjusted_threshold(head_tilt)
            
            # Get eye landmarks (3D)
            left_eye_3d = [landmarks_3d[idx] for idx in self.LEFT_EYE_INDICES]
            right_eye_3d = [landmarks_3d[idx] for idx in self.RIGHT_EYE_INDICES]
            
            # Normalize landmarks by face size and pose
            normalized_left_eye = self._normalize_by_face_size(left_eye_3d, face_landmarks)
            normalized_right_eye = self._normalize_by_face_size(right_eye_3d, face_landmarks)
            
            # Calculate EAR using normalized 3D landmarks
            left_ear = self._calculate_ear(normalized_left_eye)
            right_ear = self._calculate_ear(normalized_right_eye)
            
            # Get eye landmarks (2D for drawing)
            left_eye_2d = [landmarks_2d[idx] for idx in self.LEFT_EYE_INDICES]
            right_eye_2d = [landmarks_2d[idx] for idx in self.RIGHT_EYE_INDICES]
            
            # Draw reference points used for normalization
            nose_point = landmarks_2d[self.NOSE_INDEX]
            left_temple = landmarks_2d[self.LEFT_TEMPLE_INDEX]
            right_temple = landmarks_2d[self.RIGHT_TEMPLE_INDEX]
            
            # Draw head tilt reference points
            forehead_point = landmarks_2d[self.FOREHEAD_INDEX]
            chin_point = landmarks_2d[self.CHIN_INDEX]
            
            # Draw reference triangle used for normalization
            cv2.circle(frame, nose_point, 3, (255, 0, 255), -1)  # Magenta for nose
            cv2.circle(frame, left_temple, 3, (255, 0, 255), -1)  # Magenta for left temple
            cv2.circle(frame, right_temple, 3, (255, 0, 255), -1)  # Magenta for right temple
            cv2.line(frame, nose_point, left_temple, (255, 0, 255), 1)  # Reference lines
            cv2.line(frame, nose_point, right_temple, (255, 0, 255), 1)
            cv2.line(frame, left_temple, right_temple, (255, 0, 255), 1)
            
            # Draw head tilt reference line
            cv2.circle(frame, forehead_point, 3, (255, 255, 0), -1)  # Yellow for forehead
            cv2.circle(frame, chin_point, 3, (255, 255, 0), -1)      # Yellow for chin
            cv2.line(frame, forehead_point, nose_point, (255, 255, 0), 1)  # Head tilt line
            cv2.line(frame, nose_point, chin_point, (255, 255, 0), 1)      # Head tilt line
            
            # Draw eye contours
            cv2.polylines(frame, [np.array(left_eye_2d)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_eye_2d)], True, (0, 255, 0), 1)
            
            # Draw EAR vertical and horizontal lines
            # Left eye EAR visualization
            cv2.line(frame, left_eye_2d[1], left_eye_2d[5], (0, 165, 255), 1)  # Orange vertical
            cv2.line(frame, left_eye_2d[2], left_eye_2d[4], (0, 165, 255), 1)  # Orange vertical
            cv2.line(frame, left_eye_2d[0], left_eye_2d[3], (255, 255, 0), 1)  # Yellow horizontal
            
            # Right eye EAR visualization
            cv2.line(frame, right_eye_2d[1], right_eye_2d[5], (0, 165, 255), 1)  # Orange vertical
            cv2.line(frame, right_eye_2d[2], right_eye_2d[4], (0, 165, 255), 1)  # Orange vertical
            cv2.line(frame, right_eye_2d[0], right_eye_2d[3], (255, 255, 0), 1)  # Yellow horizontal
            
            # Calculate secondary metric: eye area
            left_area = self._calculate_eye_area(left_eye_3d)
            right_area = self._calculate_eye_area(right_eye_3d)
            avg_area = (left_area + right_area) / 2.0
            
            # Visualize eye area by filling the eye contours with semi-transparent color
            # Color intensity based on area size
            alpha = 0.3
            overlay = frame.copy()
            area_color = (0, 0, 255) if avg_area < 0.0015 else (0, 255, 255)  # Red if below threshold, cyan if above
            
            cv2.fillPoly(overlay, [np.array(left_eye_2d)], area_color)
            cv2.fillPoly(overlay, [np.array(right_eye_2d)], area_color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Combined decision based on both EAR, area, and adjusted threshold
            is_blinking = avg_ear < adjusted_threshold
            
            # Adjust for head tilt - if area is still significant despite low EAR, 
            # it might be due to head tilt rather than actual blinking
            area_threshold = 0.0015  # This value may need tuning
            if is_blinking and avg_area > area_threshold:
                # If the eye area is still large, it's likely not a real blink
                is_blinking = False
            
            # Determine if blinking
            if is_blinking:
                detection_state = "Blinking"
                cv2.putText(frame, "BLINKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Only count blink if previous state wasn't blinking
                if self.last_detection_state != "Blinking":
                    self.blink_count += 1
                    self.total_blinks += 1
            else:
                detection_state = "Eyes open"
                cv2.putText(frame, "EYES OPEN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check if a minute has passed
            current_minute = int((time.time() - self.start_time) / 60)
            if current_minute > self.last_minute:
                print(f"{self.blink_count} blinks in 60 seconds (Total blinks: {self.total_blinks})")
                self.blink_count = 0
                self.last_minute = current_minute
                self.minutes_elapsed += 1

            # Display blink count
            cv2.putText(frame, f"Blinks this minute: {self.blink_count}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Total blinks: {self.total_blinks}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display EAR, Area, Head Tilt and Threshold values
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Area: {avg_area:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Head tilt: {head_tilt:.1f}°", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Threshold: {adjusted_threshold:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display time information
            cv2.putText(frame, f"Time since blink: {time_since_blink:.1f}s", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Required interval: {self.seconds_between_blinks:.1f}s", (10, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display countdown
            if seconds_until_reminder > 0:
                cv2.putText(frame, f"Reminder in: {seconds_until_reminder:.1f}s", (10, 300),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "PLEASE BLINK!", (10, 300),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Only print if the state has changed
        if detection_state != self.last_detection_state:
            if detection_state:
                if detection_state == "Blinking" and self.last_ear_value is not None:
                    print(f"{detection_state} (EAR: {avg_ear:.2f})")
                else:
                    print(detection_state)
            self.last_detection_state = detection_state
            self.last_ear_value = avg_ear
            
        # Update last blink time
        if detection_state == "Blinking":
            self.last_blink_time = current_time
        
        return frame, detection_state, play_reminder

def process_single_image(image_path: str, output_path: str, detector: BlinkDetector) -> None:
    """
    Process a single image and save the annotated version.
    
    Args:
        image_path: Path to input image
        output_path: Path to save annotated image
        detector: BlinkDetector instance
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Process the frame
    processed_frame, detection_state, _ = detector.process_frame(frame)
    
    # Save the annotated image
    cv2.imwrite(output_path, processed_frame)
    print(f"Processed image saved to {output_path}")
    if detection_state:
        print(f"Detection state: {detection_state}")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Webcam Blink Detection')
    parser.add_argument('--frames', type=int, default=1, help='Process one frame for every N frames')
    parser.add_argument('--threshold', type=float, default=0.45, help='EAR threshold for blink detection')
    parser.add_argument('--required-bpm', type=int, default=6, help='Required blinks per minute')
    parser.add_argument('--annotate-single-frame', type=str, help='Path to single image to annotate')
    args = parser.parse_args()
    
    # Convert required blinks per minute to seconds between blinks
    seconds_between_blinks = 60.0 / args.required_bpm
    
    # Initialize blink detector
    blink_detector = BlinkDetector(
        process_every_n_frames=args.frames, 
        blink_threshold=args.threshold,
        seconds_between_blinks=seconds_between_blinks
    )
    
    # If single frame mode is requested
    if args.annotate_single_frame:
        output_path = args.annotate_single_frame.rsplit('.', 1)[0] + '-annotated.' + args.annotate_single_frame.rsplit('.', 1)[1]
        process_single_image(args.annotate_single_frame, output_path, blink_detector)
        return

    # Regular webcam mode
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Frame counter
    frame_count = 0
    last_reminder_time = time.time()
    
    print("Webcam Blink Detection Started. Press 'q' to quit.")
    
    # Function to play sound in a separate thread
    def play_reminder_sound():
        try:
            # Generate a simple beep sound
            frequency = 440  # 440 Hz = A4 note
            fs = 44100  # 44100 samples per second
            seconds = 0.25  # Duration of the beep
            
            # Generate time array
            t = np.linspace(0, seconds, int(seconds * fs), False)
            
            # Generate sine wave
            note = np.sin(2 * np.pi * frequency * t)
            
            # Ensure it's in int16 range and convert
            audio = note * (2**15 - 1)
            audio = audio.astype(np.int16)
            
            # Play the sound
            play_obj = sa.play_buffer(audio, 1, 2, fs)
            play_obj.wait_done()
        except Exception as e:
            print(f"Error playing sound: {e}")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            if frame_count % blink_detector.process_every_n_frames == 0:
                processed_frame, _, should_play_reminder = blink_detector.process_frame(frame)
                
                # Play reminder sound if needed (maximum once per second)
                current_time = time.time()
                if should_play_reminder and (current_time - last_reminder_time) >= 1.0:
                    threading.Thread(target=play_reminder_sound, daemon=True).start()
                    last_reminder_time = current_time
                
                cv2.imshow('Blink Detection', processed_frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        # Get final statistics
        stats = blink_detector.get_blink_stats()
        print("\nFinal Statistics:")
        print(f"Total blinks: {int(stats['total_blinks'])}")
        print(f"Total minutes: {stats['total_minutes']:.1f}")
        print(f"Average blinks per minute: {stats['average_bpm']:.1f}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
