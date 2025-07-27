from queue import Queue
from threading import Event
import cv2
import mediapipe as mp
# from pynput.keyboard import Key, Controller
import time
from collections import deque
import numpy as np
import pygame # Import numpy for array type hinting
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')

# --- Initialize MediaPipe Hands (moved out of loop for efficiency) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- Camera Object (global or passed) ---
# It's common to initialize this once and pass it, or make it global if it's the only camera.


class DetectMoving:

	cap: cv2.VideoCapture
	y_coords_history: deque
	# --- Global Settings (can be adjusted) ---
	SMOOTHING_FRAMES = 2        # Number of frames for smoothing (moving average)
	UP_THRESHOLD = -0.0015       # Threshold for detecting an upward lift (more negative = requires bigger lift)
	VELOCITY_THRESHOLD = -0.005 # Minimum Y change per frame for a FAST lift (more negative = faster lift needed)
	DOUBLE_PRESS_TIME_LIMIT = 0.7# Number of frames for smoothing (moving average)
	last_up_detection_time = 0
	key_pressed_flag = False
	showing = False

	def __init__(self, cap: cv2.VideoCapture=None,
                 smoothing_frames: int = SMOOTHING_FRAMES,
				 up_threshold: float = UP_THRESHOLD,
				 velocity_threshold: float = VELOCITY_THRESHOLD,
				 showing =False
				 ):

		if cap is None:
			self.cap = cv2.VideoCapture(0)
		else:
			self.cap = cap

		self.SMOOTHING_FRAMES=smoothing_frames
		self.UP_THRESHOLD=up_threshold
		self.VELOCITY_THRESHOLD=velocity_threshold
		self.y_coords_history = deque(maxlen=self.SMOOTHING_FRAMES)
		self.showing = showing

	
	def get_camera_frame(self, camera_capture: cv2.VideoCapture) -> tuple[np.ndarray | None, np.ndarray | None]:
		"""
		Captures a frame from the camera, flips it, and converts it to RGB.

		Args:
			camera_capture (cv2.VideoCapture): The OpenCV VideoCapture object.

		Returns:
			tuple[np.ndarray | None, np.ndarray | None]: A tuple containing:
				- The original BGR frame (flipped) for display, or None if capture fails.
				- The RGB frame for MediaPipe processing, or None if capture fails.
		"""
		success, image = camera_capture.read()
		if not success:
			print("Error: Could not retrieve frame from webcam.")
			return None, None

		# Flip the image horizontally for a "mirror" effect
		image = cv2.flip(image, 1)
		# cv2.save
		# Convert BGR image to RGB for MediaPipe
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		return image, image_rgb

	def analyze_hand_movement(self, y_coords_history: deque) -> str | None:
		"""
		Analyzes the history of Y-coordinates to determine if an upward hand movement
		warrants a key press (single or double).

		Args:
			y_coords_history (deque): A deque containing the recent history of Y-coordinates
									of the hand landmark (e.g., wrist).

		Returns:
			str | None: 'up_once' for a single up arrow press, 'up_twice' for a double press,
						or None if no action is needed yet.
		"""
		
		if len(self.y_coords_history) < self.SMOOTHING_FRAMES:
			# Not enough data for smoothing yet
			return None

		smoothed_current_y = sum(y_coords_history) / len(y_coords_history)

		# To calculate delta_y, we need a 'previous' smoothed value.
		# We take the average of all but the most recent element.
		if len(y_coords_history) > 1:
			prev_smoothed_y = sum(list(y_coords_history)[0:-1]) / (len(y_coords_history) - 1)
			delta_y = smoothed_current_y - prev_smoothed_y # Y decreases when hand goes up
		else:
			return None # Not enough history for delta calculation

		action = None
		# print(f"Delta: {delta_y} coors:{self.y_coords_history}")

		if delta_y < self.UP_THRESHOLD :
			current_time = time.time()
			action = 'up_once'
			self.key_pressed_flag = True
			self.last_up_detection_time = current_time # Store time of this first upward movement
		

			# # Check for double press
			# if (current_time - self.last_up_detection_time) < self.DOUBLE_PRESS_TIME_LIMIT and delta_y < self.VELOCITY_THRESHOLD:
			# 	action = 'up_twice'
			# 	self.key_pressed_flag = True
			# 	self.last_up_detection_time = 0  # Reset for next double event
			# elif (current_time - self.last_up_detection_time) > self.DOUBLE_PRESS_TIME_LIMIT:
			# 	# Single press (first or after a long pause)
			# 	action = 'up_once'
			# 	self.key_pressed_flag = True
			# 	self.last_up_detection_time = current_time # Store time of this first upward movement
			
		# Condition to reset the key_pressed_flag when the hand moves down or stabilizes
		# This prevents continuous key presses while the hand remains "up"
		elif delta_y > -self.UP_THRESHOLD / 2 and self.key_pressed_flag:
			self.key_pressed_flag = False
			self.last_up_detection_time = 0 # Also reset time if hand is fully down

		return action
	def detect_up(self):
		result = False
		if not self.cap.isOpened():
			print("Error: Could not open webcam. Please ensure it's connected and not in use.")
			return
		# Get frame from camera using the new method
		image, image_rgb = self.get_camera_frame(self.cap)
		if image is None: # If get_camera_frame returned None, something went wrong
			print("Error: No image captured from webcam.")
			return None
# Process the RGB image with MediaPipe Hands
		results = hands.process(image_rgb)
		if self.showing:
			cv2.imshow('Hand Movement Analyzer', image)

		if results.multi_hand_landmarks:
			# print("Hand landmarks detected.")
			for hand_landmarks in results.multi_hand_landmarks:
				mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

				# Get Y-coordinate of the wrist (Landmark 0) for analysis
				current_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
				
				self.y_coords_history.append(current_y)
				if len(self.y_coords_history) == self.SMOOTHING_FRAMES:
					# print(f"Current Y-coordinate of wrist: {current_y} coors:{self.y_coords_history}")
					pass

				# Analyze hand movement using the refactored method
				action_needed = self.analyze_hand_movement(self.y_coords_history)
				if action_needed is not None:
					result = True
					self.y_coords_history.clear()
					print(f"Hand lifted up! Pressing '{action_needed}' once.")


				# Perform keyboard action based on the analysis
				if action_needed == 'up_once':
					print("Hand lifted up! Pressing 'up arrow' once.")
					# keyboard.press(Key.up)
					# keyboard.release(Key.up)
					# time.sleep(0.1) # Small delay to avoid rapid fire if threshold is hit repeatedly
				elif action_needed == 'up_twice':
					pass
					# print("Fast double lift! Pressing 'up arrow' twice.")
					# keyboard.press(Key.up)
					# keyboard.release(Key.up)
					# time.sleep(0.1) # Small delay between presses
					# keyboard.press(Key.up)
					# keyboard.release(Key.up)
					# time.sleep(0.1) # Small delay after double press

		else:
			# If no hand is detected, clear history and reset flags
			self.y_coords_history.clear()
			self.last_up_detection_time = 0
			self.key_pressed_flag = False
		if self.showing:
			cv2.imshow('Hand Movement Analyzer', image)

		return result



	def updateEvents(self, events: list[pygame.event.Event]) -> list[pygame.event.Event]:
		"""
		Updates the camera events and processes hand movements.
		This method can be called in a loop to continuously check for hand movements.
		"""
		for i in range(20):

			if self.detect_up():
				events.append(pygame.event.Event(pygame.KEYUP, key=pygame.K_UP))
				print("ADD EVENT")
				break
		return events

	def updateEventsWhile(self,stop_detector: Event, action: Queue) -> bool:
		"""
		Updates the camera events and processes hand movements.
		This method can be called in a loop to continuously check for hand movements.
		"""
		result = False
		while True:
			if stop_detector.is_set():
				break
		# for i in range(20):
			if self.detect_up():
				self.key_pressed_flag = False  # Reset flag after processing
				self.last_up_detection_time = 0
				self.y_coords_history.clear()
				logging.info("Hand lifted up! Pressing 'up arrow' once.")
				action.put_nowait(True)  
				# events.append(pygame.event.Event(pygame.KEYUP, key=pygame.K_UP))
				print("ADD EVENT")
				result = True
				# break
		return result

	def stop(self):
		"""
		Releases the camera and closes MediaPipe Hands.
		This should be called when the application is closing to free resources.
		"""
		if self.cap.isOpened():
			self.cap.release()
		hands.close()
		cv2.destroyAllWindows()



## Main Program Loop

def main ():

	cap = cv2.VideoCapture(0)
	
	if not cap.isOpened():
		print("Error: Could not open webcam. Please ensure it's connected and not in use.")
		exit()

	print("Program started. Lift your hand vertically up to simulate 'up arrow' presses.")
	i=0
	detector = DetectMoving(cap)
	while True:
		i += 1
	# for i in range(100):
		#time.sleep(0.01)
		print(f"Starting detectsion {i}...")
		if detector.detect_up():
			print("--------------------------------------------------")
			print("Detected upward movement!")
			break
		detector.detect_up()  # Call the method to detect hand movement
	# --- Release Resources ---
	cap.release()
	cv2.destroyAllWindows()
	hands.close()


if __name__=="__main__":
	main()