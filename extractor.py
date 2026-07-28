import cv2
import os
import numpy as np

# pip install opencv-python numpy

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
VIDEO_FILE = "german_book_recording.mp4" # Put your exact video filename here
OUTPUT_FOLDER = "german_pages" # The folder where pictures will be saved
THRESHOLD = 3.0 # Sensitivity: Lower means more sensitive to movement
MIN_STABLE_FRAMES = 15 # How many frames the page must stay still to count as a "shot"

def extract_stable_pages():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_FILE}'")
        return

    print("Analyzing video file... Please wait.")
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Empty video file.")
        return

    # Convert first frame to grayscale for mathematical movement comparisons
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    page_counter = 1
    stable_frame_count = 0
    saved_on_this_pause = False
    best_candidate_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Video ended

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the absolute mathematical difference between this frame and the previous frame
        frame_diff = cv2.absdiff(gray, prev_gray)
        mean_diff = np.mean(frame_diff)

        # If movement is below the threshold, the book page is stable (currently pausing)
        if mean_diff < THRESHOLD:
            stable_frame_count += 1
            # Keep updating to the latest stable frame during the pause to ensure no hand motion blur
            best_candidate_frame = frame.copy()
            
            # If it has been still for long enough, and we haven't saved it yet, queue it up
            if stable_frame_count >= MIN_STABLE_FRAMES and not saved_on_this_pause:
                filename = os.path.join(OUTPUT_FOLDER, f"page_index_{page_counter:03d}.jpg")
                cv2.imwrite(filename, best_candidate_frame)
                print(f"Captured: {filename} (Detected stability on page {page_counter})")
                
                page_counter += 1
                saved_on_this_pause = True
        else:
            # Movement detected! (You are flipping the page) -> Reset timers
            stable_frame_count = 0
            saved_on_this_pause = False

        prev_gray = gray

    cap.release()
    print(f"\nFinished! Extracted {page_counter - 1} clear page pictures into the '{OUTPUT_FOLDER}' directory.")

if __name__ == "__main__":
    extract_stable_pages()