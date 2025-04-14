import cv2
import time
import os
import winsound  # For playing beeps on Windows
import shutil    # For moving or deleting directories
import pyttsx3   # For text-to-speech
import datetime

# Initialize text-to-speech engine
engine = pyttsx3.init()

def play_beep(frequency, duration):
    """Plays a beep sound using winsound."""
    winsound.Beep(frequency, duration)

def speak(text):
    """Speak out the text using pyttsx3."""
    engine.say(text)
    engine.runAndWait()

def capture_pose_images(pose_label, temp_pose_folder, image_count=300, capture_delay=0.05):
    """
    Captures images from the webcam and stores them in the specified temporary folder.
    Parameters:
      pose_label: string name of the pose (used for filenames)
      temp_pose_folder: target folder for saving images temporarily.
      image_count: number of images to capture (default is 300).
      capture_delay: delay between captures in seconds.
    Returns:
      The number of images successfully captured.
    """
    if not os.path.exists(temp_pose_folder):
        os.makedirs(temp_pose_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return 0

    count = 0
    for img_num in range(image_count):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to capture image, skipping...")
            continue

        filename = os.path.join(temp_pose_folder, f"{pose_label.replace(' ', '_')}_img_{img_num}.jpg")
        cv2.imwrite(filename, frame)
        count += 1

        # Optional: Print every 50 images captured
        if (img_num + 1) % 50 == 0:
            print(f"{img_num + 1} images captured...")
        time.sleep(capture_delay)

    cap.release()
    return count

def move_temp_to_final(temp_folder, final_folder):
    """
    Moves all files from the temporary folder to the final target folder.
    """
    if not os.path.exists(final_folder):
        os.makedirs(final_folder, exist_ok=True)
    # Move each file from the temp folder to the final destination
    for filename in os.listdir(temp_folder):
        src = os.path.join(temp_folder, filename)
        dst = os.path.join(final_folder, filename)
        shutil.move(src, dst)
    # Remove the now-empty temporary folder
    shutil.rmtree(temp_folder)

# Define the target folders for each pose category.
# Keys are the selection options.
pose_folders = {
    '1': ("front double bi", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\front double bi"),
    '2': ("back double bi", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\back double bi"),
    '3': ("side chest", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\side chest"),
    '4': ("abs and thighs", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\abs and thighs"),
    '5': ("not a pose (leave frame)", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\not a pose")
}

# Set the base path for temporary data (relative to the script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_base = os.path.join(script_dir, "temp_data")
os.makedirs(temp_base, exist_ok=True)

print("Pose Capture Script")
print("---------------------")
print("Available options:")
for key, (label, folder) in pose_folders.items():
    print(f"{key}: {label}")
print("Type 'q' to quit the script.")

captured_poses = 0

while True:
    choice = input("\nEnter the number corresponding to the pose you want to capture (or 'q' to quit): ").strip().lower()
    if choice == 'q':
        print("Exiting capture mode...")
        break

    if choice not in pose_folders:
        print("Invalid selection. Please choose a valid option.")
        continue

    pose_label, final_target = pose_folders[choice]
    print(f"\nYou selected: {pose_label}")
    if choice == '5':
        print("For 'not a pose', please leave the camera frame when instructed.")

    # Announce the pose via text-to-speech
    speak(f"Capturing for {pose_label}")

    # Create temporary subfolder for this pose in the temp_data directory, using a timestamp (today's date)
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    temp_pose_folder = os.path.join(temp_base, f"{pose_label.replace(' ', '_')}_{today}")
    if os.path.exists(temp_pose_folder):
        shutil.rmtree(temp_pose_folder)  # Clear out previous data if exists

    # High-pitched beep to signal start of capture
    play_beep(1500, 150)
    print("Get ready! Capturing will begin in 5 seconds...")

    # 5-second countdown
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("Capturing images now...")
    captured_count = capture_pose_images(pose_label, temp_pose_folder, image_count=300, capture_delay=0.05)
    print(f"Finished capturing {captured_count} images for {pose_label}.")

    # Low-pitched beep to signal end of capture
    play_beep(600, 150)

    # Ask the user for confirmation on what to do with the captured images
    print("\nOptions:")
    print("  A: ADD TO TRAINING DATA")
    print("  D: DELETE IMAGES")
    print("  R: DELETE IMAGES AND RETRY")
    decision = input("Enter your decision (A/D/R): ").strip().upper()

    if decision == 'A':
        # Move images from temp folder to final training data folder
        move_temp_to_final(temp_pose_folder, final_target)
        print(f"Images for {pose_label} have been added to the training data.")
        captured_poses += 1
    elif decision == 'D':
        # Delete captured images
        shutil.rmtree(temp_pose_folder)
        print(f"Images for {pose_label} have been deleted.")
    elif decision == 'R':
        # Delete images and retry capturing for this pose
        shutil.rmtree(temp_pose_folder)
        print(f"Images for {pose_label} have been deleted. Retrying capture for {pose_label}...")
        # Use continue to retry without incrementing captured_poses
        continue
    else:
        print("Invalid option. No action taken; moving to next pose.")

print("\nCapture Session Complete!")
print("---------------------------")
print(f"Total pose sessions successfully added to training data: {captured_poses}")
print("Session summary:")
for key, (label, folder) in pose_folders.items():
    print(f" - {label}: {folder}")
print("Thank you for using the Pose Capture Script!")
