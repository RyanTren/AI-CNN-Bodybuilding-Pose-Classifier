import cv2
import time
import os
import winsound
import shutil
import pyttsx3
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

def generate_unique_suffix():
    """Generate a unique timestamp-based suffix for image filenames."""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

def capture_pose_images(pose_label, temp_pose_folder, image_count=300, capture_delay=0.05):
    """
    Captures images from the webcam and stores them in the specified temporary folder.
    Each filename includes a unique timestamp to prevent overwriting.
    """
    if not os.path.exists(temp_pose_folder):
        os.makedirs(temp_pose_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return 0

    count = 0
    for _ in range(image_count):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to capture image, skipping...")
            continue

        suffix = generate_unique_suffix()
        filename = os.path.join(temp_pose_folder, f"{pose_label.replace(' ', '_')}_img_{suffix}.jpg")
        cv2.imwrite(filename, frame)
        count += 1

        if count % 50 == 0:
            print(f"{count} images captured...")
        time.sleep(capture_delay)

    cap.release()
    return count

def move_temp_to_final(temp_folder, final_folder):
    """Moves all files from the temporary folder to the final target folder."""
    if not os.path.exists(final_folder):
        os.makedirs(final_folder, exist_ok=True)
    for filename in os.listdir(temp_folder):
        src = os.path.join(temp_folder, filename)
        dst = os.path.join(final_folder, filename)
        shutil.move(src, dst)
    shutil.rmtree(temp_folder)

# Define the target folders for each pose category.
pose_folders = {
    '1': ("front double bi", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\front double bi"),
    '2': ("back double bi", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\back double bi"),
    '3': ("side chest", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\side chest"),
    '4': ("abs and thighs", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\abs and thighs"),
    '5': ("not a pose (leave frame)", r"C:\Users\Colin\OneDrive\Pictures\bb_training_data\not a pose")
}

# Set up temp folder path
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_base = os.path.join(script_dir, "temp_data")
os.makedirs(temp_base, exist_ok=True)

captured_poses = 0

while True:
    # Show the pose menu before each choice
    print("\nPose Capture Script")
    print("---------------------")
    print("Available options:")
    for key, (label, _) in pose_folders.items():
        print(f"{key}: {label}")
    print("Type 'q' to quit the script.")

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

    # Announce the pose
    speak(f"Capturing for {pose_label}")

    # Create a temp folder for this capture session
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    temp_pose_folder = os.path.join(temp_base, f"{pose_label.replace(' ', '_')}_{today}")
    if os.path.exists(temp_pose_folder):
        shutil.rmtree(temp_pose_folder)
    os.makedirs(temp_pose_folder)

    print("Get ready! Capturing will begin in 5 seconds...")

    # 5-second countdown with medium beeps
    for i in range(5, 0, -1):
        print(f"{i}...")
        play_beep(1000, 200)  # Medium beep
        time.sleep(1)

    # Start capture with high-pitch beep
    play_beep(1500, 300)
    print("Capturing images now...")
    captured_count = capture_pose_images(pose_label, temp_pose_folder)
    play_beep(1500, 300)  # End capture beep

    print(f"Finished capturing {captured_count} images for {pose_label}.")

    # Confirmation menu
    print("\nOptions:")
    print("  A: ADD TO TRAINING DATA")
    print("  D: DELETE IMAGES")
    print("  R: DELETE IMAGES AND RETRY")
    decision = input("Enter your decision (A/D/R): ").strip().upper()

    if decision == 'A':
        move_temp_to_final(temp_pose_folder, final_target)
        print(f"Images for {pose_label} have been added to the training data.")
        captured_poses += 1
    elif decision == 'D':
        shutil.rmtree(temp_pose_folder)
        print(f"Images for {pose_label} have been deleted.")
    elif decision == 'R':
        shutil.rmtree(temp_pose_folder)
        print(f"Images for {pose_label} have been deleted. Retrying capture for {pose_label}...")
        continue
    else:
        print("Invalid option. No action taken; moving to next pose.")

# Final summary
print("\nCapture Session Complete!")
print("---------------------------")
print(f"Total pose sessions successfully added to training data: {captured_poses}")
print("Session summary:")
for key, (label, folder) in pose_folders.items():
    print(f" - {label}: {folder}")
print("Thank you for using the Pose Capture Script!")
