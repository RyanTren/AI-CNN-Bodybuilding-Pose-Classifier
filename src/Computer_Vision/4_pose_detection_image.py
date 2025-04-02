import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image, ImageOps
import h5py
import tensorflow as tf


# Print and store the current working directory
working_directory = os.getcwd()
print("Current working directory:", working_directory)

# Patch the model file to remove the problematic "groups" parameter
with h5py.File("src\\ComputerVision\\keras_model.h5", "r+") as f:
    model_config = f.attrs.get("model_config")
    if model_config and '"groups": 1,' in model_config:
        new_config = model_config.replace('"groups": 1,', '')
        f.attrs.modify("model_config", new_config)
        f.flush()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("src\\ComputerVision\\keras_model.h5", compile=False)
class_names = open("src\\ComputerVision\\labels.txt", "r").readlines()

# Clear screen for Windows or Unix-based systems
clear_command = 'cls' if os.name == 'nt' else 'clear'
os.system(clear_command)
print("Current working directory:", working_directory)

def create_efficient_dataset(directory, batch_size=32):
    """
    Optimized dataset creation with aggressive augmentation for limited data
    """
    # Create the base dataset
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        image_size=(224, 224),
        seed=123,
        shuffle=True,
        validation_split=0.2,
        subset="training"
    )

    # Create validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        image_size=(224, 224),
        seed=123,
        shuffle=True,
        validation_split=0.2,
        subset="validation"
    )

    def augment_images(x, y):
        # Enhanced augmentation for limited data
        # Geometric transformations
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_rotation(x, 0.2)
        x = tf.image.random_zoom(x, 0.2)
        
        # Color transformations
        x = tf.image.random_brightness(x, 0.3)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        x = tf.image.random_saturation(x, 0.7, 1.3)
        
        # Add slight noise for robustness
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
        x = x + noise
        x = tf.clip_by_value(x, 0, 1)
        
        return x, y

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (
        dataset
        .map(augment_images, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
    
    return train_dataset, val_dataset

def analyze_pose(prediction, confidence_threshold=0.6):
    """
    Enhanced pose analysis with detailed feedback
    """
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    
    if confidence_score < confidence_threshold:
        if confidence_score < 0.3:
            return False, "Very low confidence - Please retake photo", confidence_score
        else:
            return False, "Low confidence - Adjust pose or lighting", confidence_score
    
    return True, "Valid pose detected", confidence_score

# Load model and compile with optimizations
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training setup
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

def verify_dataset_setup():
    """
    Verify the dataset structure and print helpful information
    """
    base_path = 'src/ComputerVision'
    good_path = os.path.join(base_path, 'good_image')
    bad_path = os.path.join(base_path, 'bad_image')
    
    print("\nChecking dataset setup...")
    print("-" * 50)
    
    # Check directories exist
    if not os.path.exists(good_path):
        print("❌ 'good_image' folder not found!")
        print(f"Create it at: {good_path}")
        return False
    
    if not os.path.exists(bad_path):
        print("❌ 'bad_image' folder not found!")
        print(f"Create it at: {bad_path}")
        return False
    
    # Count images
    good_images = len([f for f in os.listdir(good_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    bad_images = len([f for f in os.listdir(bad_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"✓ Found {good_images} images in good_image folder")
    print(f"✓ Found {bad_images} images in bad_image folder")
    
    # Provide recommendations
    if good_images < 50 or bad_images < 50:
        print("\nRecommendation: Add more images for better training")
        print("Aim for at least 50 images in each category")
    
    if abs(good_images - bad_images) > min(good_images, bad_images) * 0.3:
        print("\nWarning: Dataset is imbalanced")
        print("Try to keep similar numbers of good and bad images")
    
    return True

# Add this before training
if verify_dataset_setup():
    train_ds, val_ds = create_efficient_dataset('src/ComputerVision')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stopping],
        class_weight={0: 1.0, 1: 1.0}  # Adjust based on your class distribution
    )
else:
    print("\nPlease set up your dataset before training")
    exit(1)

# Main prediction loop
while(True):
    image_path = input("Enter the path to an image (or 'q' to quit): ").strip()
    
    if image_path.lower() == 'q':
        break
        
    if not os.path.exists(image_path):
        print("Image file does not exist.")
        continue
        
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img, dtype=np.float32)
        normalized_img = (img_array / 127.5) - 1
        input_data = normalized_img.reshape(1, 224, 224, 3)

        # Perform prediction
        prediction = model.predict(input_data, verbose=0)  # Reduce output noise
        is_valid, message, confidence_score = analyze_pose(prediction)
        
        if is_valid:
            index = np.argmax(prediction)
            predicted_class = class_names[index]
            print("\nResults:")
            print("-" * 40)
            print(f"Class: {predicted_class[2:]}")
            print(f"Confidence: {str(np.round(confidence_score * 100))[:-2]}%")
            print(f"Status: {message}")
        else:
            print("\nResults:")
            print("-" * 40)
            print(f"Status: {message}")
            print(f"Confidence: {str(np.round(confidence_score * 100))[:-2]}%")
            print("Suggestion: Try adjusting pose, lighting, or camera angle")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")



