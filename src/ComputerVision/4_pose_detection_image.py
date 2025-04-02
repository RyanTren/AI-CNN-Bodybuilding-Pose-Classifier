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

def create_modified_model():
    # Load the base model
    base_model = load_model("src\\ComputerVision\\keras_model.h5", compile=False)
    
    # Create a new Sequential model
    new_model = tf.keras.Sequential()
    
    # Add all layers except the last one from the base model
    for layer in base_model.layers[:-1]:
        new_model.add(layer)
        
    # Add new classification layer with 5 outputs
    new_model.add(tf.keras.layers.Dense(5, activation='softmax'))
    
    # Compile the model
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return new_model

# Create the modified model
model = create_modified_model()

# Load the labels
class_names = open("src\\ComputerVision\\labels.txt", "r").readlines()

# Clear screen for Windows or Unix-based systems
clear_command = 'cls' if os.name == 'nt' else 'clear'
os.system(clear_command)
print("Current working directory:", working_directory)

def create_efficient_dataset(directory, batch_size=32):
    """
    Dataset creation for 5-class pose classification
    """
    # Define class names in alphabetical order to match directory structure
    pose_classes = [
        'Abs and Thighs',
        'Back Double Bicep',
        'Front Double Bicep',
        'Negative_Sample',
        'Side Chest'
    ]

    # Create the training dataset
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        image_size=(224, 224),
        seed=123,
        shuffle=True,
        validation_split=0.2,
        subset="training",
        label_mode='categorical',
        class_names=pose_classes
    )

    # Create validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        image_size=(224, 224),
        seed=123,
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        label_mode='categorical',
        class_names=pose_classes
    )

    def augment_images(x, y):
        # Geometric transformations
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.3)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        x = tf.image.random_saturation(x, 0.7, 1.3)
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

def verify_dataset_setup():
    """
    Verify the dataset structure for 5-class classification
    """
    base_path = 'src/ComputerVision'
    expected_classes = ['Negative_Sample', 'Front Double Bicep', 'Back Double Bicep', 
                       'Side Chest', 'Abs and Thighs']
    
    print("\nChecking dataset setup...")
    print("-" * 50)
    
    all_exists = True
    class_counts = {}
    
    for class_name in expected_classes:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"❌ '{class_name}' folder not found!")
            print(f"Create it at: {class_path}")
            all_exists = False
        else:
            # Count images in this class
            images = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = images
            print(f"✓ Found {images} images in {class_name} folder")
    
    if class_counts:
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} images")
    
    return all_exists

def analyze_pose(prediction, confidence_threshold=0.6):
    """
    Enhanced pose analysis for 5-class classification
    """
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    
    if confidence_score < confidence_threshold:
        if confidence_score < 0.3:
            return False, "Very low confidence - Please retake photo", confidence_score
        else:
            return False, "Low confidence - Adjust pose or lighting", confidence_score
    
    return True, f"Valid {class_names[index].strip()} pose detected", confidence_score

# Training setup
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train if dataset is properly set up
if verify_dataset_setup():
    train_ds, val_ds = create_efficient_dataset('src/ComputerVision')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stopping]
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
        prediction = model.predict(input_data, verbose=0)
        is_valid, message, confidence_score = analyze_pose(prediction)
        
        print("\nResults:")
        print("-" * 40)
        print(f"Detected Pose: {message}")
        print(f"Confidence: {str(np.round(confidence_score * 100))[:-2]}%")
        if not is_valid:
            print("Suggestion: Try adjusting pose, lighting, or camera angle")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")



