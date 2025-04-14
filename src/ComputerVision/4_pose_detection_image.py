import os
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from PIL import Image, ImageOps
import h5py
import mediapipe as mp
import cv2

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,  # Use 1 instead of 2 for better performance on laptop
    enable_segmentation=False,  # Disable segmentation for better performance
    min_detection_confidence=0.5
)

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

def extract_pose_landmarks(image):
    """Extract pose landmarks using MediaPipe Pose"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Process the image
    results = pose.process(image)
    if not results.pose_landmarks:
        return None
    
    # Convert landmarks to numpy array
    landmarks = np.zeros((33, 3))  # MediaPipe uses 33 landmarks
    for i, landmark in enumerate(results.pose_landmarks.landmark):
        landmarks[i] = [landmark.x, landmark.y, landmark.z]
    
    return landmarks

def create_pose_model():
    # Load the base model
    base_model = models.load_model("src\\ComputerVision\\keras_model.h5", compile=False)
    
    # Create a new Sequential model for image features
    image_model = tf.keras.Sequential()
    for layer in base_model.layers[:-1]:
        image_model.add(layer)
    
    # Create a model for pose landmarks (MediaPipe uses 33 landmarks)
    landmark_input = layers.Input(shape=(33, 3))
    # Flatten landmarks first
    x = layers.Flatten()(landmark_input)
    landmark_features = layers.Dense(32, activation='relu')(x)  # Reduced from 64 to 32
    landmark_features = layers.BatchNormalization()(landmark_features)
    landmark_features = layers.Dropout(0.3)(landmark_features)
    
    # Now both tensors will have shape (None, N) where N is the feature dimension
    combined = layers.Concatenate()([image_model.output, landmark_features])
    
    # Add classification layers (reduced complexity)
    x = layers.Dense(64, activation='relu')(combined)  # Reduced from 128 to 64
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(5, activation='softmax')(x)
    
    # Create the final model
    model = tf.keras.Model(inputs=[image_model.input, landmark_input], outputs=output)
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path):
    """Enhanced image preprocessing with MediaPipe pose detection"""
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Extract pose landmarks
    landmarks = extract_pose_landmarks(img_array)
    if landmarks is None:
        raise ValueError("No pose detected in image")
    
    # Resize image
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)
    
    # Normalize image
    mean = np.mean(img_array)
    std = np.std(img_array)
    normalized_img = (img_array - mean) / (std + 1e-7)
    normalized_img = np.clip(normalized_img * 1.2, -1, 1)
    
    # Add batch dimension
    image_input = normalized_img.reshape(1, 224, 224, 3)
    landmark_input = landmarks.reshape(1, 33, 3)  # MediaPipe uses 33 landmarks
    
    return image_input, landmark_input

def create_pose_dataset(directory, batch_size=8):
    """Dataset creation with pose landmark extraction"""
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
        class_names=class_names
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
        class_names=class_names
    )

    def process_batch(images, labels):
        # Extract pose landmarks for each image
        def process_single_image(image):
            # Convert tensor to uint8 numpy array using tf.py_function
            def extract_landmarks(img_tensor):
                try:
                    img_array = tf.cast(img_tensor * 255.0, tf.uint8).numpy()
                    landmark = extract_pose_landmarks(img_array)
                    if landmark is None:
                        # Return zero landmarks if detection fails
                        return np.zeros((33, 3), dtype=np.float32)
                    return np.array(landmark, dtype=np.float32)
                except Exception as e:
                    print(f"Warning: Landmark extraction failed - {str(e)}")
                    # Return zero landmarks on error
                    return np.zeros((33, 3), dtype=np.float32)
            
            # Wrap the numpy operations in tf.py_function
            landmarks = tf.py_function(
                extract_landmarks,
                [image],
                tf.float32
            )
            landmarks.set_shape((33, 3))
            return landmarks
        
        # Process all images in the batch
        landmarks = tf.map_fn(
            process_single_image,
            images,
            dtype=tf.float32,
            parallel_iterations=AUTOTUNE
        )
        
        return (images, landmarks), labels

    # Optimize performance for laptop
    AUTOTUNE = 2  # Fixed value instead of tf.data.AUTOTUNE
    train_dataset = (
        dataset
        .map(process_batch, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .map(process_batch, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    
    # Clear memory
    import gc
    gc.collect()
    
    return train_dataset, val_dataset

def calculate_class_weights(directory):
    """
    Calculate class weights to handle imbalanced data
    """
    total_samples = 0
    class_counts = {}
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            total_samples += count
    
    class_weights = {}
    for class_name, count in class_counts.items():
        class_weights[list(class_counts.keys()).index(class_name)] = total_samples / (len(class_counts) * count)
    
    return class_weights

def lr_schedule(epoch):
    """
    Learning rate schedule
    """
    initial_lr = 0.0001
    if epoch < 10:
        return initial_lr
    elif epoch < 20:
        return initial_lr * 0.5
    else:
        return initial_lr * 0.1

# Training configuration
training_config = {
    'batch_size': 16,
    'epochs': 10,
    'validation_split': 0.2,
    'early_stopping_patience': 7,  # Stop if no improvement for 7 epochs
}

# Modified callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=training_config['early_stopping_patience'],
        restore_best_weights=True,
        min_delta=0.01  # 1% improvement threshold
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

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

def analyze_pose(prediction, confidence_threshold=0.4):
    """Pose analysis with pose verification"""
    try:
        # Apply temperature scaling
        temperature = 0.5
        scaled_prediction = prediction / temperature
        scaled_prediction = np.exp(scaled_prediction) / np.sum(np.exp(scaled_prediction), axis=1, keepdims=True)
        
        # Get top predictions
        top_indices = np.argsort(scaled_prediction[0])[::-1][:2]
        top_classes = [class_names[i] for i in top_indices]
        top_confidences = [scaled_prediction[0][i] for i in top_indices]
        
        # Class-specific thresholds
        class_thresholds = {
            'Abs and Thighs': 0.35,
            'Back Double Bicep': 0.35,
            'Front Double Bicep': 0.35,
            'Side Chest': 0.35,
            'Negative_Sample': 0.30
        }
        
        primary_class = top_classes[0]
        primary_confidence = top_confidences[0]
        secondary_confidence = top_confidences[1]
        
        # Check if it's a negative sample
        if primary_class == 'Negative_Sample':
            if primary_confidence < class_thresholds['Negative_Sample']:
                return False, "Not a valid pose - Please retake photo", primary_confidence
            return True, "Valid negative sample detected", primary_confidence
        
        # For pose classes
        if primary_confidence < class_thresholds[primary_class]:
            if primary_confidence < 0.25:
                return False, "Very low confidence - Please retake photo", primary_confidence
            else:
                return False, f"Low confidence for {primary_class} - Adjust pose or lighting", primary_confidence
        
        # Check for multiple poses
        if secondary_confidence > 0.25:
            return False, f"Multiple poses detected ({primary_class}: {primary_confidence:.2f}, {top_classes[1]}: {secondary_confidence:.2f})", primary_confidence
        
        return True, f"Valid {primary_class} pose detected", primary_confidence
    except Exception as e:
        return False, f"Error analyzing pose: {str(e)}", 0.0

# Create the model
model = create_pose_model()

# Load the labels
class_names = [
    'Negative_Sample',
    'Front Double Bicep',
    'Back Double Bicep',
    'Side Chest',
    'Abs and Thighs'
]

# Clear screen for Windows or Unix-based systems
clear_command = 'cls' if os.name == 'nt' else 'clear'
os.system(clear_command)
print("Current working directory:", working_directory)

# Train with memory optimization
if verify_dataset_setup():
    # Clear any existing sessions
    tf.keras.backend.clear_session()
    
    # Create datasets with optimized batch size
    train_ds, val_ds = create_pose_dataset(
        'src/ComputerVision',
        batch_size=training_config['batch_size']
    )
    
    # Train with optimized parameters
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_config['epochs'],
        callbacks=callbacks,
        workers=2,  # Limit worker processes for laptop
        use_multiprocessing=False  # Better stability on Windows
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
        # Preprocess image and extract pose landmarks
        image_input, landmark_input = preprocess_image(image_path)

        # Perform prediction
        prediction = model.predict([image_input, landmark_input], verbose=0)
        is_valid, message, confidence_score = analyze_pose(prediction)
        
        print("\nResults:")
        print("-" * 40)
        print(f"Detected Pose: {message}")
        print(f"Confidence: {str(np.round(confidence_score * 100))[:-2]}%")
        if not is_valid:
            print("Suggestion: Try adjusting pose, lighting, or camera angle")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")



