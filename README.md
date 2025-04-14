# CS3642 AI Semester Project - AI-Image Processing Model on Bodybuilding Poses

## directory structure

AI-Based-Fake-Job-Posting-Detector/
└── src/
    └── ComputerVision/
        ├── good_image/
        │   ├── pose1.jpg        # Good bodybuilding poses
        │   ├── pose2.jpg
        │   └── ...
        ├── bad_image/
        │   ├── bad1.jpg         # Bad/incorrect bodybuilding poses
        │   ├── bad2.jpg
        │   └── ...
        ├── keras_model.h5
        ├── labels.txt
        └── 4_pose_detection_image.py


- good_image/ folder should contain:
    * Correctly executed bodybuilding poses
    * Clear, well-lit photos
    * Proper form and stance
    * Professional competition poses

- bad_image/ folder should contain:
    * Incorrect pose executions
    * Poor form examples
    * Blurry or poorly lit photos
    * Common mistakes in poses


## requirements to run model!
- First have conda installed, then create an environment in the project's terminal.

```ps
conda init powershell
conda --version
```

```ps
conda create -n new-cv-env python=3.10
conda activate new-cv-env
conda install -c conda-forge opencv
conda install tensorflow
conda install pillow h5py


conda install ultralytics 
conda install -c conda-forge ultralytics #yolo8 library dependency
conda install torch torchvision torchaudio                                                                                                                           
                                                                 
```

- This will help us install all the tensorflow dependencies used to run the Keras Computer Vision model

## Notes on Implementations/Iterations
### Notes on First implementation: 
- accuracy is not there... 
- need to use a better model or grow our data set

### Notes on Second Implementation:
1. Training Progress:
- The model trained for 50 epochs
- Training accuracy reached 100% (1.0000) very quickly
- Validation accuracy improved from 13% to 78.26% by the end
- There's a significant gap between training and validation accuracy, suggesting some overfitting

2. Dataset Composition:
- Total of 230 images across 5 classes
- Training set: 184 images
- Validation set: 46 images

- Class distribution:
    * Negative_Sample: 66 images
    * Front Double Bicep: 41 images
    * Back Double Bicep: 41 images
    * Side Chest: 41 images
    * Abs and Thighs: 41 images

3. Prediction Results:
The model shows some concerning patterns:
- It frequently misclassifies images as "Side Chest" pose (class 3)
- Confidence scores are generally low (mostly between 50-65%)
- Many predictions fall below the confidence threshold, resulting in "Low confidence" messages
- Even when it makes predictions, the confidence is not very high

4. Issues Identified:
- The model appears to have a bias towards predicting "Side Chest" pose
- Low confidence scores across different pose types
- Poor generalization as evidenced by the gap between training and validation accuracy
- The model might be overfitting to the training data
5. Recommendations:
- Consider increasing the dataset size for better generalization
- Implement stronger regularization techniques to reduce overfitting
- Adjust the model architecture or hyperparameters
- Consider using data augmentation more aggressively
- The confidence threshold might need adjustment based on the actual use case

### Notes on 3rd Iteration
the model is giving very low confidence scores (mostly between 24-51%) for all images, which means it's not properly learning to classify the poses. Making these adjustments to improve the model's performance:
- Lower the confidence threshold since it's too strict
- Adjust the model architecture
- Modify the training parameters
- Add more preprocessing steps

- Model Architecture Changes:
    * Added BatchNormalization layers for better training stability
    * Added an intermediate dense layer (128 units) with ReLU activation
    * Reduced dropout rate from 0.5 to 0.3
    * Removed L2 regularization which might have been too strong
- Training Parameters:
    * Increased learning rate from 0.00001 to 0.001 for faster learning
    * Increased epochs from 50 to 100
    * Increased early stopping patience from 10 to 15
    * Made learning rate reduction less aggressive (factor 0.5 instead of 0.2)
- Confidence Threshold:
    * Lowered the main confidence threshold from 0.6 to 0.3
    * Lowered the "very low confidence" threshold from 0.3 to 0.2
- Image Preprocessing:
    * Created a dedicated preprocessing function
    * Changed normalization to [0,1] range instead of [-1,1]
    * Maintained aspect ratio preservation during resizing


## Improvement Log:
### Improve Data Quality (Fastest Impact):
* Focus on quality over quantity
* Use high-quality competition photos
* Ensure consistent lighting and angles

```python
import tensorflow as tf

def create_augmented_dataset(directory, batch_size=32):
    """
    Create a dataset with augmentation specifically tuned for bodybuilding poses
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

    # Enhanced augmentation for bodybuilding poses
    def augment_images(x, y):
        # Random flip - good for pose symmetry
        x = tf.image.random_flip_left_right(x)
        
        # Slight rotation - accounts for camera angle variations
        x = tf.image.random_rotation(x, 0.1)  # 10% rotation max
        
        # Brightness/contrast - handles different lighting conditions
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        
        # Slight zoom - handles different distances
        x = tf.image.random_zoom(x, 0.1)
        
        return x, y

    # Apply augmentation and optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (
        dataset
        .map(augment_images, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return train_dataset, val_dataset

# Usage example:
def train_model():
    # Create datasets
    train_ds, val_ds = create_augmented_dataset(
        "path/to/pose/images",
        batch_size=32
    )

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds
```

### Add data augmentation:

### Use Transfer Learning (Quick Win):

### Improve Pose Detection Logic:

### Add Basic Feature Extraction:

### Implement Simple Ensemble Method:

### Priority Order:
* Use transfer learning with MobileNetV2 or EfficientNetB0 (quick and effective)
* Implement basic data augmentation
* Add pose validation checks
* Improve the confidence threshold system
* Add basic feature extraction if time permits


