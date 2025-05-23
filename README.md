# AI-Based Bodybuilding Pose Classification

This project explores the application of deep learning for classifying competitive bodybuilding poses using computer vision. Two separate approaches were implemented and evaluated:

- **Custom CNN** – A convolutional neural network designed and trained from scratch using TensorFlow/Keras.
- **Transfer Learning CNN** – A model built on top of MobileNetV2 using pretrained ImageNet weights and fine-tuned for our dataset.

Each method is stored in its own folder with detailed documentation, code, training pipeline, and evaluation metrics:

- [`Custom CNN/`](./Custom%20CNN) – Fully custom model architecture.
- [`Transfer Learning CNN/`](./Transfer%20Learning%20CNN) – Transfer learning implementation with MobileNetV2.

> Refer to the README files in each folder for complete information about the models, results, and usage.


| Feature                     | Transfer Learning Model | Custom CNN Model |
|-----------------------------|----------------------------------------|--------------------------------|
| **Base Architecture**       | MobileNetV2 (Pre-trained)              | Custom-built CNN               |
| **Training Time**           | Shorter due to pre-trained layers      | Longer, trained from scratch   |
| **Data Requirement**        | Less data needed                      | Requires more data             |
| **Performance**             | Higher initial accuracy                | Potential for higher accuracy with sufficient data |
| **Flexibility**             | Limited to MobileNetV2 architecture    | Fully customizable             |
| **Complexity**              | Easier to implement                    | More complex implementation    |

