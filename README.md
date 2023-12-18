# CIFAR-10 Image Classification

This project focuses on image classification using the CIFAR-10 dataset. It includes a Convolutional Neural Network (CNN) and a Transfer Learning approach using the VGG16 model.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Getting Started

1. Download and extract the CIFAR-10 dataset:
   ```bash
   python download_extract_data.py
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script to train the models:
   ```bash
   python main.py
   ```

## Project Structure

- `download_extract_data.py`: Script to download and extract the CIFAR-10 dataset.
- `main.py`: Main script to execute the project. It includes modularized functions for different tasks.
- `model.h5`: Saved CNN model.
- `transfer_model.h5`: Saved Transfer Learning model.
- `output/`: Folder containing output files (sample images, training history plot).
- `requirements.txt`: List of required Python packages.

## Results

The training results and accuracy metrics for both the CNN and Transfer Learning models are saved in the `output/output.txt` file.

Sample images and the training history plot can be found in the `output/` folder.

## Improving Accuracy

If you want to improve the accuracy of the models, consider experimenting with different hyperparameters, model architectures, learning rates, and data augmentation techniques. Early stopping is implemented to prevent overfitting.

## Dependencies

- Python 3.x
- TensorFlow
- Matplotlib
- NumPy
- tqdm

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
