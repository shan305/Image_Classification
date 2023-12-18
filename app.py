import ssl
from urllib.request import urlretrieve
import tarfile
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()

def download_extract_data(url, filename):
    download_path = Path(".") / filename

    # Download the file
    urlretrieve(url, download_path)

    # Extract the contents
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall()

def display_sample_images(train_images, train_labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.savefig('output/sample_images.png')
    plt.close()

def data_augmentation(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    datagen.fit(train_images)

    augmented_images, augmented_labels = next(datagen.flow(train_images, train_labels, batch_size=len(train_images)))
    train_images = np.concatenate([train_images, augmented_images])
    train_labels = np.concatenate([train_labels, augmented_labels])

    return train_images, train_labels

def build_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10, batch_size=64):
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels))

    return history

def build_transfer_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model_transfer = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model_transfer

def evaluate_model(model, test_images, test_labels, model_type):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy ({model_type}): {test_acc}')

def visualize_training_history(history, history_transfer):
    plt.plot(history.history['accuracy'], label='CNN training accuracy')
    plt.plot(history.history['val_accuracy'], label='CNN validation accuracy')
    plt.plot(history_transfer.history['accuracy'], label='Transfer Learning training accuracy')
    plt.plot(history_transfer.history['val_accuracy'], label='Transfer Learning validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('output/training_history.png')
    plt.close()

def save_model(model, model_filename):
    model.save(model_filename)

def main():
    # Redirect print statements to a file
    sys.stdout = Logger('output/output.txt')

    # Download and extract CIFAR-10 dataset
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    download_extract_data(url, filename)

    # Load and preprocess the data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Display some sample images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    display_sample_images(train_images, train_labels, class_names)

    # Data Augmentation
    train_images, train_labels = data_augmentation(train_images, train_labels)

    # Build the CNN Model
    cnn_model = build_cnn_model()

    # Compile the CNN Model
    compile_model(cnn_model)

    # Train the CNN Model
    history = train_model(cnn_model, train_images, train_labels, test_images, test_labels)

    # Transfer Learning
    model_transfer = build_transfer_model()

    # Compile the Transfer Learning Model
    compile_model(model_transfer)

    # Train the Transfer Learning Model
    history_transfer = train_model(model_transfer, train_images, train_labels, test_images, test_labels)

    # Evaluate the Models
    evaluate_model(cnn_model, test_images, test_labels, 'CNN')
    evaluate_model(model_transfer, test_images, test_labels, 'Transfer Learning')

    # Visualize Training History
    visualize_training_history(history, history_transfer)

    # Save the Models
    save_model(cnn_model, 'model.h5')
    save_model(model_transfer, 'transfer_model.h5')

    # Close the Logger
    sys.stdout.close()

# Run the main function
if __name__ == "__main__":
    main()
