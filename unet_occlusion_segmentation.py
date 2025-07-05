import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set up paths for the dataset
BASE_PATH = r"C:\NNDL"
MASK_PATH = os.path.join(BASE_PATH, "Image Mask")
ORIGINAL_PATH = os.path.join(BASE_PATH, "Surgical Masked Image")
SAVED_MODEL_PATH = "object_detection_model.h5"

# Constants
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 16
EPOCHS = 1

# Function to load and preprocess images (for ImageDataGenerator)
def load_image(img_path, mask_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    
    mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
    mask = img_to_array(mask) / 255.0  # Normalize to [0, 1]
    
    return img, mask

# Create a custom data generator class that yields pairs of images and masks
class ImageMaskGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, shuffle=True, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Get the list of paths for the current batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in batch_indexes]
        batch_mask_paths = [self.mask_paths[k] for k in batch_indexes]
        
        # Load and return a batch of images and masks
        images, masks = self.__data_generation(batch_image_paths, batch_mask_paths)
        return images, masks

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_paths, batch_mask_paths):
        # Generate a batch of images and masks
        images = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        masks = np.empty((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

        for i, (img_path, mask_path) in enumerate(zip(batch_image_paths, batch_mask_paths)):
            img, mask = load_image(img_path, mask_path)
            
            # Data augmentation (if enabled)
            if self.augment:
                img, mask = self.augment_data(img, mask)

            images[i,] = img
            masks[i,] = mask

        return images, masks

    def augment_data(self, img, mask):
        # Perform augmentation here if necessary (you can use ImageDataGenerator's augmentation)
        # For example, randomly flip the image and mask horizontally or vertically
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)

        return img, mask

# Function to create dataset (image paths and mask paths)
def create_dataset(original_dir, mask_dir):
    image_paths = [os.path.join(original_dir, img_name) for img_name in os.listdir(original_dir)]
    mask_paths = [os.path.join(mask_dir, img_name) for img_name in os.listdir(mask_dir)]

    return image_paths, mask_paths

# Load dataset
print("Loading dataset...")
image_paths, mask_paths = create_dataset(ORIGINAL_PATH, MASK_PATH)
print(f"Dataset loaded: {len(image_paths)} images")

# Split the dataset into training and validation sets
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Create training and validation generators
train_generator = ImageMaskGenerator(train_image_paths, train_mask_paths, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_generator = ImageMaskGenerator(val_image_paths, val_mask_paths, batch_size=BATCH_SIZE, shuffle=False)

# Build U-Net for object detection
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate

def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(shape=input_shape)

    # Encoder (Contracting Path)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # MaxPooling for downsampling

    # Bottleneck (Center of the network)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)

    # Decoder (Expansive Path) - 1 Layer with Conv2DTranspose for upsampling
    u3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same")(c2)
    merge3 = concatenate([u3, c1], axis=3)  # Skip connection
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(merge3)
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(c3)

    # Output layer (binary segmentation)
    outputs = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(c3)

    model = Model(inputs, outputs)
    return model


# Build the model
model = build_unet()
model.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=["accuracy"])
model.summary()

# Train the model
print("Training the model...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    shuffle=True
)

# Save the trained model
model.save(SAVED_MODEL_PATH)
print(f"Model saved to {SAVED_MODEL_PATH}")

# Visualize results (using a small batch of predictions)
def visualize_results(model, dataset, num_samples=5):
    sample_images, sample_masks = next(iter(dataset))
    preds = model.predict(sample_images)

    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(sample_images[i])
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(sample_masks[i].squeeze(), cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(preds[i].squeeze(), cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Visualize predictions on a small batch
visualize_results(model, val_generator)
