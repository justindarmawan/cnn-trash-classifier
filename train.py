import os
import numpy as np
import tensorflow as tf
import uuid
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil
from datasets import load_dataset

def prepare_data(dataset):
    base_path = "./trashnet_data"
    os.makedirs(base_path, exist_ok=True)

    for split in dataset.keys():
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)

        for data in dataset[split]:
            unique_id = uuid.uuid4()
            label = data['label']
            image = data['image']
            label_path = os.path.join(split_path, str(label))
            os.makedirs(label_path, exist_ok=True)
            
            image.save(os.path.join(label_path, f"{unique_id}.jpg"))

def predict_image(image_path, model, class_indices):
    image = load_img(image_path, target_size=(image_size, image_size))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_name = list(class_indices.keys())[np.argmax(prediction)]
    match class_name:
        case '0':
            return 'cardboard'
        case '1':
            return 'glass'
        case '2':
            return 'metal'
        case '3':
            return 'paper'
        case '4':
            return 'plastic'
        case '5':
            return 'trash'


def main():
    dataset = load_dataset("garythung/trashnet")
    prepare_data(dataset)
    image_size = 128
    batch_size = 32

    # Augmentasi untuk dataset training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Memisahkan data validasi
    )

    train_generator = train_datagen.flow_from_directory(
        './trashnet_data/train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        './trashnet_data/train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )



    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')  # Output sesuai jumlah kelas
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator
    )

    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation Accuracy: {accuracy:.2f}")


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()
    # class_indices = train_generator.class_indices
    # print(predict_image('./trashnet_data/test/trash.jpg', model, class_indices))


if __name__ == "__main__":
    main()

