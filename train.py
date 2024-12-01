import os
import numpy as np
import tensorflow as tf
import uuid
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil
import wandb
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

def log_epoch_metrics(epoch, logs):
    wandb.log({
        "epoch": epoch + 1,
        "loss": logs["loss"],
        "accuracy": logs["accuracy"],
        "val_loss": logs["val_loss"],
        "val_accuracy": logs["val_accuracy"]
    })

def main():
    dataset = load_dataset("garythung/trashnet")
    prepare_data(dataset)
    image_size = 128
    batch_size = 32
    epoch = 20
    lr =  0.001

    wandb.init(
    project="cnn_trash_classifier",
    entity="jdarmawan-jd-justin",
    config={"learning_rate": lr, "epochs": epoch, "batch_size": batch_size} 
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=50.0,
        validation_split=0.2
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
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    epoch_logger = LambdaCallback(on_epoch_end=lambda epoch, logs: log_epoch_metrics(epoch, logs))

    model.fit(
        train_generator,
        epochs=epoch,
        validation_data=val_generator,
        callbacks=[epoch_logger]
    )

    loss, accuracy = model.evaluate(val_generator)
    wandb.log({"final_loss": loss, "final_accuracy": accuracy})
    wandb.save("cnn_trash_classifier.pth")


if __name__ == "__main__":
    main()

