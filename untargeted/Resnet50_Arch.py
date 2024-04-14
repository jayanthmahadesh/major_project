# import the necessary packages
from pyimagesearch.datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
import numpy as np
import os
from PIL import Image
# load CIFAR-10 dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading CIFAR-10 dataset...")

def load_lfw_dataset(dataset_path):
    dataX = []  
    dataY = [] 

    classes = os.listdir(dataset_path)
    num_classes = len(classes)

    for i, person_dir in enumerate(classes):
        person_path = os.path.join(dataset_path, person_dir)
        image_count = 0
        print(person_path)
        print("____________________________")
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                print(image_path)
                image = Image.open(image_path).convert('RGB')  
                image = image.resize((32, 32)) 
                image_array = np.asarray(image)
                dataX.append(image_array)
                dataY.append(i)  # Assign the class index as a label

                image_count += 1
                # if num_images_per_class > 0 and image_count >= num_images_per_class:
                #     break  # Limit the number of images per class 

    # Convert to NumPy arrays
    trainX = np.array(dataX)  
    trainY = np.array(dataY)

    # Simulate a test set 
    testX = trainX[-500:]  
    testY = trainY[-500:]

    return trainX, trainY, testX, testY 

# (trainX, trainY), (testX, testY) = cifar10.load_data()
trainX, trainY, testX, testY = load_lfw_dataset("/Users/jayanthmahadeshkps/Desktop/archive/lfw-deepfunneled/lfw-deepfunneled")
print("____")
print(len(trainX)," ",len(trainY)," ",len(testX)," ",len(testY))
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
# one-hot encode our labels
trainY = to_categorical(trainY, 0)
testY = to_categorical(testY, 0)
# initialize ResNet50 model
print("[INFO] loading pre-trained ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# freeze the base model layers
base_model.trainable = False
dataset_path = "/Users/jayanthmahadeshkps/Desktop/archive/lfw-deepfunneled/lfw-deepfunneled"
classes = os.listdir(dataset_path)
num_classes = len(classes)
print(num_classes)




# add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5750, activation='softmax')(x)

# create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# compile the model
print("[INFO] compiling model...")
opt = Adam(0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the ResNet model on CIFAR-10
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=1, verbose=1)

# make predictions on the testing set for the model trained on
# non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# generate a set of adversarial examples with FGSM
print("[INFO] generating adversarial examples with FGSM...\n")
# Your code for generating adversarial examples can be added here

# generate a set of adversarial examples with FGSM
print("[INFO] generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(model, len(testX), testX, testY, (32, 32, 3), eps=0.1))

# re-evaluate the model on the adversarial images
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# lower the learning rate and re-compile the model for fine-tuning
print("[INFO] re-compiling model...")
opt = Adam(0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# fine-tune the model on the adversarial examples
print("[INFO] fine-tuning network on adversarial examples...")
model.fit(advX, advY, batch_size=64, epochs=1, verbose=1)

# evaluate the model on normal testing images again after fine-tuning
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# evaluate the model on adversarial images again after fine-tuning
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))
