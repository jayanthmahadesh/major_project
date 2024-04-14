# USAGE
# python3 gaussian_defence.py --input adversarial.png --output defence_adversarial
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import argparse
import cv2

def preprocess_image(image):
	# swap color channels, resize the input image, and add a batch
	# dimension
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)

	# return the preprocessed image
	return image

def add_gaussian_noise(image, noise_std=0.1):
    # Assumes image might be 3D or 4D
    if len(image.shape) == 3:
        row, col, ch = image.shape
    else:
        _, row, col, ch = image.shape  # Unpack the batch dimension

    mean = 0.0
    gauss = np.random.normal(mean, noise_std, (row, col, ch))

    # If the original image was 4D, create a suitable noisy_image
    if len(image.shape) == 4:
        noisy_image = np.zeros_like(image)
        for i in range(image.shape[0]):  # Iterate over batch dimension
            noisy_image[i] = image[i] + gauss
    else:
        noisy_image = image + gauss

    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image
def apply_input_transformations(image, transformations=["jpeg", "bitdepth"]):
    for transformation in transformations:
        if transformation == "jpeg":
            image = tf.image.decode_jpeg(tf.image.encode_jpeg(image, quality=80), channels=3)
        elif transformation == "bitdepth":
            image = tf.image.convert_image_dtype(image, tf.uint8)  # Cast to 8-bit
        elif transformation == "denoise":
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) 
            image = tf.convert_to_tensor(image) 
        else:
            raise ValueError(f"Invalid transformation: {transformation}")
    image = np.clip(image, 0, 255)
    return image
# EPS = 2 / 255.0
# LR = 0.1

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to original input image")
args = vars(ap.parse_args())


print("[INFO] loading image...")
image = cv2.imread(args["input"])
# image = preprocess_image(image)
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")

# # initialize optimizer and loss function
# optimizer = Adam(learning_rate=LR)
# sccLoss = SparseCategoricalCrossentropy()

defence_adverImage = apply_input_transformations(image)
# defence_adverImage = preprocess_image(defence_adverImage)
# defence_adverImage = add_gaussian_noise(defence_adverImage,noise_std=10)
defence_adverImage = preprocess_image(defence_adverImage)
# defence_adverImage = image
print("[info] writing output image")
# cv2.imwrite(args["output"], defence_adverImage)


# run inference with this adversarial example, parse the results,
# and display the top-1 predicted result
print("[INFO] running inference on the adversarial example...")
preprocessedImage = preprocess_input(defence_adverImage)
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]
print(predictions)
label = predictions[0][1]
confidence = predictions[0][2] * 100
print("[INFO] label: {} confidence: {:.2f}%".format(label,confidence))

# draw the top-most predicted label on the adversarial image along
# with the confidence score
# text = "{}: {:.2f}%".format(label, confidence)
# cv2.putText(defence_adverImage, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)

# show the output image
# cv2.imshow("Output", defence_adverImage)
# cv2.waitKey(0)