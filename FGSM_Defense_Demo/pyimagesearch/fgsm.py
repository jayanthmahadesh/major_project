# import the necessary packages
from tensorflow.keras.losses import MSE
import tensorflow as tf
import numpy as np

def generate_image_adversary(model, image, label, eps=8 / 255.0):
	# cast the image
	image = tf.cast(image, tf.float32)
	# record our gradients
	with tf.GradientTape() as tape:
		# explicitly indicate that our image should be tacked for
		# gradient updates
		tape.watch(image)
		# use our model to make predictions on the input image and
		# then compute the loss
		pred = model(image)
		# loss = MSE(label, pred)
		# Compute cross-entropy loss
		loss = tf.keras.losses.categorical_crossentropy([label], pred)
	# calculate the gradients of loss with respect to the image, then
	# compute the sign of the gradient
	gradient = tape.gradient(loss, image)
	signedGrad = tf.sign(gradient)
	# construct the image adversary
	adversary = (image + (signedGrad * eps)).numpy()
	# return the image adversary to the calling function
	# adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
	return adversary