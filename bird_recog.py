import tensorflow as tf
import os
from PIL import Image
import numpy as np

import image_handling

# Remove some warnings from output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

graph_def = tf.GraphDef()
labels = []

# Import the TF graph
with tf.gfile.FastGFile('model-birds.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels
with open('labels-birds.txt', 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

# Load the test images from a file
imageFile = 'bird-test.jpg'
image = Image.open(imageFile)

# Update orientation based on EXIF tags, if the file has orientation info
image = image_handling.update_orientation(image)

# Convert to OpenCV (Open Source Computer Vision) format
image = image_handling.convert_to_opencv(image)

# If the image has either w or h greater than 1600 we resize it down respecting
# aspect ratio such that the largest dimension is 1600
image = image_handling.resize_down_to_1600_max_dim(image)

# We next get the largest center square
h, w = image.shape[:2]
min_dim = min(w,h)
max_square_image = image_handling.crop_center(image, min_dim, min_dim)

# Resize that square down to 256x256
augmented_image = image_handling.resize_to_256_square(max_square_image)

# The compact models have a network size of 227x227, the model requires this size
network_input_size = 227

# Crop the center for the specified network_input_Size
augmented_image = image_handling.crop_center(augmented_image, network_input_size, network_input_size)

# These names are part of the model and cannot be changed.
output_layer = 'loss:0'
input_node = 'Placeholder:0'

with tf.Session() as sess:
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    predictions = sess.run(prob_tensor, {input_node: [augmented_image] })

    # Print the highest probability label
    highest_probability_index = np.argmax(predictions)
    print('Classified as: ' + labels[highest_probability_index])
    print()

    # And print out each of the results mapping labels with their probabilities
    label_index = 0
    for p in predictions[0]:
        truncated_probablity = np.float64(np.round(p,8))
        print (labels[label_index], truncated_probablity)
        label_index += 1

