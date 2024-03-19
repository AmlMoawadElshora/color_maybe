from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from collections import Counter

app = Flask(__name__)

COLOR_VALUES = {
    'red': ([0, 255, 255]),
    'green': ([60, 255, 255]),
    'blue': ([120, 255, 255]),
    'yellow': ([30, 255, 255]),
    'orange': ([15, 255, 255]),
    'purple': ([150, 255, 255]),
    'pink': ([170, 255, 255]),
    'cyan': ([90, 255, 255]),
    'brown': ([10, 255, 165]),
    'white': ([0, 0, 255]),
    'black': ([0, 0, 0]),
    'gray': ([0, 0, 128]),
    'light_blue': ([100, 255, 255]),
    'magenta': ([140, 255, 255]),
    'lime': ([70, 255, 255]),
    'olive': ([40, 255, 128]),
    'maroon': ([0, 255, 128]),
    'teal': ([80, 255, 128]),
    'navy': ([120, 255, 128]),
    'sky_blue': ([120, 255, 255]),
    'violet': ([170, 255, 128]),
    'indigo': ([100, 255, 128]),
}


@app.route('/')
def upload_form():
    return render_template('upload_form.html')

import math

@app.route('/detect_color', methods=['POST'])
def detect_color():
    # Get the image file from the request
    image_file = request.files['image']

    # Read the image
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Initialize a dictionary to store the distance for each color
    color_distances = {}

    # Loop through each color value and calculate distance
    for color_name, hsv_value in COLOR_VALUES.items():
        # Convert HSV value to numpy array
        hsv_value = np.array([hsv_value])
        # Calculate distance using Euclidean distance formula
        distance = np.linalg.norm(img_hsv - hsv_value, axis=-1)
        # Store the mean distance
        color_distances[color_name] = np.mean(distance)

    # Select the color with the minimum distance
    dominant_color = min(color_distances, key=color_distances.get)

    return jsonify({'color': dominant_color})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
