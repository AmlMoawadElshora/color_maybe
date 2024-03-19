from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from collections import Counter

app = Flask(__name__)

COLOR_RANGES = {
    'red': ([0, 100, 100], [10, 255, 255]),
    'green': ([36, 25, 25], [86, 255, 255]),
    'blue': ([100, 150, 0], [140, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'orange': ([10, 100, 100], [20, 255, 255]),
    'purple': ([130, 50, 50], [170, 255, 255]),
    'pink': ([140, 100, 100], [170, 255, 255]),
    'cyan': ([90, 100, 100], [110, 255, 255]),
    'brown': ([0, 50, 50], [20, 255, 255]),
    'white': ([0, 0, 200], [180, 25, 255]),
    'black': ([0, 0, 0], [180, 255, 30]),
    'gray': ([0, 0, 80], [180, 40, 200]),
    'light_blue': ([90, 50, 50], [130, 255, 255]),
    'magenta': ([130, 100, 100], [170, 255, 255]),
    'lime': ([40, 100, 100], [70, 255, 255]),
    'olive': ([20, 50, 50], [40, 255, 255]),
    'maroon': ([0, 100, 50], [10, 255, 200]),
    'teal': ([80, 100, 50], [100, 255, 200]),
    'navy': ([100, 100, 50], [140, 255, 200]),
    'sky_blue': ([100, 50, 50], [130, 255, 255]),
    'violet': ([130, 50, 50], [170, 255, 200]),
    'indigo': ([100, 100, 0], [140, 255, 100]),
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

    # Initialize a dictionary to store the count of pixels for each color
    color_counts = {color_name: 0 for color_name in COLOR_RANGES}

    # Loop through each color range and count pixels
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
        color_counts[color_name] = cv2.countNonZero(mask)

    # Select the color with the highest pixel count
    dominant_color = max(color_counts, key=color_counts.get)

    # If no color has a significant pixel count, return 'Can't detect color'
    if color_counts[dominant_color] < 1000:
        dominant_color = "Can't detect color"

    return jsonify({'color': dominant_color})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
