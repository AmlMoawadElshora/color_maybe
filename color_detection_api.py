from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

COLOR_RANGES = {
    'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
    'green': (np.array([36, 25, 25]), np.array([86, 255, 255])),
    'blue': (np.array([100, 150, 0]), np.array([140, 255, 255])),
    'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
    'orange': (np.array([10, 100, 100]), np.array([20, 255, 255])),
    'purple': (np.array([130, 50, 50]), np.array([170, 255, 255])),
    'pink': (np.array([140, 100, 100]), np.array([170, 255, 255])),
    'cyan': (np.array([90, 100, 100]), np.array([110, 255, 255])),
    'brown': (np.array([0, 50, 50]), np.array([20, 255, 255])),
    'white': (np.array([0, 0, 200]), np.array([180, 25, 255])),
    'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
    'gray': (np.array([0, 0, 80]), np.array([180, 40, 200])),
    'light_blue': (np.array([90, 50, 50]), np.array([130, 255, 255])),
    'magenta': (np.array([130, 100, 100]), np.array([170, 255, 255])),
    'lime': (np.array([40, 100, 100]), np.array([70, 255, 255])),
    'olive': (np.array([20, 50, 50]), np.array([40, 255, 255])),
    'maroon': (np.array([0, 100, 50]), np.array([10, 255, 200])),
    'teal': (np.array([80, 100, 50]), np.array([100, 255, 200])),
    'navy': (np.array([100, 100, 50]), np.array([140, 255, 200])),
    'sky_blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
    'violet': (np.array([130, 50, 50]), np.array([170, 255, 200])),
    'indigo': (np.array([100, 100, 0]), np.array([140, 255, 100])),
}


@app.route('/')
def upload_form():
    return render_template('upload_form.html')

@app.route('/detect_colors', methods=['POST'])
def detect_colors():
    # Get the image file from the request
    image_file = request.files['image']

    # Read the image
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect colors
    colors = []
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(img_hsv, lower, upper)
        count = cv2.countNonZero(mask)
        if count > 0:
            colors.append(color_name)

    return jsonify({'colors': colors})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
