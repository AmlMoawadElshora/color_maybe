from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

COLOR_RANGES = {
    'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
    'green': (np.array([36, 25, 25]), np.array([86, 255, 255])),
    'blue': (np.array([100, 150, 0]), np.array([140, 255, 255])),
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
    app.run(debug=True)
