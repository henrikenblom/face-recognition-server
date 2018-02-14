import face_recognition
import os
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from flask import Flask, request, jsonify
from flask_cors import CORS

MARGIN = 20
app = Flask(__name__)
CORS(app)


@app.route('/profile_image_upload', methods=['POST'])
def profile_image_upload():
    file = request.files['file']
    return detect_faces_in_image(file, request.form['name'])


def detect_faces_in_image(file_stream, filename):
    pil_image = Image.open(file_stream)
    pil_image.thumbnail((800, 800))
    image = np.array(pil_image)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    if not face_locations:
        return jsonify(status='NO_FACE')

    if len(face_locations) > 1:
        return jsonify(status='TOO_MANY_FACES')

    top, right, bottom, left = face_locations[0]
    full_image = ImageEnhance.Sharpness(ImageOps.autocontrast(Image.fromarray(image))).enhance(2)

    output_directory = "static/{}".format(filename)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_filename = "static/{}/{}.jpg".format(filename, time.time())

    bottom += (MARGIN * 3)
    right += MARGIN
    left -= MARGIN
    top -= MARGIN

    if bottom > full_image.height:
        bottom = full_image.height
    if right > full_image.width:
        right = full_image.width
    if left < 0:
        left = 0
    if top < 0:
        top = 0

    face_landmarks_list = face_recognition.face_landmarks(image[top:bottom, left:right])
    if not face_landmarks_list:
        return jsonify(status='NO_FULL_FACE')

    face_landmarks = face_landmarks_list[0]
    cropped_image = full_image.crop((left, top, right, bottom))
    d = ImageDraw.Draw(cropped_image, 'RGBA')

    d.line(face_landmarks['chin'], fill=(77, 182, 172, 180), width=1)

    d.line(face_landmarks['left_eyebrow'], fill=(77, 182, 172, 180), width=1)
    d.line(face_landmarks['right_eyebrow'], fill=(77, 182, 172, 180), width=1)

    d.line(face_landmarks['nose_bridge'], fill=(77, 182, 172, 180), width=1)
    d.line(face_landmarks['nose_tip'], fill=(77, 182, 172, 180), width=1)

    d.line(face_landmarks['top_lip'], fill=(77, 182, 172, 180), width=1)
    d.line(face_landmarks['bottom_lip'], fill=(77, 182, 172, 180), width=1)

    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(77, 182, 172, 180), width=1)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(77, 182, 172, 180), width=1)

    cropped_image.save(output_filename, 'jpeg')

    return jsonify(status='OK', url='http://titan.enblom.com/' + output_filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
