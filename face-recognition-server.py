import face_recognition
import os
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from flask import Flask, request, jsonify
from flask_cors import CORS

MARGIN = 100
LANDMARK_FILL = (77, 182, 172, 200)
LANDMARK_WIDTH = 3
ORIGINAL_CONSTRAINTS = (1000, 1000)
THUMBNAIL_CONSTRAINTS = (400, 400)
app = Flask(__name__)
CORS(app)


@app.route('/profile_image_upload', methods=['POST'])
def profile_image_upload():
    file = request.files['file']
    return detect_faces_in_image(file, request.form['name'])


def detect_faces_in_image(file_stream, filename):
    pil_image = Image.open(file_stream)
    pil_image.thumbnail(ORIGINAL_CONSTRAINTS)
    image = np.array(pil_image)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    if not face_locations:
        return jsonify(status='NO_FACE')

    if len(face_locations) > 1:
        return jsonify(status='TOO_MANY_FACES')

    top, right, bottom, left = face_locations[0]
    full_image = ImageEnhance.Sharpness(ImageOps.autocontrast(Image.fromarray(image))).enhance(3)

    output_directory = "static/{}".format(filename)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_filename = "static/{}/{}.jpg".format(filename, int(time.time() * 1000))

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
    pre_cropped_image = full_image.crop((left, top, right, bottom))
    d = ImageDraw.Draw(pre_cropped_image, 'RGBA')

    d.line(face_landmarks['chin'], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)

    d.line(face_landmarks['left_eyebrow'], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)
    d.line(face_landmarks['right_eyebrow'], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)

    d.line(face_landmarks['nose_bridge'], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)
    d.line(face_landmarks['nose_tip'], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)

    d.line(face_landmarks['top_lip'], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)
    d.line(face_landmarks['bottom_lip'], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)

    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=LANDMARK_FILL, width=LANDMARK_WIDTH)

    scale = (face_landmarks['chin'][8][1] - face_landmarks['nose_bridge'][0][1]) / 100
    bottom = face_landmarks['chin'][8][1] + (10 * scale)
    top = bottom - (50 * scale)
    left = face_landmarks['nose_bridge'][0][0] - (20 * scale)
    right = face_landmarks['nose_bridge'][0][0] + (20 *scale)

    cropped_image = pre_cropped_image.crop((left, top, right, bottom))

    cropped_image.thumbnail(THUMBNAIL_CONSTRAINTS)
    cropped_image.save(output_filename, 'jpeg')

    return jsonify(status='OK', url='http://titan.enblom.com/' + output_filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
