import face_recognition
from PIL import Image
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/profile_image_upload', methods=['POST'])
def profile_image_upload():
    file = request.files['file']
    return detect_faces_in_image(file, request.form['name'])


def detect_faces_in_image(file_stream, filename):
    image = face_recognition.load_image_file(file_stream)

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    if not face_locations:
        return 'NO_FACE'

    if len(face_locations) > 1:
        return 'TOO_MANY_FACES'

    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    outputfilename = 'static/' + filename + '.jpg'
    pil_image.save(outputfilename, 'jpeg')
    return 'http://titan.enblom.com/' + outputfilename


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
