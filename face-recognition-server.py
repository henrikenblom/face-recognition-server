from flask import Flask
import face_recognition
from PIL import Image

app = Flask(__name__)


@app.route('/profile_image_upload', methods=['POST'])
def profile_image_upload():
    return 'Hello World!'


def detect_faces_in_image(file_stream):
    image = face_recognition.load_image_file(file_stream)

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
