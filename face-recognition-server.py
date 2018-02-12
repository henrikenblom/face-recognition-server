from flask import Flask
import face_recognition


app = Flask(__name__)


@app.route('/profile_image_upload', methods=['POST'])
def profile_image_upload():
    return 'Hello World!'

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("biden.jpg")

# Find all the faces in the image using a pre-trained convolutional neural network.
# This method is more accurate than the default HOG model, but it's slower
# unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
# this will use GPU acceleration and perform well.
# See also: find_faces_in_picture.py
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
