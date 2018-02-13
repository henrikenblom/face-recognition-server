FROM colemurray/medium-facenet-tutorial:latest-gpu

RUN apt-get install git
ADD $PWD/requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -U -r /requirements.txt
RUN pip3 install git+https://github.com/ageitgey/face_recognition_models

WORKDIR /app
RUN mkdir /app/static

ADD $PWD/*.py /app

EXPOSE 3000

CMD ["python3", "face-recognition-server.py"]