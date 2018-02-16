FROM nvidia/cuda:9.1-devel

ADD $PWD/requirements.txt /requirements.txt

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install git -y
RUN apt-get install cmake -y
RUN apt-get install python3 -y 
RUN apt-get install python3-pip -y
RUN apt-get install libopenblas-dev liblapack-dev
RUN pip3 install --upgrade pip
RUN pip3 install -U -r /requirements.txt

RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7.0.5 /usr/local/cuda/lib64/libcudnn.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7.0.5 /usr/local/cuda/lib64/libcudnn.so.7
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7.0.5 /usr/local/cuda/lib64/libcudnn.so.7.0.5
RUN git clone https://github.com/davisking/dlib
WORKDIR dlib
RUN python3 setup.py install

RUN pip3 install git+https://github.com/ageitgey/face_recognition_models

WORKDIR /app
RUN mkdir /app/static

ADD $PWD/*.py /app

EXPOSE 3000

CMD ["python3", "face-recognition-server.py"]