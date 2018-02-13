FROM colemurray/medium-facenet-tutorial:latest-gpu

ADD $PWD/requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -U -r /requirements.txt

WORKDIR /app
RUN mkdir /app/static

ADD $PWD/*.py /app

EXPOSE 3000

CMD ["python3", "face-recognition-server.py"]