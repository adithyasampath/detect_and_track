FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

COPY ./config /home/object_tracking/config
COPY ./examples /home/object_tracking/examples
COPY ./generated /home/object_tracking/generated
COPY ./resources /home/object_tracking/resources
COPY ./src /home/object_tracking/src
COPY ./tests /home/object_tracking/tests
COPY ./utils /home/object_tracking/utils

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1 && \
    pip3 install -r /home/object_tracking/config/requirements.txt 

CMD ["python3","-u", "home/object_tracking/src/server.py" ]
