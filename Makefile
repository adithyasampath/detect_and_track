# Parameters
MKDIR=mkdir
SHELL:=/bin/bash
PYTHON=python3
DOCKRNAME=object_tracking
DOCKRVER=v1
HTTP_PROXY=http://10.158.100.6:8080 

all: clean build docker  

build:
	$(MKDIR) -p generated
	$(PYTHON) -m grpc_tools.protoc  -I ./protos --python_out=./generated --grpc_python_out=./generated object_detector.proto

docker:
	docker build -t $(DOCKRNAME):$(DOCKRVER) --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy=$(HTTP_PROXY) .

clean:
	rm -rf generated
