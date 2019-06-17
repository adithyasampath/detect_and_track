##  go to folder object_tracking

# To generate the PROTOS
 make build

# To build docker image
 make docker

# To generate protos and build docker image
 make all

# To clean generated files
 make clean

# To modify TF Serving IP/Port (or) GRPC Port
    modify in ./config/config.py

# To run server (from docker)
    sudo xhost +
    export DISPLAY=:0.0
    docker run -it  --env DISPLAY=unix$DISPLAY --privileged -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --rm object_tracking:v1 

# To run client (from local machine)
    python3 ./src/client.py

# To run integration test
    python3 ./tests/tracker_test.py


