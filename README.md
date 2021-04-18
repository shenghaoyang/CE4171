# NTU IOT course project - Deep Learning Server

## Introduction

This is the repository hosting the deep learning server portion of the
course project, meant to accept inference and training requests via RPC.

The deep learning task that this server tackles is covered in 
[this](https://www.tensorflow.org/tutorials/audio/simple_audio)
 TensorFlow tutorial.

## Model training

The training steps used to build the server's model are outlined in a
Jupyter notebook, stored at `train/training.ipynb`.

Accuracy and other model metrics can also be found in the same notebook.

After executing all notebook cells, the model will be trained and saved at
`train/saved/audiorecog`.

## Implementation

The server is implemented using Python 3 and it communicates with clients
using gRPC. It leans heavily on Python's asyncio asynchronous programming API
to:

- Serve multiple concurrent clients
- Perform training in the background
- Switch models in the background

## Running the server (Docker)

The server *requires* Python 3.9.1 or above, and these installation
steps do not install any GPU-dependent packages.

0. Install the package using `flit` in a clean virtualenv: `flit install`.
    - Note that you may need to install `flit` in the virtualenv first.
1. Train the model using the steps outlined in the Jupyter notebook.
    - Customize `server_container.ini` as necessary, changing the listening
      endpoint and other parameters as necessary.
    - Paths specified in the configuration file should not be changed.
2. Build distribution artifacts.
    ```shell
    $ flit build
    ```
2. Build the Docker container.
    ```shell
    $ docker build -t dlserver:<tag> .
    ```
3. Start the Docker container.
    - (Optional) Create volumes to persist configuration and uploaded 
    inference and training samples:
    ```shell
    $ docker volume create dlserver_data
    $ docker volume create dlserver_config
    ```
    - Start the container (omit `--volume` commands if volumes are not desired)
    ```shell
    $ docker run docker run -p 55221:55221/tcp \
        --volume dlserver_data:/var/lib/dlserver \
        --volume dlserver_config:/etc/dlserver/ 
        --name dlserver dlserver:<tag>
    ```
    The container exposes TCP port 55221, which is used for gRPC communication.
    This can be remapped to another port if necessary.

4. Interact with the server using the client application.
