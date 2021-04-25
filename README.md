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

## Running the server (Containerized)

### Running a prebuilt container

This container is available on a repository at 
[quay.io](https://quay.io/repository/shenghaoyang/dlserver).

1. Start the Docker container.
    - (Optional) Create volumes to persist configuration and uploaded 
    inference and training samples:
    ```shell
    $ docker volume create dlserver_data
    $ docker volume create dlserver_config
    ```
    - Start the container (omit `--volume` commands if volumes are not desired)
    ```shell
    $ docker run -p 55221:55221/tcp \
        --volume dlserver_data:/var/lib/dlserver \
        --volume dlserver_config:/etc/dlserver/ 
        --name dlserver quay.io/shenghaoyang/dlserver:latest
    ```
    The container exposes TCP port 55221, which is used for gRPC communication.
    This can be remapped to another port if necessary.

    Podman can also be used to run this container.

6. Interact with the server using the client application (Android).

### Building your own container

The server *requires* Python 3.9.1 or above, and these installation
steps do not install any GPU-dependent packages.

1. Install the package using `flit` in a clean virtualenv: `flit install`.
    - Note that you may need to install `flit` in the virtualenv first.
2. Train the model using the steps outlined in the Jupyter notebook.
    - Note that this requires that the virtualenv is based on Python 3.9.1 or above.
    - Customize `server_container.ini`, changing the listening endpoint and 
    other parameters as necessary - paths specified in the configuration file
    should not be changed.
3. Build distribution artifacts.
    ```shell
    $ flit build
    ```
4. Build the Docker container.
    ```shell
    $ docker build -t dlserver:<tag> .
    ```
5. Start the Docker container.
    - (Optional) Create volumes to persist configuration and uploaded 
    inference and training samples:
    ```shell
    $ docker volume create dlserver_data
    $ docker volume create dlserver_config
    ```
    - Start the container (omit `--volume` commands if volumes are not desired)
    ```shell
    $ docker run -p 55221:55221/tcp \
        --volume dlserver_data:/var/lib/dlserver \
        --volume dlserver_config:/etc/dlserver/ 
        --name dlserver dlserver:<tag>
    ```
    The container exposes TCP port 55221, which is used for gRPC communication.
    This can be remapped to another port if necessary.

    Note that the port mapping would not make sense if the endpoint address
    in `server_container.ini` has been modified. In that case, modify the
    container start command line appropriately.

    Podman can also be used to run this container.

6. Interact with the server using the client application (Android).

## Running the server directly

0. Install the package using `flit` in a clean virtualenv: `flit install`.
    - Note that you may need to install `flit` in the virtualenv first.

1. Train the model using the steps outlined in the Jupyter notebook.
    - Note that, as above, this requires that the virtualenv is based off
    Python 3.9.1 or above.
    
2. Customize `server.ini` with appropriate paths and fill in the endpoint
   address you wish to use.
   
3. Run `dlserver server.ini` in the repository's root directory.

## Making requests to the server

### Android application

Follow the instructions in the Android app repository.

### Using the command line test client

A test client is included in this repository that reads samples in the form
of WAV files and sends them to the deep learning server.

The client expects a WAV file of a duration less than equal to a second
sampled at 16000Hz to be passed over STDIN.

The first argument must be the address & port of the server, e.g. '127.0.0.1:55221'.

For submitting retraining suggestions, a second argument specifying the label
(e.g. `up`, `down`, `left`, ...) of the sample needs to be provided.

If the training data has already been downloaded (by executing - up to, and including -
the data download cell in the training notebook) then samples in the
 `train/data/mini_speech_commads/` directory can be used for testing.

#### Containerized test client

If the server has already been started as a container (and assigned the name `dlserver`),
then the test client can be run by starting it within the container.

- For inference:
    ```shell
    $ docker exec -i dlserver dlserver_testclient 127.0.0.1:55221 \
        < train/data/mini_speech_commands/yes/32561e9e_nohash_0.wav
    ```

- For submitting a retraining suggestion:
    ```shell
    $ docker exec -i dlserver dlserver_testclient 127.0.0.1:55221 yes \
        < train/data/mini_speech_commands/yes/32561e9e_nohash_0.wav
    ```

#### Running the client directly

The test client can also be run directly. Note that Python 3.9.1 and above
is required.

1. Install the package using `flit` in a clean virtualenv: `flit install`.
    - Note that you may need to install `flit` in the virtualenv first.

2. Run the test client:

    - For inference:
        ```shell
        $ dlserver_testclient 127.0.0.1:55221 \
            < train/data/mini_speech_commands/yes/32561e9e_nohash_0.wav
        ```

    - For submitting a retraining suggestion:
        ```shell
        $ dlserver_testclient 127.0.0.1:55221 yes \
            < train/data/mini_speech_commands/yes/32561e9e_nohash_0.wav
        ```
