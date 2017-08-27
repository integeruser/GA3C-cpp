# GA3C-cpp
This repository contains a fast C++ multithreaded implementation of the [asynchronous advantage actor-critic](https://arxiv.org/abs/1602.01783) (A3C) algorithm based on the [GA3C architecture](http://research.nvidia.com/publication/reinforcement-learning-through-asynchronous-advantage-actor-critic-gpu). I tested it on the `CartPole-v0` [OpenAI Gym](https://github.com/openai/gym) environment using the [gym-uds-api](https://github.com/integeruser/gym-uds-api), taking model configuration and parameters from [this other implementation](https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py).

## Requisites
This project requires any recent version of Google's [TensorFlow](https://www.tensorflow.org/). The steps for building it from sources are well explained in the [official documentation](https://www.tensorflow.org/install/install_sources); as a quick recap, you are required to:
1. Clone TensorFlow from the [official repository](https://github.com/tensorflow/tensorflow)
2. Switch to a stable release using e.g. `git checkout r1.2`
4. Run `./configure`
5. Build TensorFlow as a shared library using `bazel build --config=opt //tensorflow:libtensorflow_cc.so`
6. Copy `bazel-bin/tensorflow/libtensorflow_cc.so` to any directory searched by the run-time loader, e.g. `/usr/local/lib/`

## Installation
1. Recursively clone this repository, i.e. `git clone --recursive https://github.com/integeruser/GA3C-cpp.git`
2. Run `./build.sh <path_to_tensorflow_repo>` to automatically copy the necessary C++ headers and sources into `third-party/`

## Usage
1. Generate the TensorFlow model:
```
/o/GA3C-cpp $ python3 model.py generate
```
2. Train the agent:
```
/o/GA3C-cpp $ make
mkdir -p bin
c++ -std=c++11 -march=native -O2 -pthread -l tensorflow_cc -o bin/GA3C \
    -I include -I third-party -I third-party/gym-uds-api/binding-cpp/include -I third-party/third_party \
    third-party/gym-uds-api/binding-cpp/src/gym-uds.cc third-party/gym-uds-api/binding-cpp/src/gym-uds.pb.cc \
    src/model.cc src/main.cc
/o/GA3C-cpp $ ./start-gym-uds-servers.sh
/o/GA3C-cpp $ bin/GAC3
```
3. The updated parameters of the model are saved back to disk. Test the trained agent using:
```
/o/GA3C-cpp $ python3 model.py test
```

Keep in mind to regenerate the empty model using the Python script before training again.
