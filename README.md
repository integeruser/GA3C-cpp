# GA3C-tf-cpp
This repository contains a fast C++ multithreaded implementation of ["Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU"](http://research.nvidia.com/publication/reinforcement-learning-through-asynchronous-advantage-actor-critic-gpu), tested on some [OpenAI Gym](https://github.com/openai/gym) environments using the [gym-uds-api](https://github.com/integeruser/gym-uds-api).

## Requisites
This project requires any recent version of Google's [TensorFlow](https://www.tensorflow.org/). The steps for building it from sources are well explained in the [official documentation](https://www.tensorflow.org/install/install_sources); as a quick recap, you are required to:
1. Clone TensorFlow from the [official repository](https://github.com/tensorflow/tensorflow)
2. Switch to a stable release using e.g. `git checkout r1.2`
4. Run `./configure`
5. Build TensorFlow as a shared library using `bazel build --config=opt //tensorflow:libtensorflow_cc.so`
6. Copy `bazel-bin/tensorflow/libtensorflow_cc.so` to any directory searched by the run-time loader, e.g. `/usr/local/lib/`

## Installation
1. Recursively clone this repository, i.e. `git clone --recursive https://github.com/integeruser/GA3C-tf-cpp.git`
2. Run `./build.sh <path_to_tensorflow_repo>` to copy the necessary C++ headers and sources

## Usage
1. Build the binary:
```
/o/GA3C-tf-cpp $ make
clang++ -std=c++11 -O2 -pthread -o GAC3 \
	-I include -I include/third_party -l tensorflow_cc \
	src/gym.cc src/gym-uds.pb.cc src/model.cc src/main.cc
```
2. Run it
