# GA3C-tf-cpp
This repository contains a fast C++ multithreaded implementation of ["Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU"](http://research.nvidia.com/publication/reinforcement-learning-through-asynchronous-advantage-actor-critic-gpu), tested on some [OpenAI Gym](https://github.com/openai/gym) environments.

## Requisites
This projects requires any recent version of Google's [TensorFlow](https://www.tensorflow.org/) and the [gym-uds-api](https://github.com/integeruser/GA3C-tf-cpp.git).

## Installation
These steps assume you have already built Tensorflow from sources and downloaded gym-uds-api.

1. Clone or download this repository, e.g. `git clone https://github.com/integeruser/GA3C-tf-cpp.git`
2. Run `./build.sh` to copy the necessary C++ headers and sources from TensorFlow and gym-uds-api into `include/third_party`

## Usage
1. Build the binary:
```
/o/GA3C-tf-cpp $ make
clang++ -std=c++11 -O2 -pthread -o GAC3 \
	-I include -I include/third_party -l tensorflow_cc \
	src/gym.cc src/gym-uds.pb.cc src/model.cc src/main.cc
```
2. Run it:
```
/o/GA3C-tf-cpp $ ./GAC3
gym::Environment::connect: Connection refused
```
