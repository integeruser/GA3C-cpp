# GA3C-cpp
This repository contains a fast C++ multithreaded implementation of the [asynchronous advantage actor-critic](https://arxiv.org/abs/1602.01783) (A3C) algorithm based on the [GA3C architecture](http://research.nvidia.com/publication/reinforcement-learning-through-asynchronous-advantage-actor-critic-gpu). I tested it on the `CartPole-v0` [OpenAI Gym](https://github.com/openai/gym) environment using [TensorFlow](https://www.tensorflow.org/) and the [gym-uds-api](https://github.com/integeruser/gym-uds-api), taking model configuration and parameters from [this other implementation](https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py).

## Requisites
##### (only for testing the code on `CartPole-v0`)
This project requires any recent version of Google's TensorFlow. The steps for building it from sources are well explained in the [official documentation](https://www.tensorflow.org/install/install_sources); as a quick recap, you are required to:
1. Clone TensorFlow from the [official repository](https://github.com/tensorflow/tensorflow)
2. Switch to a stable release using e.g. `git checkout r1.2`
4. Run `./configure`
5. Build TensorFlow as a shared library using `bazel build --config=opt //tensorflow:libtensorflow_cc.so`
6. Copy `bazel-bin/tensorflow/libtensorflow_cc.so` to any directory searched by the run-time loader, e.g. `/usr/local/lib/`

## Installation
##### (only for testing the code on `CartPole-v0`)
1. Recursively clone this repository, i.e. `git clone --recursive https://github.com/integeruser/GA3C-cpp.git`
2. Run `./build.sh <path_to_tensorflow_repo>` to automatically copy the necessary C++ headers and sources into `third-party/`

## Usage
To try the code in an empty environment and on an empty model, just compile `GA3C` and run it (no dependencies):
```
$ make GA3C
mkdir -p bin
clang++ -std=c++11 -O2 -march=native -Wall -pthread -o bin/GA3C -I include \
	src/ga3c.cc src/main.cc
$ bin/GA3C
Training finished in 2.03081 seconds
```

To test the code on  `CartPole-v0`, first generate the TensorFlow model:
```
$ python3 cartpole-v0/cartpole-v0.py generate
```
Then, compile and train the agent:
```
$ make GA3C-cartpole-v0
mkdir -p bin
clang++ -std=c++11 -O2 -march=native -Wall -ltensorflow_cc -pthread -o bin/GA3C-cartpole-v0 -I include \
	-I third-party -I third-party/gym-uds-api/binding-cpp/include -I third-party/third_party -I cartpole-v0/include \
	third-party/gym-uds-api/binding-cpp/src/gym-uds.cc third-party/gym-uds-api/binding-cpp/src/gym-uds.pb.cc \
	src/ga3c.cc cartpole-v0/src/model-cartpole-v0.cc cartpole-v0/src/main.cc
$ bin/GA3C-cartpole-v0
[2017-08-31 23:25:52,655] Making new env: CartPole-v0
[2017-08-31 23:25:52,655] Making new env: CartPole-v0
…
[00] Ep. 00250 reward: 200
[02] Ep. 00260 reward: 200
Training finished in 10.768 seconds
```
The updated weights of the model are saved back to disk. Test the trained agent using:
```
$ python3 cartpole-v0/cartpole-v0.py test
```
Keep in mind to regenerate the empty model using the Python script before training again.
