# GA3C-cpp
This repository contains a C++ multithreaded implementation of the asynchronous advantage actor-critic (A3C) algorithm based on [NVIDIA's GA3C](http://research.nvidia.com/publication/reinforcement-learning-through-asynchronous-advantage-actor-critic-gpu). It has been tested on the CartPole-v0 [OpenAI Gym](https://github.com/openai/gym) environment using [TensorFlow](https://www.tensorflow.org/) and [integeruser/gym-uds-api](https://github.com/integeruser/gym-uds-api), with model configuration and parameters as described in [jaromiru/AI-blog](https://github.com/jaromiru/AI-blog/blob/master/CartPole-A3C.py).

## Requisites
This project requires building TensorFlow 1.3 from sources. Example instructions are provided for macOS (tested on macOS Catalina 10.15.3).

0. Install [Homebrew](https://brew.sh).

1. Install [OpenJDK 8](https://adoptopenjdk.net):

        ~$ brew cask install homebrew/cask-versions/adoptopenjdk8

1. Download [bazel-0.4.5-jdk7-installer-darwin-x86_64.sh](https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-jdk7-installer-darwin-x86_64.sh), make it executable with `chmod`, and install Bazel 0.4.5 (to `$HOME/bin/bazel`):

        Downloads$ env JAVA_HOME="/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home" ./bazel-0.4.5-jdk7-installer-darwin-x86_64.sh --user

1. Install [pyenv](https://github.com/pyenv/pyenv) and Python 3.6.10 (to `$HOME/.pyenv/versions/3.6.10/bin/python`):

        ~$ brew install pyenv
        ~$ pyenv install 3.6.10

1. Install the wheel, [NumPy](https://numpy.org) and dlib pip packages:

        ~$ $HOME/.pyenv/versions/3.6.10/bin/pip install wheel numpy dlib

1. Clone TensorFlow from the official repository:

        ~$ git clone https://github.com/tensorflow/tensorflow

1. `cd` to the TensorFlow directory (assumed to be the working directory for all the next steps), then switch to version 1.3:

        tensorflow$ git checkout r1.3

1. Configure TensorFlow, specifying, when asked, `$HOME/.pyenv/versions/3.6.10/bin/python` as the location of Python (but expanding `$HOME` to its value):

        tensorflow$ env PATH="$HOME/bin:$PATH" JAVA_HOME="/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home" ./configure

1. Build the TensorFlow shared library (to `bazel-bin/tensorflow/libtensorflow_cc.so`):

        tensorflow$ env PATH="$HOME/bin:$PATH" JAVA_HOME="/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home" bazel build //tensorflow:libtensorflow_cc.so

1. Build and install the TensorFlow pip package:

        tensorflow$ env PATH="$HOME/bin:$PATH" JAVA_HOME="/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home" bazel build //tensorflow/tools/pip_package:build_pip_package
        tensorflow$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
        tensorflow$ $HOME/.pyenv/versions/3.6.10/bin/pip install /tmp/tensorflow_pkg/tensorflow-1.3.1-cp36-cp36m-macosx_10_15_x86_64.whl

1. Install the [OpenAI Gym](https://github.com/openai/gym) pip package:

        $ $HOME/.pyenv/versions/3.6.10/bin/pip install gym

## Installation
1. Recursively clone this repository:

        $ git clone --recursive https://github.com/integeruser/GA3C-cpp.git

1. `cd` to the `GA3C-cpp` directory and compile the code for testing on CartPole-v0:

        GA3C-cpp$ make TENSORFLOW_DIRPATH=/absolute/path/to/tensorflow/repository GA3C-cartpole-v0

## Usage
1. `cd` to the `GA3C-cpp/cartpole-v0` directory and start the gym-uds servers:

        GA3C-cpp/cartpole-v0$ env PATH="$HOME/.pyenv/versions/3.6.10/bin:$PATH" ./start-gym-uds-servers.sh

1. Generate the nontrained TensorFlow model:

        GA3C-cpp/cartpole-v0$ $HOME/.pyenv/versions/3.6.10/bin/python ./cartpole-v0.py generate

1. Train the agent (specifying `DYLD_LIBRARY_PATH` for finding `libtensorflow_cc.so`):

        GA3C-cpp/cartpole-v0$ env DYLD_LIBRARY_PATH="/absolute/path/to/tensorflow/repository/bazel-bin/tensorflow" bin/GA3C-cartpole-v0

1. The updated weights of the model are saved back to disk. Lastly, see the trained agent in action:

        GA3C-cpp/cartpole-v0$ $HOME/.pyenv/versions/3.6.10/bin/python ./cartpole-v0.py test
