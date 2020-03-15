# gym-uds-api
This project provides a local [Unix domain socket](https://en.wikipedia.org/wiki/Unix_domain_socket) API to the [OpenAI Gym](https://github.com/openai/gym) toolkit, allowing development in languages other than Python with a faster interprocess communication than the [gym-http-api](https://github.com/openai/gym-http-api).

The API comes with a C++ binding example which supports one-dimensional observation spaces of type `Box` and action spaces of type `Discrete`.

## Requisites
The code requires the `gym` package for Python 3 and any recent version of Google's [Protocol Buffers](https://developers.google.com/protocol-buffers/).

## Installation
1. Clone or download this repository
2. Run `./build.sh` to generate the necessary protobuf headers and sources

## Usage
1. Start the server:
```
/o/gym-uds-api $ python3 ./gym-uds-server.py CartPole-v0
```
2. Run the (useless, for testing only) Python client or the C++ binding:
```
/o/gym-uds-api $ ./gym-uds-test-client.py
Ep. 1: 15.00
Ep. 2: 12.00
Ep. 3: 20.00
/o/gym-uds-api $ ./gym-uds-test-client.py
Ep. 1: 14.00
Ep. 2: 14.00
Ep. 3: 16.00
```
```
/o/g/binding-cpp $ make
mkdir -p bin
c++ -std=c++11 -O2 -o bin/gym-uds-client -I include -l protobuf src/*.cc
/o/g/binding-cpp $ bin/gym-uds-client
Ep. 1: 19
Ep. 2: 13
Ep. 3: 10
/o/g/binding-cpp $ bin/gym-uds-client
Ep. 1: 12
Ep. 2: 21
Ep. 3: 21
```
