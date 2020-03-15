#!/usr/bin/env bash
dirname () { python -c "import os; print(os.path.dirname(os.path.realpath('$0')))"; }
cd "$(dirname "$0")"

if [[ ! -z $1 ]] ; then
    PROTOC=$1
else
    PROTOC=$(which protoc)
fi

if [[ ! -z $PROTOC ]] ; then
    $PROTOC --python_out=. gym-uds.proto &&
    $PROTOC --cpp_out=. gym-uds.proto &&
    mv gym-uds.pb.h binding-cpp/include/ &&
    mv gym-uds.pb.cc binding-cpp/src/
else
    echo "protoc not found!"
    exit 1
fi
