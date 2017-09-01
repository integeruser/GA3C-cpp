#!/usr/bin/env bash
cd $(dirname $(realpath $0))

if [[ $# -ne 1 ]] ; then
    echo "usage: build.sh TENSORFLOW_DIR"
    exit 1
fi

TENSORFLOW_DIR=$1
if [[ ! -d $TENSORFLOW_DIR ]] ; then
    echo "TENSORFLOW_DIR not found!"
    exit 1
fi

rm -rf third-party/google
rm -rf third-party/tensorflow
rm -rf third-party/third_party
mkdir third-party/third_party

cp -r $TENSORFLOW_DIR/tensorflow third-party
cp -r $TENSORFLOW_DIR/bazel-genfiles/tensorflow third-party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/external/protobuf/src/google third-party
cp -r $TENSORFLOW_DIR/third_party/eigen3 third-party/third_party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/external/eigen_archive/. third-party/third_party/eigen3
cp -r third-party/third_party/eigen3/Eigen third-party/third_party

third-party/gym-uds-api/build.sh $TENSORFLOW_DIR/bazel-out/host/bin/external/protobuf/protoc
