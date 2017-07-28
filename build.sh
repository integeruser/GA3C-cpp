#!/usr/bin/env bash
cd $(dirname $(realpath $0))

TENSORFLOW_DIR="/opt/tensorflow"
if [[ ! -d $TENSORFLOW_DIR ]] ; then
    echo "TENSORFLOW_DIR not found!"
    exit 1
fi

rm -rf include/third_party
mkdir include/third_party

cp -r $TENSORFLOW_DIR/tensorflow include/third_party
cp -r $TENSORFLOW_DIR/bazel-genfiles/tensorflow include/third_party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/../../external/protobuf/src/google include/third_party
cp -r $TENSORFLOW_DIR/third_party/eigen3 include/third_party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/../../external/eigen_archive/. include/third_party/eigen3
cp -r include/third_party/eigen3/Eigen include/third_party

gym-uds-api/build.sh $TENSORFLOW_DIR/bazel-out/host/bin/external/protobuf/protoc
