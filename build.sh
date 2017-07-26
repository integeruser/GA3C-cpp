#!/usr/bin/env bash
cd $(dirname $(realpath $0))

TENSORFLOW_DIR="/opt/tensorflow"
GYM_UDS_API_DIR="/opt/gym-uds-api"
if [[ ! -d $TENSORFLOW_DIR ]] || [[ ! -d $GYM_UDS_API_DIR ]] ; then
    echo "At least one of the required folders does not exist."
    exit 1
fi

rm -f include/gym-uds.pb.h src/gym-uds.pb.cc
$TENSORFLOW_DIR/bazel-out/host/bin/external/protobuf/protoc --python_out=$GYM_UDS_API_DIR --proto_path=$GYM_UDS_API_DIR $GYM_UDS_API_DIR/gym-uds.proto
$TENSORFLOW_DIR/bazel-out/host/bin/external/protobuf/protoc --cpp_out=. --proto_path=$GYM_UDS_API_DIR $GYM_UDS_API_DIR/gym-uds.proto
mv gym-uds.pb.h include && mv gym-uds.pb.cc src
cp $GYM_UDS_API_DIR/binding-cpp/include/gym.h include && cp $GYM_UDS_API_DIR/binding-cpp/src/gym.cc src

rm -rf include/third_party
mkdir include/third_party
cp -r $TENSORFLOW_DIR/tensorflow include/third_party
cp -r $TENSORFLOW_DIR/bazel-genfiles/tensorflow include/third_party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/../../external/protobuf/src/google include/third_party
cp -r $TENSORFLOW_DIR/third_party/eigen3 include/third_party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/../../external/eigen_archive/. include/third_party/eigen3
cp -r include/third_party/eigen3/Eigen include/third_party
