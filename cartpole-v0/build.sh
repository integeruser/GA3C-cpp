#!/usr/bin/env bash
cd $(dirname $(realpath $0))

if [[ $# -ne 1 ]] ; then
    echo "usage: build.sh ABS_PATH_TO_TENSORFLOW_REPO"
    exit 1
fi

ABS_PATH_TO_TENSORFLOW_REPO=$1
if [[ ! -d $ABS_PATH_TO_TENSORFLOW_REPO ]] ; then
    echo "ABS_PATH_TO_TENSORFLOW_REPO not found!"
    exit 1
fi

rm -rf third-party/google
rm -rf third-party/tensorflow
rm -rf third-party/third_party
mkdir third-party/third_party

cp -r $ABS_PATH_TO_TENSORFLOW_REPO/tensorflow third-party
cp -r $ABS_PATH_TO_TENSORFLOW_REPO/bazel-genfiles/tensorflow third-party
cp -r $ABS_PATH_TO_TENSORFLOW_REPO/bazel-tensorflow/external/protobuf/src/google third-party
cp -r $ABS_PATH_TO_TENSORFLOW_REPO/third_party/eigen3 third-party/third_party
cp -r $ABS_PATH_TO_TENSORFLOW_REPO/bazel-tensorflow/external/eigen_archive/unsupported/* third-party/third_party/eigen3/unsupported/
cp -r $ABS_PATH_TO_TENSORFLOW_REPO/bazel-tensorflow/external/eigen_archive/Eigen/* third-party/third_party/eigen3/Eigen/
cp -r third-party/third_party/eigen3/Eigen third-party/third_party

third-party/gym-uds-api/build.sh $ABS_PATH_TO_TENSORFLOW_REPO/bazel-out/host/bin/external/protobuf/protoc
