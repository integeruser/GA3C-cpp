#!/usr/bin/env bash
TENSORFLOW_DIR="/opt/tensorflow"
GYM_UDS_API_DIR="/opt/gym-uds-api"

if [[ ! -d $TENSORFLOW_DIR ]] || [[ ! -d $GYM_UDS_API_DIR ]] ; then
    echo "At least one of the required folders does not exist."
    exit 1
fi


cd $(dirname $(realpath $0))
rm -rf include/third_party
mkdir include/third_party
cp -r $TENSORFLOW_DIR/tensorflow include/third_party
cp -r $TENSORFLOW_DIR/bazel-genfiles/tensorflow include/third_party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/../../external/protobuf/src/google include/third_party
cp -r $TENSORFLOW_DIR/third_party/eigen3 include/third_party
cp -r $TENSORFLOW_DIR/bazel-tensorflow/../../external/eigen_archive/. include/third_party/eigen3
cp -r include/third_party/eigen3/Eigen include/third_party

rm -rf /usr/local/lib/libtensorflow_cc.so
cp $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/


(cd $GYM_UDS_API_DIR; ./build.sh $TENSORFLOW_DIR/bazel-out/host/bin/external/protobuf/protoc)
cp $GYM_UDS_API_DIR/binding-cpp/include/{gym.h,gym-uds.pb.h} include
cp $GYM_UDS_API_DIR/binding-cpp/src/{gym.cc,gym-uds.pb.cc} src
