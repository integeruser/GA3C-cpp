CXXFLAGS=-std=c++11 -O2 -march=native -Wall
LDFLAGS=-pthread

GA3C:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o bin/GA3C \
		-I include \
		src/ga3c.cc src/main.cc

GA3C-cartpole-v0:
	mkdir -p cartpole-v0/bin
	test ${TENSORFLOW_DIRPATH}  # TENSORFLOW_DIRPATH must be defined
	cartpole-v0/third-party/gym-uds-api/build.sh ${TENSORFLOW_DIRPATH}/bazel-out/host/bin/external/protobuf/protoc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o cartpole-v0/bin/GA3C-cartpole-v0 \
		-I include \
		-I cartpole-v0/include \
		-I cartpole-v0/third-party/gym-uds-api/binding-cpp/include \
		-I ${TENSORFLOW_DIRPATH}/bazel-genfiles \
		-I ${TENSORFLOW_DIRPATH}/bazel-tensorflow \
		-I ${TENSORFLOW_DIRPATH}/bazel-tensorflow/external/protobuf/src \
		-I ${TENSORFLOW_DIRPATH}/bazel-tensorflow/external/eigen_archive \
		-L ${TENSORFLOW_DIRPATH}/bazel-bin/tensorflow -ltensorflow_cc \
		cartpole-v0/third-party/gym-uds-api/binding-cpp/src/gym-uds.cc \
		cartpole-v0/third-party/gym-uds-api/binding-cpp/src/gym-uds.pb.cc \
		src/ga3c.cc cartpole-v0/src/model-cartpole-v0.cc cartpole-v0/src/main.cc

clean:
	rm -rf bin/GA3C
	rm -rf cartpole-v0/bin/GA3C-cartpole-v0
