CXX=clang++
CXXFLAGS=-std=c++11 -O2 -march=native -Wall
LDFLAGS=-ltensorflow_cc

GA3C:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -pthread -o bin/GA3C -I include \
	src/ga3c.cc src/main.cc

GA3C-cartpole-v0:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -pthread -o bin/GA3C-cartpole-v0 -I include \
	-I third-party -I third-party/gym-uds-api/binding-cpp/include -I third-party/third_party -I cartpole-v0/include \
	third-party/gym-uds-api/binding-cpp/src/gym-uds.cc third-party/gym-uds-api/binding-cpp/src/gym-uds.pb.cc \
	src/ga3c.cc cartpole-v0/src/model-cartpole-v0.cc cartpole-v0/src/main.cc

clean:
	rm -rf bin/GA3C bin/GA3C-cartpole-v0
