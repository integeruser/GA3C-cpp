CXXFLAGS=-std=c++11 -O2 -march=native -Wall
LDFLAGS=-pthread -ltensorflow_cc

GA3C-cartpole-v0:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o bin/GA3C-cartpole-v0 -I ../include \
	-I third-party -I third-party/gym-uds-api/binding-cpp/include -I third-party/third_party -I include \
	third-party/gym-uds-api/binding-cpp/src/gym-uds.cc third-party/gym-uds-api/binding-cpp/src/gym-uds.pb.cc \
	../src/ga3c.cc src/model-cartpole-v0.cc src/main.cc

clean:
	rm -rf bin/GA3C-cartpole-v0
