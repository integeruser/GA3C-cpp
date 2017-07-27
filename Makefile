default:
	mkdir -p bin
	$(CXX) -std=c++11 -O2 -pthread -o bin/GAC3 \
	-I include -I include/third_party -l tensorflow_cc \
	include/third_party/gym.cc include/third_party/gym-uds.pb.cc src/*.cc
