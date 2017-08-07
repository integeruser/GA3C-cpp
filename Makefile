default:
	mkdir -p bin
	$(CXX) -std=c++11 -O2 -pthread -o bin/GA3C -l tensorflow_cc \
	-I include -I gym-uds-api/binding-cpp/include -I third-party -I third-party/third_party \
	gym-uds-api/binding-cpp/src/gym-uds.cc gym-uds-api/binding-cpp/src/gym-uds.pb.cc src/*.cc

clean:
	rm -rf bin/GA3C
