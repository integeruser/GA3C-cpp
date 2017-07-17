default:
	clang++ -std=c++11 -O2 -pthread -o GAC3 \
	-I include -I include/third_party -l tensorflow_cc \
	src/gym.cc src/gym-uds.pb.cc src/model.cc src/main.cc

clean:
	rm -rf GAC3
