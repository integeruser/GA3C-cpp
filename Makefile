default:
	$(CXX) -std=c++11 -O2 -pthread -o GAC3 \
	-I include -I include/third_party -l tensorflow_cc src/*.cc

clean:
	rm -rf GAC3
