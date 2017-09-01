CXX=clang++
CXXFLAGS=-std=c++11 -O2 -march=native -Wall
LDFLAGS=-pthread

GA3C:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o bin/GA3C -I include \
	src/ga3c.cc src/main.cc

clean:
	rm -rf bin/GA3C
