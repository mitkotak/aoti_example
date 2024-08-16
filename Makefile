# Default target
all: build

# Build target
build:
	cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
	cmake --build build

# Run target
run: build
	./build/inference ${PWD}/si-deployed.so

# Clean target
clean:
	rm -rf build

.PHONY: all build run clean
