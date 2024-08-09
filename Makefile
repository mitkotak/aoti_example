# Default target
all: build

# Build target
build:
	cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
	cmake --build build

# Clean target
clean:
	rm -rf build

.PHONY: all build clean
