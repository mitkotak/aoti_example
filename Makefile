LIBTORCH_PATH ?= /home/mkotak/atomic_architects/lib/libtorch

# Default target
all: build

# Build target
build: clean
	cmake -Bbuild -DCMAKE_PREFIX_PATH=$(LIBTORCH_PATH)
	cmake --build build

# Run target
run:
	./build/inference /home/mkotak/atomic_architects/lib/allegro/export/si-deployed.so

# Clean target
clean:
	rm -rf build

.PHONY: all build run clean
