BUILD_DIR := build
BUILD_TYPE ?= Release
CMAKE_FLAGS ?=
TARGET := terrainformer

.PHONY: all build run perf clean

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS) .. && make

run: build
	@__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./$(BUILD_DIR)/bin/$(TARGET)

perf: CMAKE_FLAGS := -DBUILD_PERF=ON
perf: build

clean:
	@rm -rf $(BUILD_DIR)