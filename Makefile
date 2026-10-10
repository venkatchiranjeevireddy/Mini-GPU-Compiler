CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror
BIN      := build/mini-ptx
SRC      := src/main.cpp
HDRS     := $(wildcard src/*.hpp)
EXAMPLES := $(wildcard examples/*.k)
PTX      := $(patsubst examples/%.k,examples/ptx/%.ptx,$(EXAMPLES))

all: $(BIN)

$(BIN): $(SRC) $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

examples/ptx/%.ptx: examples/%.k $(BIN)
	@mkdir -p examples/ptx
	$(BIN) $< -o $@

examples: $(PTX)

test: $(BIN)
	python3 tests/run_tests.py

clean:
	rm -rf build examples/ptx

.PHONY: all examples test clean
