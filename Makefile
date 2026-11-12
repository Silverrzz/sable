CARGO ?= cargo
TARGET_DIR ?= target
unexport CC CXX

ifeq ($(OS),Windows_NT)
EXE ?= sable.exe
PYTHON ?= py -3
else
EXE ?= sable
PYTHON ?= python3
endif

.PHONY: native all build clean

native:
ifeq ($(strip $(SABLE_EVAL_FILE)$(EVALFILE)),)
	$(PYTHON) ./download_net.py
endif
	$(CARGO) rustc --release --bin sable-engine --target-dir "$(TARGET_DIR)" -- --emit link="$(EXE)"

all build: native

clean:
	$(CARGO) clean --target-dir "$(TARGET_DIR)"
