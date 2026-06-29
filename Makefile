ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

EXE ?= $(ROOT)/sable
CARGO ?= cargo
CC ?= cc
EVALFILE ?=
MANIFEST := $(ROOT)/Cargo.toml
TARGET_DIR ?= $(ROOT)/target
unexport CC
unexport CXX

ifeq ($(OS),Windows_NT)
BIN_SUFFIX := .exe
MSYS_PATHS := $(filter /%,$(TARGET_DIR) $(EXE))
ifneq ($(strip $(MSYSTEM) $(MSYS_PATHS)),)
COPY = cp -f "$(1)" "$(2)"
else
COPY = powershell.exe -NoProfile -Command "Copy-Item -LiteralPath '$(1)' -Destination '$(2)' -Force"
endif
else
BIN_SUFFIX :=
COPY = cp -f "$(1)" "$(2)"
endif

BUILT_EXE := $(TARGET_DIR)/release/sable-engine$(BIN_SUFFIX)

CARGO_ENV :=
ifneq ($(strip $(EVALFILE)),)
CARGO_ENV += SABLER_EVAL_FILE="$(EVALFILE)" SABLER_DEFAULT_EVAL=nnue
endif

.PHONY: all build clean

all: build

build:
	$(CARGO_ENV) $(CARGO) build --release --manifest-path "$(MANIFEST)" --target-dir "$(TARGET_DIR)"
	$(call COPY,$(BUILT_EXE),$(EXE))

clean:
	$(CARGO) clean --manifest-path "$(MANIFEST)" --target-dir "$(TARGET_DIR)"
