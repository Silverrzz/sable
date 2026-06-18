EXE ?= sable
CARGO ?= cargo
CC ?= cc
EVALFILE ?=

ifeq ($(OS),Windows_NT)
BIN_SUFFIX := .exe
COPY := copy /Y
NULL := NUL
else
BIN_SUFFIX :=
COPY := cp -f
NULL := /dev/null
endif

CARGO_ENV :=
ifneq ($(strip $(EVALFILE)),)
CARGO_ENV += SABLER_EVAL_FILE="$(EVALFILE)" SABLER_DEFAULT_EVAL=nnue
endif

.PHONY: all build clean

all: build

build:
	$(CARGO_ENV) $(CARGO) build --release
	$(COPY) target/release/sable-engine$(BIN_SUFFIX) $(EXE) > $(NULL)

clean:
	$(CARGO) clean
