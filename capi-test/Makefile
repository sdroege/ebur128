CC ?= gcc
CFLAGS ?= -O2 -g -Wall

all: capi-test

capi-test: capi-test.c
	$(CC) -o capi-test $(CFLAGS) capi-test.c $(shell pkg-config --libs --cflags libebur128) -lm

check: capi-test
	LD_LIBRARY_PATH=$(shell pkg-config --variable=libdir libebur128) ./capi-test

check-valgrind: capi-test
	LD_LIBRARY_PATH=$(shell pkg-config --variable=libdir libebur128) valgrind --error-exitcode=-1 --track-origins=yes --leak-check=full ./capi-test

clean:
	rm -f capi-test
