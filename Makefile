PRESET ?= dev

.DEFAULT_GOAL := check

.PHONY: help check configure fresh build test format format-check tidy
.PHONY: dev release tsan

help:
	./build.sh help

check configure fresh build test format format-check tidy:
	./build.sh $@ $(PRESET)

dev release tsan:
	./build.sh check $@
