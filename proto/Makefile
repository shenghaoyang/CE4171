.PHONY: clean

all: ../dlserver/dlserver_pb2.py

clean:
	rm -rf ../dlserver/*_pb2*.py

../dlserver/dlserver_pb2.py: dlserver.proto
	python -m grpc_tools.protoc -I . \
	--python_out=../dlserver --grpc_python_out=../dlserver $<
