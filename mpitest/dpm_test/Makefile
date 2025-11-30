CFLAGS:= -std=c99 -g -O0 

# If we are configured for openmpi set -DOPENMPI
ifneq ($(shell which ompi_info),)
$(info "Configuring for OPENMPI")
CFLAGS+= -DOPENMPI=1
endif

all: server client

server: server.c smcomm.h
	mpicc $(CFLAGS) -o server server.c

client: client.c smcomm.h
	mpicc $(CFLAGS) -o client client.c

clean:
	rm -rf server server.dSYM client client.dSYM nameserver.id *.log hosts.txt
