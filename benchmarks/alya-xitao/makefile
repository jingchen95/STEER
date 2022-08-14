#ifndef XITAO_ROOT
#$(error XITAO_ROOT env variable is not set)
#endif
#include $(XITAO_ROOT)/makefile.sched
include ../../makefile.sched
CXXFLAGS += -I$(XITAO_ROOT)/include -I.
CXXFLAGS += -O3

CXXFLAGS += -ffast-math --std=c++11 ${EXTRAEINC}

LIBS = -lm ${EXTRAELIBS} -L$(XITAO_ROOT)/lib -L. -lxitao -pthread
PROGRAM_OBJS=main_cg.o functions.o 

all: solver.x

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

solver.x: $(PROGRAM_OBJS)
	$(CXX) $^ -o $@ ${LIBS}

run: solver.x 
	./solver.x C100K.csr

clean:
	rm -rf $(PROGRAM_OBJS) html latex
