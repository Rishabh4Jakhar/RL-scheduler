#===============================================================================
# User Options
#===============================================================================

CC = clang
# CC = cc

# Compiler can be set below, or via environment variable
ifeq ($(CC),)
CC    = clang
endif
OPTIMIZE    = yes
OPENMP      = yes
DEBUG       = no
PROFILE     = no
MPI         = no
AML         = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench$(BINARY_SUFFIX)

source = \
Main.c \
io.c \
Simulation.c \
GridInit.c \
XSutils.c \
Materials.c

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Make sure we didn't pick up lower case cc
ifeq (cc,$(CC))
CC    = gcc
endif

# Standard Flags
CFLAGS := -std=gnu99 -Wall -I../../ 

# Linker Flags
LDFLAGS = 

# Regular gcc Compiler
ifneq (,$(findstring gcc,$(CC)))
  CFLAGS += -flto
  ifeq ($(OPENMP),yes)
	CFLAGS += -fopenmp -DOPENMP
  endif
endif

# LLVM Compiler
ifneq (,$(findstring clang,$(CC)))
  CFLAGS += -flto
  ifeq ($(OPENMP),yes)
	CFLAGS += -fopenmp=libomp -DOPENMP -lm -O3
  endif
endif

# Intel Compiler
ifneq (,$(findstring intel,$(CC)))
  CFLAGS += -ipo
  ifeq ($(OPENMP),yes)
	CFLAGS += -fopenmp -DOPENMP
  endif
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# MPI
ifeq ($(MPI),yes)
  CC = mpicc
  CFLAGS += -DMPI
endif

# AML
ifeq ($(AML),yes)
  LDFLAGS += $(shell pkg-config --libs aml)
  CFLAGS += -DAML $(shell pkg-config --cflags aml)
endif

LIBRARY = 
CFLAGS += $(ADD_CFLAGS) $(ADD_LDFLAGS) $(LIBRARY) -DDYN_TOOL
LDFLAGS  += $(ADD_LDLIBS)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) XSbench_header.h

run:
	./$(program)
