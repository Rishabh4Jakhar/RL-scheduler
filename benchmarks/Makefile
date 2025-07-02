export BENCHMARKS_ROOT=/home/$(USER)/profiler/benchmarks

# define CFLAGS, LDFLAGS and LINKING LIBRARIES here
INCLUDES := -I/home/$(USER)/profiler/tools/inc
ADD_CFLAGS := -DHARMONIZER -I/home/$(USER)/profiler/tools/inc
ADD_LDFLAGS := -L/home/$(USER)/profiler/tools -Wl,--rpath,/home/$(USER)/profiler/tools
ADD_LDLIBS := -ldl -lpthread -lrt -lm 
BINARY_SUFFIX := _test

# Explicit paths to each benchmark
CoMD := $(BENCHMARKS_ROOT)/CoMD/src-openmp
SimpleMOC := $(BENCHMARKS_ROOT)/simplemoc/src
XSBench := $(BENCHMARKS_ROOT)/XSBench/openmp-threading
HPCCG := $(BENCHMARKS_ROOT)/hpccg
MinTally := $(BENCHMARKS_ROOT)/minTally

# Define the projects list with specific paths
PROJECTS := $(CoMD) $(SimpleMOC) $(XSBench) $(HPCCG) $(MinTally)

.PHONY: all clean $(PROJECTS)

# Build all projects with the specified flags
all: $(PROJECTS)

# Rule to build each project, passing CFLAGS and LDFLAGS
$(CoMD):
	$(MAKE) -C $(CoMD) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

$(SimpleMOC):
	$(MAKE) -C $(SimpleMOC) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

$(XSBench):
	$(MAKE) -C $(XSBench) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

$(MinTally):
	$(MAKE) -C $(MinTally) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

$(HPCCG):
	$(MAKE) -C $(HPCCG) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

# Clean all projects
clean:
	$(MAKE) -C $(CoMD) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(SimpleMOC) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(MinTally) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(XSBench) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(HPCCG) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"

