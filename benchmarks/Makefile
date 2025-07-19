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
CoHMM := $(BENCHMARKS_ROOT)/CoHMM
AMG := $(BENCHMARKS_ROOT)/AMG
miniMD := $(BENCHMARKS_ROOT)/miniMD
miniQMC := $(BENCHMARKS_ROOT)/miniQMC
QuickSilver := $(BENCHMARKS_ROOT)/QuickSilver

# Define the projects list with specific paths
PROJECTS := $(CoHMM) $(CoMD) $(SimpleMOC) $(XSBench) $(HPCCG) $(MinTally) $(AMG) $(miniMD) $(miniQMC) $(Quicksilver)

.PHONY: all clean $(PROJECTS)

# Build all projects with the specified flags
all: $(PROJECTS) $(miniMD)

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

$(CoHMM):
	$(MAKE) -C $(CoHMM) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

$(AMG):
	$(MAKE) -C $(AMG) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

$(miniMD):
	cd $(miniMD)/ref && mkdir -p Obj_openmpi
	cp $(miniMD)/ref/Makefile.kokkos $(miniMD)/ref/Obj_openmpi/Makefile
	cd $(miniMD)/ref/Obj_openmpi && $(MAKE) -j
	cp $(miniMD)/ref/Obj_openmpi/miniMD_openmpi $(miniMD)/miniMD_test
$(miniQMC):
	$(MAKE) -C $(miniQMC) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

$(Quicksilver):
	$(MAKE) -C $(Quicksilver) ADD_CFLAGS="$(ADD_CFLAGS)" ADD_LDFLAGS="$(ADD_LDFLAGS)" ADD_LDLIBS="$(ADD_LDLIBS)" BINARY_SUFFIX="$(BINARY_SUFFIX)"

# Clean all projects
clean:
	$(MAKE) -C $(CoMD) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(SimpleMOC) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(MinTally) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(XSBench) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(HPCCG) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(CoHMM) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	$(MAKE) -C $(AMG) clean BINARY_SUFFIX="$(BINARY_SUFFIX)"
	rm -rf $(miniMD)/ref/Obj_openmpi
	rm -f $(miniMD)/miniMD_test
	rm -rf $(miniMD)/CMakeFiles $(miniMD)/CMakeCache.txt $(miniMD)/miniMD_test $(miniMD)/Makefile $(miniMD)/*.o $(miniMD)/cmake_install.cmake
	rm -rf $(miniQMC)/ref/Obj_openmpi $(miniQMC)/miniQMC_test
	rm -f $(Quicksilver)/qs
	rm -f $(Quicksilver)/git_vers.hh $(Quicksilver)/git_hash.hh
	rm -f $(Quicksilver)/*.o
	@echo "All benchmarks cleaned."
