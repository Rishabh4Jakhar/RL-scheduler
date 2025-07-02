#ifndef CoMD_info_hpp
#define CoMD_info_hpp

#define CoMD_VARIANT "CoMD"
#define CoMD_HOSTNAME "hmaster"
#define CoMD_KERNEL_NAME "'Linux'"
#define CoMD_KERNEL_RELEASE "'5.15.0-131-generic'"
#define CoMD_PROCESSOR "'x86_64'"

#define CoMD_COMPILER "'/usr/bin/clang++'"
#define CoMD_COMPILER_VERSION "'Ubuntu clang version 17.0.6 (++20231208085846+6009708b4367-1~exp1~20231208085949.74)'"
#define CoMD_CFLAGS "'-std=c++11 -fopenmp=libomp -DDOUBLE -g -O3 -I../../   -DHARMONIZER -I/home/vanshika/profiler/tools/inc'"
#define CoMD_LDFLAGS "'-DDYN_TOOL   -L/home/vanshika/profiler/tools -Wl,--rpath,/home/vanshika/profiler/tools -ldl -lpthread -lrt -lm '"

#endif
