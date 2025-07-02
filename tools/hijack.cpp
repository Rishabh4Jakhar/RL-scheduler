#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h> 
#include "hijack.h"

using namespace std;

static int index = 0;
static pid_t prev_tid = 0;

typedef int (*modified_pthread_create)(pthread_t * thread, const pthread_attr_t * attr, void *(*start_routine)(void *), void * arg);
typedef int (*modified_pthread_join)(pthread_t thread, void **retval);

modified_pthread_create new_pthread_create;
modified_pthread_join new_pthread_join;

// Just for checking function called from benchmark.
pid_t Thread_ids[48];
pid_t* get_thread_ids() {
    return Thread_ids;
}

void create_shared_memory() {
  char obj[] = "__tid_shared_memory";
  int shm_fd = shm_open(obj, O_CREAT | O_RDWR, S_IRUSR | S_IWOTH);
  ftruncate(shm_fd, sizeof(SharedData));
  SharedData *shared_data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

  cout<<"\nShared memory created";
  fflush(stdout);
}

void define_start_index() {
    int id = atoi(getenv("BENCHMARK_ID"));
    index = id * THREAD_COUNT;
}

void sigusr1(int s, siginfo_t *info, void *v) {
    pid_t tid = gettid();
    int int_val = info->si_value.sival_int;
    Thread_ids[int_val] = tid;

    int shm_fd = shm_open("__tid_shared_memory", O_RDWR , 0);
    SharedData *shared_data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_WRITE, MAP_SHARED, shm_fd, 0);

    shared_data->thread_ids[int_val] = tid;

    munmap(shared_data, sizeof(SharedData));
    close(shm_fd);

    printf("\nThread id of current thread %d is: %d\n", int_val, tid);
    fflush(stdout);
}

int pthread_create(pthread_t * thread, const pthread_attr_t * attr, void *(*start_routine)(void *), void * arg) {

    if(prev_tid != gettid()) {
        define_start_index();
        pid_t tid = gettid();
        Thread_ids[index] = tid;
        cout<<"\n=== hijack =====   Thread id for thread " << 0 << " is " << tid << "\n";
        fflush(stdout);

        prev_tid = tid;

        int shm_fd = shm_open("__tid_shared_memory", O_RDWR , 0);
        SharedData *shared_data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_WRITE, MAP_SHARED, shm_fd, 0);
        shared_data->thread_ids[index] = tid;

        munmap(shared_data, sizeof(SharedData));
        close(shm_fd);

        index += 1;
    }
    
    new_pthread_create = (modified_pthread_create)dlsym(RTLD_NEXT, "pthread_create");

    cout << "\npthread_create Hijacked \n";
    fflush(stdout);

    int status = new_pthread_create(thread, attr, start_routine, arg);

    struct sigaction action;
    sigval value;
    action.sa_flags = SA_SIGINFO; 
    action.sa_sigaction = &sigusr1;

    if (sigaction(SIGUSR1, &action, NULL) == -1) { 
        perror("sigusr: sigaction");
        _exit(1);
    }
    value.sival_int = index; // change this value to thread number.
    pthread_sigqueue(*thread, SIGUSR1, value);

    index += 1;

    return status;
}

int pthread_join(pthread_t thread, void **retval) {

    new_pthread_join = (modified_pthread_join)dlsym(RTLD_NEXT, "pthread_join");

    cout<< "\npthread_join Hijacked \n";
    fflush(stdout);

    new_pthread_join(thread, nullptr);
    return 0;
}