#ifdef __cplusplus
extern "C" {
#endif

#define CORES_PER_APP 10
#define MAX_THREADS 20
#define THREAD_COUNT 20

pid_t* get_thread_ids();
typedef struct SharedData{
    pid_t thread_ids[MAX_THREADS];
} SharedData;

#ifdef __cplusplus
}
#endif
