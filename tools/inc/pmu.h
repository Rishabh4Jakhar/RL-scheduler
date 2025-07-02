#ifdef __cplusplus
extern "C" {
#endif

#define NUM_EVENTS 5  // Number of performance events
#define SLEEP_DURATION 100

void libpfm_init();  
void libpfm_finalize();  

void accumulate_energy();
void accumulate_pmcs();
void accumulate_cbox();

double getEventOnCore(int core_id, int event_index);  
double getEventOnSocket(int socket_id, int event_index);  

#ifdef __cplusplus
}
#endif