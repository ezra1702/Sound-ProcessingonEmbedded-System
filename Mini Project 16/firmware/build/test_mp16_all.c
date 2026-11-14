#include "mp14_realtime_constants.h"
#include "mp14_ring_buffer.h"
#include "mp15_classical_model_int8.h"
#include "mp16_project_config.h"
#include "mp16_command_fsm.h"
int main(void){ float x[39]={0}; cmd_fsm_t f; fsm_init(&f); return fsm_step(&f,1,1,mp15_classical_predict(x),1) < 0; }
