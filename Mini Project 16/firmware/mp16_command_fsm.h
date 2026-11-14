// MP16 FITUR TAMBAHAN: command state machine (identik dengan CommandFSM Python). Dipanggil tiap inference tick (100 ms).
#pragma once
#include <stdint.h>
enum { FSM_IDLE = 0, FSM_COOLDOWN = 1, FSM_FAULT = 2 };
enum { ACT_NONE, ACT_ON, ACT_OFF, ACT_LEVEL_UP, ACT_LEVEL_DOWN, ACT_REJECT, ACT_FAILSAFE_OFF, ACT_RECOVER, ACT_IGNORED_BOUND };
#define FSM_WD_TICKS 5
#define FSM_COOLDOWN_TICKS 8
#define FSM_LEVEL_MIN 0
#define FSM_LEVEL_MAX 10
/* urutan kelas: 0=mati, 1=naik, 2=nyala, 3=turun */
#define CLS_NYALA 2
#define CLS_MATI 0
#define CLS_NAIK 1
#define CLS_TURUN 3
typedef struct { uint8_t state, power; int8_t level; int16_t cd, wd; } cmd_fsm_t;
static inline void fsm_init(cmd_fsm_t *f) { f->state = FSM_IDLE; f->power = 0; f->level = 5; f->cd = 0; f->wd = 0; }
static inline int fsm_step(cmd_fsm_t *f, int audio_ok, int endpoint, int cls, int accepted) {
    int act = ACT_NONE;
    if (!audio_ok) {
        f->wd++;
        if (f->wd >= FSM_WD_TICKS && f->state != FSM_FAULT) { f->state = FSM_FAULT; f->power = 0; act = ACT_FAILSAFE_OFF; }
        return act;
    }
    f->wd = 0;
    if (f->state == FSM_FAULT) { f->state = FSM_IDLE; return ACT_RECOVER; }
    if (f->state == FSM_COOLDOWN) { if (--f->cd <= 0) f->state = FSM_IDLE; return act; }
    if (endpoint) {
        if (!accepted) act = ACT_REJECT;
        else if (cls == CLS_NYALA) { f->power = 1; act = ACT_ON; }
        else if (cls == CLS_MATI) { f->power = 0; act = ACT_OFF; }
        else if (cls == CLS_NAIK) { if (f->power && f->level < FSM_LEVEL_MAX) { f->level++; act = ACT_LEVEL_UP; } else act = ACT_IGNORED_BOUND; }
        else if (cls == CLS_TURUN) { if (f->power && f->level > FSM_LEVEL_MIN) { f->level--; act = ACT_LEVEL_DOWN; } else act = ACT_IGNORED_BOUND; }
        f->state = FSM_COOLDOWN; f->cd = FSM_COOLDOWN_TICKS;
    }
    return act;
}
