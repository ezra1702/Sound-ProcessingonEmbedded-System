#pragma once
// MP14 FITUR TAMBAHAN: ring buffer blok DMA (SPSC) dengan kebijakan drop-oldest + overflow counter.
// ISR (producer) hanya memanggil ring_push(); task DSP (consumer) memanggil ring_pop().
// Pada MCU, CRIT_ENTER/CRIT_EXIT = portENTER_CRITICAL/portEXIT_CRITICAL (FreeRTOS) atau __disable_irq().
#include <stdint.h>
#include <string.h>
#include "mp14_realtime_constants.h"
#ifndef CRIT_ENTER
#define CRIT_ENTER()
#define CRIT_EXIT()
#endif
typedef struct {
    int16_t  data[AUDIO_RING_CAPACITY_BLOCKS][DMA_BLOCK_SAMPLES];
    uint32_t seq[AUDIO_RING_CAPACITY_BLOCKS];
    volatile uint32_t head, tail;      /* head: tulis berikut, tail: baca berikut (monoton) */
    volatile uint32_t overflow_count;
} block_ring_t;

static inline void ring_init(block_ring_t *r) { memset(r, 0, sizeof(*r)); }
static inline uint32_t ring_level(const block_ring_t *r) { return r->head - r->tail; }

/* Dipanggil dari ISR DMA half/full-complete. */
static inline void ring_push(block_ring_t *r, const int16_t *blk, uint32_t seq) {
    CRIT_ENTER();
    if (r->head - r->tail >= AUDIO_RING_CAPACITY_BLOCKS) { r->tail++; r->overflow_count++; } /* drop oldest */
    uint32_t i = r->head % AUDIO_RING_CAPACITY_BLOCKS;
    memcpy(r->data[i], blk, sizeof(r->data[i])); r->seq[i] = seq; r->head++;
    CRIT_EXIT();
}

/* Dipanggil dari task DSP; return 1 jika ada blok. */
static inline int ring_pop(block_ring_t *r, int16_t *out, uint32_t *seq) {
    CRIT_ENTER();
    if (r->head == r->tail) { CRIT_EXIT(); return 0; }
    uint32_t i = r->tail % AUDIO_RING_CAPACITY_BLOCKS;
    memcpy(out, r->data[i], sizeof(r->data[i])); *seq = r->seq[i]; r->tail++;
    CRIT_EXIT();
    return 1;
}
