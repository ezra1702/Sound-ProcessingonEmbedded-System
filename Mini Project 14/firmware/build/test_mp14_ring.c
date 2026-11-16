#include <stdio.h>
#include "../mp14_ring_buffer.h"
static const int BUSY[200] = {2,1,0,2,0,0,6,1,0,0,9,0,9,6,1,9,0,2,9,6,0,2,2,6,0,0,2,1,6,2,1,0,6,2,6,0,6,0,1,9,2,0,1,0,9,0,0,0,9,1,1,0,6,9,0,9,0,1,0,2,0,0,0,0,0,0,1,0,9,1,0,6,6,0,0,0,1,0,6,0,2,6,0,0,1,0,0,0,0,2,0,1,6,0,1,0,0,1,0,6,9,9,2,0,9,2,0,0,0,0,6,0,6,0,0,1,0,9,6,9,0,0,6,1,0,0,9,0,0,0,1,9,9,0,9,6,0,0,6,1,2,0,0,2,6,9,2,1,2,9,0,9,0,2,6,6,0,2,9,0,6,1,2,9,2,6,2,9,0,2,0,2,0,6,1,0,6,1,6,9,1,0,2,0,0,0,0,0,0,1,0,6,0,9,0,0,1,2,0,0};
int main(void){ static block_ring_t r; ring_init(&r); int16_t blk[DMA_BLOCK_SAMPLES], out[DMA_BLOCK_SAMPLES];
 uint32_t seq; int wait=0, k=0; const int NB=(int)(sizeof(BUSY)/sizeof(BUSY[0]));
 for(int tick=0; tick<260; ++tick){ for(int i=0;i<DMA_BLOCK_SAMPLES;++i) blk[i]=(int16_t)(tick*7+i);
   ring_push(&r, blk, (uint32_t)tick);
   if(wait>0){ --wait; continue; }
   if(k<NB && ring_pop(&r, out, &seq)){ if(out[3]!=(int16_t)(seq*7+3)){printf("DATA_ERR\n");return 2;} printf("%u ", seq); wait=BUSY[k++]; } }
 printf("\nOVF %u\n", r.overflow_count); return 0; }
