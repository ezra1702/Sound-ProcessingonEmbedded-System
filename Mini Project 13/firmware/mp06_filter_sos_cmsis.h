#pragma once
/* Fitur tambahan Mini Project 6: koefisien SOS untuk CMSIS-DSP */
/* Fungsi: arm_biquad_cascade_df1_f32, urutan per stage {b0, b1, b2, -a1, -a2} */
/* Filter: iir_ellip_hp */
#define AUDIO_FS_HZ 16000
#define FILTER_NUM_STAGES 2
static const float FILTER_SOS_CMSIS[10] = { 0.833866f, -1.66749907f, 0.833866f, 1.87919533f, -0.888234079f, 1f, -1.99866033f, 1f, 1.98421013f, -0.987662554f };
/* State: float state[4 * FILTER_NUM_STAGES]; jangan reset antar blok audio. */
