#ifndef SOH_WAVEFORM_H
#define SOH_WAVEFORM_H
#include <stddef.h>
/* Segmented physical stage, relative seconds, nondecreasing timestamps
 * (XJTU preprocessing keeps the first of duplicate timestamps).
 * Families: 1 MIT discharge (51 values), 2 NASA discharge (44),
 * 3 XJTU charge OR discharge (23; caller concatenates the two stages).
 * Caller owns input buffers. Scratch is constant; no allocation or capacity input.
 * This batch reference API is not yet an ADC streaming interface. */
int soh_waveform_features(int family, size_t n, const double *time_s,
    const double *voltage, const double *current, const double *temperature,
    float *output);
typedef struct { double mean,m2; size_t n; } SohMoments;
typedef struct {
    double grid[3][11],previous[4],end,step,seen_time,charge,energy,max_temperature;
    SohMoments voltage,current,absolute_current,temperature;
    size_t count;
    int family,grid_count,next_grid,temperature_ok,have_previous,boundary_seen,error,finished;
} SohWaveStream;
/* Stage start is supplied by the caller (no automatic cycle/phase detection).
 * push: 0 accepted, 1 beyond prefix/duplicate XJTU ignored, negative error.
 * NASA consumes the first boundary-bracketing sample just as training does.
 * finish is called once. State has fixed size independent of sample count. */
int soh_wave_stream_init(SohWaveStream *state,int family);
size_t soh_wave_stream_state_size(void);
int soh_wave_stream_push(SohWaveStream *state,double seconds,double voltage,
    double current,double temperature);
int soh_wave_stream_finish(SohWaveStream *state,float *output);
#endif
