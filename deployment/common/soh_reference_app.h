#ifndef SOH_REFERENCE_APP_H
#define SOH_REFERENCE_APP_H
/* Build selects known physical family; caller supplies charger stage events.
 * 0 success; push may return 1 to ignore beyond-prefix samples; negative error.
 * Source cycle must be 1..100. XJTU charge precedes matching discharge. */
void soh_app_init(void);
int soh_app_begin_stage(int source_cycle,int discharge);
int soh_app_sample(double relative_seconds,double voltage_V,double current_A,double temperature_C);
int soh_app_end_stage(void);
int soh_app_predict(int future_source_cycle,const float conditions[5],float *result);
#endif
