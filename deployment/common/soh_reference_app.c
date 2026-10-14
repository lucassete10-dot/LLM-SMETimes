/* Reference Cortex-M4 integration, NOT a board-specific ADC/charger firmware.
 * Model/family is fixed at build time. Stage events must come from the charger.
 * The volatile mailbox only keeps the API reachable for linker-size analysis. */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "soh_waveform.h"
#include "soh_stream.h"
#include "soh_tree.h"
#include "soh_selected_columns.h"
#include "soh_reference_app.h"
static SohWaveStream waveform;
static SohStream history;
static float raw[51],selected[SOH_RAW_COUNT],query[SOH_INPUT_WIDTH];
static int cycle_number,discharge_stage,stage_active,charge_cycle;
#ifndef SOH_HOST_VERIFY
static float conditions[5];
static volatile float output_soh;
static volatile int output_status;
struct Command {int command,cycle,discharge,future;double seconds,voltage,current,temperature;float conditions[5];};
volatile struct Command soh_command;
#endif
void soh_app_init(void) {
    soh_stream_init(&history,SOH_FAMILY==3);
    cycle_number=charge_cycle=stage_active=0;
}
int soh_app_begin_stage(int cycle,int discharge) {
    if(stage_active||cycle<1||cycle>100||cycle<=history.last_cycle)return -1;
    if(discharge!=0&&discharge!=1)return -1;
    if(SOH_FAMILY!=3&&!discharge)return -1;
    if(SOH_FAMILY==3&&discharge&&charge_cycle!=cycle)return -1;
    if(SOH_FAMILY==3&&!discharge)charge_cycle=0;
    cycle_number=cycle;discharge_stage=discharge;stage_active=1;
    return soh_wave_stream_init(&waveform,SOH_FAMILY);
}
int soh_app_sample(double t,double v,double i,double temperature) {
    if(!stage_active)return -1;
    return soh_wave_stream_push(&waveform,t,v,i,temperature);
}
int soh_app_end_stage(void) {
    if(!stage_active)return -1;
    stage_active=0;
    int offset=SOH_FAMILY==3&&discharge_stage?23:0;
    int status=soh_wave_stream_finish(&waveform,raw+offset);
    if(status)return status;
    if(SOH_FAMILY==3&&!discharge_stage){charge_cycle=cycle_number;return 0;}
    for(int k=0;k<SOH_RAW_COUNT;k++)selected[k]=raw[soh_raw_columns[k]];
    if(soh_stream_update(&history,cycle_number,selected)!=1)return -1;
    charge_cycle=0;return 0;
}
int soh_app_predict(int future,const float actual_conditions[5],float *result) {
    if(!result||!actual_conditions||stage_active)return -1;
    /* Missing metadata is NaN and uses the frozen train-side tree imputer. */
    for(int k=0;k<5;k++)if(isinf(actual_conditions[k]))return -1;
    int status=soh_stream_query(&history,future,actual_conditions,query);
    if(status)return status;
    *result=soh_tree_predict(query);return isfinite(*result)?0:-1;
}
#ifndef SOH_HOST_VERIFY
extern unsigned long _sidata,_sdata,_edata,_sbss,_ebss,_estack;
void Default_Handler(void){for(;;){}}
void Reset_Handler(void) {
    unsigned long *src=&_sidata,*dst=&_sdata;
    while(dst<&_edata)*dst++=*src++;
    for(dst=&_sbss;dst<&_ebss;dst++)*dst=0;
    soh_app_init();
    for(;;) {
        int command=soh_command.command;if(!command)continue;
        if(command==1)output_status=soh_app_begin_stage(soh_command.cycle,soh_command.discharge);
        else if(command==2)output_status=soh_app_sample(soh_command.seconds,soh_command.voltage,soh_command.current,soh_command.temperature);
        else if(command==3)output_status=soh_app_end_stage();
        else if(command==4){float result;
            for(int k=0;k<5;k++)conditions[k]=soh_command.conditions[k];
            output_status=soh_app_predict(soh_command.future,conditions,&result);
            if(!output_status)output_soh=result;
        }else output_status=-1;
        soh_command.command=0;
    }
}
/* Core vectors only; actual STM32 interrupt vectors and clocks are absent. */
__attribute__((section(".isr_vector"),used))
const uintptr_t core_vectors[16]={(uintptr_t)&_estack,(uintptr_t)Reset_Handler,
 (uintptr_t)Default_Handler,(uintptr_t)Default_Handler,(uintptr_t)Default_Handler,
 (uintptr_t)Default_Handler,(uintptr_t)Default_Handler,0,0,0,0,
 (uintptr_t)Default_Handler,(uintptr_t)Default_Handler,0,
 (uintptr_t)Default_Handler,(uintptr_t)Default_Handler};
#endif
