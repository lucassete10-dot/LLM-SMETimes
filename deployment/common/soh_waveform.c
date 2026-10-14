#include "soh_waveform.h"
#include <math.h>
#include <string.h>
typedef SohMoments Moments;
typedef char SohWaveStateFitsOneKiB[(sizeof(SohWaveStream)<=1024)?1:-1];
size_t soh_wave_stream_state_size(void) { return sizeof(SohWaveStream); }
static void add(Moments *s,double x) {
    double d=x-s->mean; s->n++;s->mean+=d/(double)s->n;s->m2+=d*(x-s->mean);
}
static double sd(const Moments *s) { return sqrt(fmax(0,s->m2/(double)s->n)); }
static double sample(size_t n,const double *t,const double *x,double target) {
    size_t j=0;
    while(j+1<n && t[j+1]<=target) j++;
    if(target<t[0])return x[0];
    if(j+1==n)return x[j];
    return x[j]+(x[j+1]-x[j])*(target-t[j])/(t[j+1]-t[j]);
}
static double quantile(const double *x,int n,double p) {
    double a[11],z=p*(n-1);int j=(int)z;
    for(int k=0;k<n;k++)a[k]=x[k];
    for(int k=1;k<n;k++){double v=a[k];int h=k;while(h>0 && a[h-1]>v){a[h]=a[h-1];h--;}a[h]=v;}
    return a[j]+(a[j<n-1?j+1:j]-a[j])*(z-j);
}
int soh_waveform_features(int family,size_t n,const double *t,const double *v,
    const double *i,const double *temp,float *out) {
    double end=family==1?300:family==2?1000:120,step=family==2?100:30;
    int g=family==3?5:11,temp_ok=1;
    double vg[11],ig[11],tg[11];
    Moments mv={0},mi={0},ma={0},mt={0};
    double ah=0,wh=0,max_temp=-INFINITY;
    if(family<1||family>3||n<2||!t||!v||!i||!temp||!out)return -1;
    for(size_t k=0;k<n;k++) {
        if(!isfinite(t[k])||!isfinite(v[k])||!isfinite(i[k])||(k&&t[k]<t[k-1]))return -2;
        if(t[k]<=end && !isfinite(temp[k]))temp_ok=0;
    }
    if(t[0]<0||t[0]>30||t[n-1]<(family==1?285:family==2?1000:118))return -3;
    if(!temp_ok && family!=1)return -4;
    size_t grid_n=n;
    if(family==1) {grid_n=0;while(grid_n<n&&t[grid_n]<=end)grid_n++;if(grid_n<3)return -3;}
    for(int k=0;k<g;k++) {
        vg[k]=sample(grid_n,t,v,step*k);ig[k]=sample(grid_n,t,i,step*k);
        tg[k]=temp_ok?sample(grid_n,t,temp,step*k):NAN;
        out[k]=(float)vg[k];out[g+k]=(float)ig[k];out[2*g+k]=(float)tg[k];
    }
    if(family==3) {
        for(int k=0;k<g;k++) {
            add(&mv,vg[k]);add(&mi,ig[k]);add(&mt,tg[k]);
            if(k)ah+=(fabs(ig[k-1])+fabs(ig[k]))*.5*step;
        }
        double x[]={vg[g-1]-vg[0],mv.mean,(vg[g-1]-vg[0])/end,mi.mean,
                    sd(&mi),tg[g-1]-tg[0],mt.mean,ah/3600};
        for(int k=0;k<8;k++)out[15+k]=(float)x[k];
        return 0;
    }
    /* MIT/NASA statistics count native measurements plus a padded endpoint,
     * including duplicate endpoint exactly as the Python training contract. */
    double prev_t=0,prev_v=0,prev_i=0;int have=0;
    for(size_t k=0;k<=n;k++) {
        if(k<n&&t[k]>end)continue;
        double tt=k==n?end:t[k],vv=k==n?vg[g-1]:v[k];
        double ii=k==n?ig[g-1]:i[k],temperature=k==n?tg[g-1]:temp[k];
        add(&mv,vv);add(&mi,ii);add(&ma,fabs(ii));
        if(temp_ok){add(&mt,temperature);if(temperature>max_temp)max_temp=temperature;}
        if(have){double dt=tt-prev_t;ah+=.5*(fabs(prev_i)+fabs(ii))*dt;
            wh+=.5*(fabs(prev_i)*prev_v+fabs(ii)*vv)*dt;}
        prev_t=tt;prev_v=vv;prev_i=ii;have=1;
    }
    if(family==2) {
        double x[]={vg[0]-vg[g-1],mv.mean,sd(&mv),ma.mean,sd(&mi),mt.mean,sd(&mt),
                    tg[g-1]-tg[0],max_temp,ah/3600,wh/3600};
        for(int k=0;k<11;k++)out[33+k]=(float)x[k];
    } else {
        double integral=0;
        for(int k=1;k<g;k++)integral+=(vg[k-1]+vg[k])*.5*step;
        double x[]={vg[0]-vg[g-1],mv.mean,sd(&mv),ma.mean,sd(&mi),ah/3600,wh/3600,
          (vg[3]-vg[0])/90,(vg[g-1]-vg[g-4])/90,vg[0]-2*vg[g/2]+vg[g-1],
          quantile(vg,g,.5),quantile(vg,g,.75)-quantile(vg,g,.25),integral,
          quantile(ig,g,.5),quantile(ig,g,.75)-quantile(ig,g,.25),tg[g-1]-tg[0],
          temp_ok?quantile(tg,g,.5):NAN,temp_ok?quantile(tg,g,.75)-quantile(tg,g,.25):NAN};
        for(int k=0;k<18;k++)out[33+k]=(float)x[k];
    }
    return 0;
}
int soh_wave_stream_init(SohWaveStream *s,int family) {
    if(!s||family<1||family>3)return -1;
    memset(s,0,sizeof(*s));s->family=family;s->temperature_ok=1;
    s->grid_count=family==3?5:11;s->step=family==2?100:30;
    s->end=family==1?300:family==2?1000:120;s->max_temperature=-INFINITY;
    return 0;
}
static void stream_stat(SohWaveStream *s,double t,double v,double i,double temp) {
    add(&s->voltage,v);add(&s->current,i);add(&s->absolute_current,fabs(i));
    if(isfinite(temp)){add(&s->temperature,temp);if(temp>s->max_temperature)s->max_temperature=temp;}
    if(s->have_previous) {
        double dt=t-s->previous[0],pv=s->previous[1],pi=s->previous[2];
        s->charge+=.5*(fabs(pi)+fabs(i))*dt;
        s->energy+=.5*(fabs(pi)*pv+fabs(i)*v)*dt;
    }
}
int soh_wave_stream_push(SohWaveStream *s,double t,double v,double i,double temp) {
    if(!s||s->error||s->finished||s->family<1||s->family>3)return -1;
    if(s->boundary_seen)return 1;
    if(!isfinite(t)||t<0||!isfinite(v)||!isfinite(i))return s->error=-2;
    if(s->have_previous&&t<s->seen_time)return s->error=-2;
    if(s->family==3&&s->have_previous&&t==s->seen_time)return 1;
    s->seen_time=t;
    if(t>s->end&&s->family!=2) {s->boundary_seen=1;return 1;}
    if(!isfinite(temp)) {
        if(s->family!=1)return s->error=-4;
        s->temperature_ok=0;
    }
    if(!s->have_previous&&t>30)return s->error=-3;
    /* Delay exact-time grid points until the next sample, so duplicate MIT/
     * NASA timestamps use the last observation, matching numpy.interp. */
    while(s->next_grid<s->grid_count&&s->next_grid*s->step<t) {
        double target=s->next_grid*s->step,values[]={v,i,temp};
        for(int c=0;c<3;c++) {
            double value=values[c];
            if(s->have_previous) {
                double p=s->previous[c+1],pt=s->previous[0];
                value=p+(value-p)*(target-pt)/(t-pt);
            }
            s->grid[c][s->next_grid]=value;
        }
        s->next_grid++;
    }
    if(t<=s->end) {
        if(s->family!=3)stream_stat(s,t,v,i,temp);
        s->previous[0]=t;s->previous[1]=v;s->previous[2]=i;s->previous[3]=temp;
        s->have_previous=1;s->count++;
    } else s->boundary_seen=1;
    return 0;
}
int soh_wave_stream_finish(SohWaveStream *s,float *out) {
    if(!s||!out||s->error||s->finished||!s->have_previous)return -1;
    size_t minimum=s->family==1?3:s->family==2?2:90;
    double required=s->family==1?285:s->family==2?1000:118;
    if(s->count<minimum||s->seen_time<required)return -3;
    while(s->next_grid<s->grid_count) {
        for(int c=0;c<3;c++)s->grid[c][s->next_grid]=s->previous[c+1];
        s->next_grid++;
    }
    int g=s->grid_count;double *v=s->grid[0],*i=s->grid[1],*t=s->grid[2];
    for(int k=0;k<g;k++) {
        if(!s->temperature_ok)t[k]=NAN;
        out[k]=(float)v[k];out[g+k]=(float)i[k];out[2*g+k]=(float)t[k];
    }
    if(s->family==3) {
        for(int k=0;k<g;k++) {
            add(&s->voltage,v[k]);add(&s->current,i[k]);add(&s->temperature,t[k]);
            if(k)s->charge+=.5*(fabs(i[k-1])+fabs(i[k]))*s->step;
        }
        double x[]={v[g-1]-v[0],s->voltage.mean,(v[g-1]-v[0])/s->end,
          s->current.mean,sd(&s->current),t[g-1]-t[0],s->temperature.mean,s->charge/3600};
        for(int k=0;k<8;k++)out[15+k]=(float)x[k];
    } else {
        stream_stat(s,s->end,v[g-1],i[g-1],t[g-1]);
        if(s->family==2) {
            double x[]={v[0]-v[g-1],s->voltage.mean,sd(&s->voltage),s->absolute_current.mean,
              sd(&s->current),s->temperature.mean,sd(&s->temperature),t[g-1]-t[0],
              s->max_temperature,s->charge/3600,s->energy/3600};
            for(int k=0;k<11;k++)out[33+k]=(float)x[k];
        } else {
            double integral=0;for(int k=1;k<g;k++)integral+=.5*(v[k-1]+v[k])*s->step;
            double x[]={v[0]-v[g-1],s->voltage.mean,sd(&s->voltage),s->absolute_current.mean,
              sd(&s->current),s->charge/3600,s->energy/3600,(v[3]-v[0])/90,
              (v[g-1]-v[g-4])/90,v[0]-2*v[g/2]+v[g-1],quantile(v,g,.5),
              quantile(v,g,.75)-quantile(v,g,.25),integral,quantile(i,g,.5),
              quantile(i,g,.75)-quantile(i,g,.25),t[g-1]-t[0],
              s->temperature_ok?quantile(t,g,.5):NAN,
              s->temperature_ok?quantile(t,g,.75)-quantile(t,g,.25):NAN};
            for(int k=0;k<18;k++)out[33+k]=(float)x[k];
        }
    }
    s->finished=1;return 0;
}
