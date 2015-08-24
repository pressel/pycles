#pragma once

struct StatsProfileStruct{
    long dim;
    long sum_n;
    double* data ;
};

void init_stats_profile_struct(struct StatsProfileStruct *PS, long dim){
    PS->dim = dim;
    PS->sum_n = 0;
    PS->data = calloc(dim, sizeof(double));

    for(long i=0; i<PS->dim; i++){
        PS->data[i] = 0.0;
    }
}

void stats_profile_reset(struct StatsProfileStruct *PS){
    for (long i=0; i<PS->dim; i++){
        PS->data[i] = 0.0;
    }
    PS->sum_n = 0;
}

void stats_profile_struct_free(struct StatsProfileStruct *PS){
    // Free memory allocated in init_stats_profile_struct
    free(PS->data);
}