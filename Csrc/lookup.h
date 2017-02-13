#pragma once
#include <stdio.h>

struct LookupStruct{
    size_t n_table;
    float  y_max;
    float  y_min;
    float  x_max;
    float  x_min;
    float  dx;
    float  dxi;
    float * y;

};

void init_table(struct LookupStruct *LT, size_t n, float * x, float * y){


    /// Allocate memory required for storing the table
    LT->y = calloc(n,sizeof(float ));

    /// Determine bounds and set array table values
    LT->x_min = 9e16;
    LT->x_max =  -9e16;
    LT->y_min = 9e16;
    LT->y_max = -9e16;
    for(size_t i=0;i<n;i++){
        LT->x_max = fmax(LT->x_max,x[i]);
        LT->x_min = fmin(LT->x_min,x[i]);

        LT->y_max = fmax(LT->y_max,y[i]);
        LT->y_min = fmin(LT->y_min,y[i]);

        LT->y[i] = y[i];
    }

    LT->dx = x[1] - x[0];
    LT->dxi = 1.0/LT->dx;

};


void free_table(struct LookupStruct *LT){
    /// Free memory allocated in init_table
    free(LT->y);
};

inline float  lookup(struct LookupStruct *LT, float  x){
    const size_t indx = floor((x - LT->x_min)*LT->dxi);
    const float  y1 = LT->y[indx] ;
    const float  y2 = LT->y[indx + 1];
    const float  x1 = LT->x_min + LT-> dx * indx ;
    return y1 + (x - x1) * (y2 - y1)*LT->dxi;
};
