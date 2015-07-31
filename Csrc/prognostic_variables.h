#pragma once
#include "grid.h"

#pragma once
struct VelocityDofs {
long u;
long v;
long w;
};

void build_buffer(long nv, long dim, long s, struct DimStruct *dims, double *values, double *buffer);

void buffer_to_values(long dim, long s, struct DimStruct *dims, double *values, double *buffer);

void set_bcs(long dim, long s, double bc_factor, struct DimStruct *dims, double *values);

void set_to_zero(long nv, struct DimStruct *dims, double *array);