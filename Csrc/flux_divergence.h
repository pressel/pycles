#pragma once
#include "grid.h"

void scalar_flux_divergence(struct DimStruct *dims, double *alpha0, double *alpha0_half, double *flux, double *tendency,
    double dx, size_t d){

    size_t imin = dims->gw;
    size_t jmin = dims->gw;
    size_t kmin = dims->gw;

    size_t imax = dims->nlg[0] - dims->gw;
    size_t jmax = dims->nlg[1] - dims->gw;
    size_t kmax = dims->nlg[2] - dims->gw;



}