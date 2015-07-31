#pragma once
struct DimStruct {

        long dims;

        long  n[3];
        long  ng[3];
        long  nl[3];
        long  nlg[3];

        long npd;
        long npl;
        long npg;
        long gw;

        long  indx_lo[3];
        long  indx_lo_g[3];

        long nbuffer[3];
        long ghosted_stride[3];

        double dx[3];
        double dxi[3];};