#pragma once
struct DimStruct {

        long dims; /// number of physical dimensions

        long  n[3];  /// Total number of grid points in grid direction
        long  ng[3];  /// ""           ""           ""    Including ghost points
        long  nl[3];  /// number of local grid points owned by local MPI rank
        long  nlg[3]; /// ""           ""           ""    Including ghost points

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
