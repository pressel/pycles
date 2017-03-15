#pragma once
struct DimStruct {

        long dims; /// number of physical dimensions

        long  n[3];  /// Total number of grid points in grid direction
        long  ng[3];  /// ""           ""           ""    including ghost points
        long  nl[3];  /// Total number of local grid points in grid direction  owned by local MPI rank
        long  nlg[3]; /// ""           ""           ""    including ghost points

        long npd;  /// Total grid points in domain not including ghost points
        long npl;  /// Number of grid points not including ghost points owned by local MPI rank
        long npg;  /// Number of grid points including ghost point owned by local MPI rank
        long gw;  /// Number of ghost/halo points

        long  indx_lo[3]; /// Lower left hand point (globally numbered) of subgrid owned by local MPI rank
        long  indx_lo_g[3];/// Lower left hand point (globally numbered ) of subgrid owned by local MPI rank including ghostpoints

        long nbuffer[3]; /// Number of points in buffer for ghostpoint update
        long ghosted_stride[3];

        double zp_half_0;
        double zp_0;

        double dx[3]; /// Grid spacing in grid direction
        double dxi[3]; // Inverse gird spacing in grid direction

        double *met;
        double *imet;
        double *met_half;
        double *imet_half;

        double *metl;
        double *imetl;
        double *metl_half;
        double *imetl_half;

        double *zpl;
        double *zpl_half;

        double *dzpl_half;
        double *dzpl;
        };
