cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from ForcingReference cimport *
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport Surface
from TimeStepping cimport TimeStepping
cimport Radiation
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics



cdef extern from "grid.h":
    struct DimStruct:

        int dims

        int [3] n
        int [3] ng
        int [3] nl
        int [3] nlg
        int [3] indx_lo_g
        int [3] indx_lo

        int npd
        int npl
        int npg
        int gw

        int [3] nbuffer
        int [3] ghosted_stride

        double [3] dx
        double [3] dxi


cdef class Forcing:
    cdef:
        object scheme
    cpdef initialize(self, Grid Gr, ReferenceState.ReferenceState Ref,  Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra, TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingNone:
    cpdef initialize(self, Grid Gr, ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra, TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingBomex:
    cdef:
        double [:] ug
        double [:] vg
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double coriolis_param
    cpdef initialize(self, Grid Gr,ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra, TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingSullivanPatton:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param
    cpdef initialize(self, Grid Gr,ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra, TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingGabls:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param
    cpdef initialize(self, Grid Gr,ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur,Radiation.RadiationBase Ra, TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingDyCOMS_RF01:
    cdef:
        double [:] ug
        double [:] vg
        double divergence
        double [:] subsidence
        double coriolis_param
        bint rf02_flag
    cpdef initialize(self, Grid Gr, ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra,TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class ForcingRico:
    cdef:
        double [:] ug
        double [:] vg
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double coriolis_param
        Py_ssize_t momentum_subsidence
    cpdef initialize(self, Grid Gr,ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra,TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class ForcingCGILS:
    cdef:
        Py_ssize_t loc
        bint is_p2
        bint is_ctl_omega
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double z_relax
        double z_relax_plus
        double tau_inverse
        double tau_vel_inverse
        double qt_floor
        Py_ssize_t floor_index
        double[:] gamma_zhalf
        double[:] gamma_z
        double [:] nudge_qt
        double [:] nudge_temperature
        double [:] nudge_u
        double [:] nudge_v
        double [:] source_qt_floor
        double [:] source_qt_nudge
        double [:] source_t_nudge
        double [:] source_u_nudge
        double [:] source_v_nudge
        double [:] source_s_nudge
        double [:] s_ls_adv

    cpdef initialize(self, Grid Gr,ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra, TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)




cdef class ForcingZGILS:
    cdef:
        Py_ssize_t loc
        bint varsub
        double varsub_factor
        bint adjust_t_adv
        double co2_factor
        double n_double_co2
        str reference_type
        double qt_ls_factor
        double t_ls_factor
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double [:] ug
        double [:] vg
        double [:] reference_qt
        double [:] reference_t
        double [:] source_qt_nudge
        double [:] source_t_nudge
        double [:] source_s_nudge
        double [:] source_u_nudge
        double [:] source_v_nudge
        double [:] s_ls_adv
        double coriolis_param
        double divergence
        double t_adv_max
        double qt_adv_max
        double tau_relax_inverse
        double alpha_h
        double h_BL
        double SST_1xCO2
        double divergence_factor
        ClausiusClapeyron CC


    cpdef initialize(self, Grid Gr,ReferenceState.ReferenceState Ref, Surface.SurfaceBase Sur,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef update(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, Radiation.RadiationBase Ra, TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa, ForcingReferenceBase FoRef)
    cpdef stats_io(self, Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

