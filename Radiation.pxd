cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport TimeStepping
cimport Surface
from Forcing cimport AdjustedMoistAdiabat

cdef class RadiationBase:
    cdef:
        double [:] heating_rate
        double [:] dTdt_rad
        double [:] uflux_lw
        double [:] dflux_lw
        double [:] uflux_sw
        double [:] dflux_sw
        double [:] heating_rate_clear
        double [:] uflux_lw_clear
        double [:] dflux_lw_clear
        double [:] uflux_sw_clear
        double [:] dflux_sw_clear
        ParallelMPI.Pencil z_pencil
        double srf_lw_down
        double srf_lw_up
        double srf_sw_down
        double srf_sw_up
        double toa_lw_down
        double toa_lw_up
        double toa_sw_down
        double toa_sw_up

        double srf_lw_down_clear
        double srf_lw_up_clear
        double srf_sw_down_clear
        double srf_sw_up_clear
        double toa_lw_down_clear
        double toa_lw_up_clear
        double toa_sw_down_clear
        double toa_sw_up_clear



    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationNone(RadiationBase):
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS,ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationDyCOMS_RF01(RadiationBase):
    cdef:
        double alpha_z
        double kap
        double f0
        double f1
        double divergence

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur,TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationSmoke(RadiationBase):
    cdef:
        double f0
        double kap


    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationIsdac(RadiationBase):
    cdef:
        double kap
        double f0
        double f1
        double [:] radiative_flux

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class RadiationRRTM(RadiationBase):
    cdef:
        str profile_name
        bint modified_adiabat
        AdjustedMoistAdiabat reference_profile
        double Tg_adiabat
        double Pg_adiabat
        double RH_adiabat
        Py_ssize_t n_buffer
        Py_ssize_t n_ext
        double stretch_factor
        double patch_pressure
        double [:] p_ext
        double [:] t_ext
        double [:] rv_ext
        double [:] p_full
        double [:] pi_full


        double co2_factor
        double h2o_factor
        int dyofyr_init
        int dyofyr
        double scon
        double adjes
        double solar_constant
        double coszen
        double adif
        double adir
        double radiation_frequency
        double next_radiation_calculate
        bint daily_mean_sw
        double hourz_init
        double hourz
        double latitude
        double longitude

        double [:] o3vmr
        double [:] co2vmr
        double [:] ch4vmr
        double [:] n2ovmr
        double [:] o2vmr
        double [:] cfc11vmr
        double [:] cfc12vmr
        double [:] cfc22vmr
        double [:] ccl4vmr
        bint uniform_reliq

        double [:, :] uflx_lw_pencils
        double [:, :] dflx_lw_pencils
        double [:, :] uflx_sw_pencils
        double [:, :] dflx_sw_pencils
        double IsdacCC_dT


    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cdef update_RRTM(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur,ParallelMPI.ParallelMPI Pa)

    cpdef stats_io(self, Grid.Grid Gr,  ReferenceState.ReferenceState Ref, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
