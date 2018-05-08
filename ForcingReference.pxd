cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from TimeStepping cimport TimeStepping
cimport Thermodynamics
from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats

cdef class ForcingReferenceBase:
    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        Thermodynamics.ClausiusClapeyron CC
        double sst
        Py_ssize_t npressure
        double [:] pressure
        double [:] s
        double [:] qt
        double [:] temperature
        double [:] rv
        double [:] u
        double [:] v
        bint is_init
        bint adjust_S_minus_L
        double S_minus_L_fixed_val
        double reference_S_minus_L_subtropical
        double subtropical_area_fraction
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa,double S_minus_L, TimeStepping TS)
    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef restart(self, Restart)

cdef class ForcingReferenceNone(ForcingReferenceBase):
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa,double S_minus_L, TimeStepping TS)
    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)
    cpdef restart(self, Restart)

cdef class AdjustedMoistAdiabat(ForcingReferenceBase):
    cdef:
        double Tg
        double Pg
        double RH_ref
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa, double S_minus_L, TimeStepping TS)
    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef restart(self, Restart)

cdef class ReferenceRCE(ForcingReferenceBase):
    cdef:
        str filename
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa, double S_minus_L, TimeStepping TS)
    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef restart(self, Restart)


cdef class InteractiveReferenceRCE_new(ForcingReferenceBase):
    cdef:
        bint read_pkl
        str pkl_file
        double dt_rce
        Py_ssize_t nlayers
        Py_ssize_t nlevels
        double p_surface
        double RH_tropical
        double RH_subtrop
        double RH_surface
        str lapse_rate_type
        double  delta_T_max
        double delta_T
        double toa_error_max
        double toa_update_criterion
        double toa_update_timescale
        double max_steps
        double ohu
        double net_toa_target
        double net_toa_computed
        double [:] p_levels
        double [:] p_layers
        double [:] t_layers
        double [:] qv_layers
        double [:] dTdt_rad_lw
        double [:] dTdt_rad_sw
        double [:] uflux_lw
        double [:] dflux_lw
        double [:] uflux_sw
        double [:] dflux_sw
        double [:] o3vmr
        double [:] co2vmr
        double [:] ch4vmr
        double [:] n2ovmr
        double [:] o2vmr
        double [:] cfc11vmr
        double [:] cfc12vmr
        double [:] cfc22vmr
        double [:] ccl4vmr
        double co2_factor
        int dyofyr
        double scon
        double adjes
        double solar_constant
        double coszen
        double adif
        double adir
        bint verbose
        str out_dir
        double first_guess_sst
        double first_guess_tropo

    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize_radiation(self)
    cpdef compute_radiation(self)
    cpdef update_qv(self, double p, double t, double rh, double maxval)
    cpdef initialize_adiabat(self)
    cpdef convective_adjustment(self)
    cdef lapse_rate(self,double p, double T, double *qt)
    cdef lapse_rate_saturated(self,double p, double T, double *qt)
    cdef lapse_rate_subsaturated(self,double p, double T, double *qt)
    cdef lapse_rate_mixed(self,double p, double T, double *qt)
    cpdef rce_fixed_toa(self, ParallelMPI.ParallelMPI Pa)
    cpdef initialize(self, Grid Gr, ParallelMPI.ParallelMPI Pa, NetCDFIO_Stats NS, double  S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa, double S_minus_L, TimeStepping TS)
    cpdef stats_io(self, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef restart(self, Restart)

cdef class LookupProfiles:
    cdef:
        Py_ssize_t nprofiles
        Py_ssize_t nz
        double [:,:] table_vals
        double [:] access_vals
        double [:] profile_interp
    cpdef lookup(self, double val)
    cpdef communicate(self,  ParallelMPI.ParallelMPI Pa)