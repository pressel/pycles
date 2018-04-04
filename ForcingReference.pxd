cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics

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
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa,  double S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa,double S_minus_L)
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)

cdef class ForcingReferenceNone(ForcingReferenceBase):
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa,  double S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa,double S_minus_L)
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)


cdef class AdjustedMoistAdiabat(ForcingReferenceBase):
    cdef:
        double Tg
        double Pg
        double RH_ref
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa,  double S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa, double S_minus_L)


cdef class ReferenceRCE(ForcingReferenceBase):
    cdef:
        str filename
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa,  double S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa, double S_minus_L)

# cdef class InteractiveReferenceRCE_old(ForcingReferenceBase):
#     cdef:
#         bint fix_wv
#         str filename
#         double dt_rce
#         double [:] p_tropo_store
#         double [:] toa_store
#         double [:] tci_store
#         double RH_surf
#         double RH_tropical
#         double RH_subtrop
#         Py_ssize_t index_h
#         Py_ssize_t index_h_min
#         bint tropo_converged
#         double toa_flux
#         double total_column_influx
#         Py_ssize_t nlayers
#         Py_ssize_t nlevels
#         double [:] p_levels
#         double [:] p_layers
#         double [:] t_layers
#         double [:] qv_layers
#         double [:] t_tend_rad
#         double [:] o3vmr
#         double [:] co2vmr
#         double [:] ch4vmr
#         double [:] n2ovmr
#         double [:] o2vmr
#         double [:] cfc11vmr
#         double [:] cfc12vmr
#         double [:] cfc22vmr
#         double [:] ccl4vmr
#         double co2_factor
#         int dyofyr
#         double scon
#         double adjes
#         double solar_constant
#         double coszen
#         double adif
#         double adir
#         LookupProfiles t_table
#         LookupProfiles t_table_wv
#
#     cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
#     cpdef eos(self, double p0, double s, double qt)
#     cpdef initialize_radiation(self, double co2_factor)
#     cpdef compute_radiation(self)
#     cpdef update_qv(self, double p, double t, double rh)
#     cpdef compute_adiabat(self, double Tg, double Pg, double RH_surf)
#     cpdef rce_step(self, double Tg)
#     cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double sst_tropical,
#                      double S_minus_L)
#     cpdef update(self, double [:] pressure_array, double Tg)

cdef class InteractiveReferenceRCE_new(ForcingReferenceBase):
    cdef:
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
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa,  double S_minus_L)
    cpdef update(self, ParallelMPI.ParallelMPI Pa, double S_minus_L)


cdef class LookupProfiles:
    cdef:
        Py_ssize_t nprofiles
        Py_ssize_t nz
        double [:,:] table_vals
        double [:] access_vals
        double [:] profile_interp
    cpdef lookup(self, double val)
    cpdef communicate(self,  ParallelMPI.ParallelMPI Pa)