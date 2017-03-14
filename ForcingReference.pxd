cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics

cdef class ForcingReferenceBase:
    cdef:
        double [:] s
        double [:] qt
        double [:] temperature
        double [:] rv
        double [:] u
        double [:] v
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)



cdef class AdjustedMoistAdiabat(ForcingReferenceBase):
    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        Thermodynamics.ClausiusClapeyron CC
    cpdef get_pv_star(self, t)
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)


cdef class ReferenceRCE(ForcingReferenceBase):
    cdef:
        str filename
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)

cdef class InteractiveReferenceRCE(ForcingReferenceBase):
    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        Thermodynamics.ClausiusClapeyron CC
        double sst
        double RH_tropo
        double dt_rce
        Py_ssize_t index_tropopause
        double h_tropopause
        double ocean_heat_flux
        double sw_down_srf
        double sw_up_srf
        double lw_down_srf
        double lw_up_srf
        Py_ssize_t nlayers
        Py_ssize_t nlevels
        double [:] p_levels
        double [:] p_layers
        double [:] t_layers
        double [:] qv_layers
        double [:] t_tend_rad
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
    cpdef get_pv_star(self, double t)
    cpdef entropy_fixedRH(self, double p0, double T, double rh)
    cpdef initialize_radiation(self)
    cpdef compute_radiation(self)
    cpdef compute_rce(self)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)
