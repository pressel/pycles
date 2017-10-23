cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics

cdef class ForcingReferenceBase:
    cdef:
        double sst
        double [:] s
        double [:] qt
        double [:] temperature
        double [:] rv
        double [:] u
        double [:] v
        bint is_init
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)
    cpdef update(self, double [:] pressure_array, double Tg)



cdef class AdjustedMoistAdiabat(ForcingReferenceBase):
    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        Thermodynamics.ClausiusClapeyron CC
    cpdef get_pv_star(self, t)
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)
    cpdef update(self, double [:] pressure_array, double Tg)


cdef class ReferenceRCE(ForcingReferenceBase):
    cdef:
        str filename
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)
    cpdef update(self, double [:] pressure_array, double Tg)

cdef class InteractiveReferenceRCE(ForcingReferenceBase):
    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        Thermodynamics.ClausiusClapeyron CC
        double dt_rce
        double [:] p_tropo_store
        double [:] toa_store
        double [:] tci_store
        double RH_surf
        double RH_tropical
        double RH_subtrop
        Py_ssize_t index_h
        Py_ssize_t index_h_min
        bint tropo_converged
        double toa_flux
        double total_column_influx
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
        LookupProfiles t_table

    cpdef get_pv_star(self, double t)
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize_radiation(self)
    cpdef compute_radiation(self)
    cpdef update_qv(self, double p, double t, double rh)
    cpdef compute_adiabat(self, double Tg, double Pg, double RH_surf)
    cpdef rce_step(self, double Tg)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, double Pg, double Tg, double RH)
    cpdef update(self, double [:] pressure_array, double Tg)



cdef class LookupProfiles:
    cdef:
        Py_ssize_t nprofiles
        Py_ssize_t nz
        double [:,:] table_vals
        double [:] access_vals
        double [:] profile_interp
    cpdef lookup(self, double val)
    cpdef communicate(self,  ParallelMPI.ParallelMPI Pa)