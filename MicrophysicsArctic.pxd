cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
from libc.math cimport pow

from Thermodynamics cimport ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Fields, NetCDFIO_Stats

cdef:
    double lambda_constant_Arctic(double T) nogil
    double latent_heat_constant_Arctic(double T, double T) nogil
    double lambda_Arctic(double T) nogil
    double latent_heat_Arctic(double T, double Lambda) nogil


cdef inline double lambda_constant_Arctic(double T) nogil:
    return 1.0

cdef inline double latent_heat_constant_Arctic(double T, double Lambda) nogil:
    return 2.501e6

cdef inline double lambda_Arctic(double T) nogil:
    cdef:
        double Twarm = 273.0
        double Tcold = 235.0
        double pow_n = 0.1
        double Lambda = 0.0

    if T >= Tcold and T <= Twarm:
        Lambda = pow((T - Tcold)/(Twarm - Tcold), pow_n)
    elif T > Twarm:
        Lambda = 1.0
    else:
        Lambda = 0.0

    return Lambda

cdef inline double latent_heat_Arctic(double T, double Lambda) nogil:
    cdef:
        double Lv = 2.501e6
        double Ls = 2.8334e6

    return (Lv * Lambda) + (Ls * (1.0 - Lambda))

cdef inline double latent_heat_variable_Arctic(double T, double Lambda) nogil:
    cdef:
        double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0

cdef extern from "micro_parameters.h":
    struct hm_parameters:
        double a
        double b
        double c
        double d
        double gb1
        double gbd1
        double gd3
        double gd6
        double gamstar
        double alpha_acc
        double d_min
        double d_max

    struct hm_properties:
        double mf
        double diam
        double vel
        double lam
        double n0

    struct ret_acc:
        double dyr
        double dys
        double dyl
        double dyi

cdef class MicrophysicsArctic:


    cdef public:
        str thermodynamics_type

    cdef:
        double ccn
        double n0_ice

        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil

        ClausiusClapeyron CC

        hm_properties rain_prop
        hm_properties snow_prop
        hm_properties ice_prop
        hm_properties liquid_prop
        ret_acc src_acc

        double [:] autoconversion
        double [:] evaporation
        double [:] accretion
        double [:] melting
        double [:] liquid_fraction


    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

