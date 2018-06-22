cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
from libc.math cimport pow, exp

from Thermodynamics cimport ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Fields, NetCDFIO_Stats

cdef:
    double lambda_constant_Arctic(double T) nogil
    double latent_heat_constant_Arctic(double T, double Lambda) nogil
    double lambda_Arctic(double T) nogil
    double latent_heat_Arctic(double T, double Lambda) nogil
    double latent_heat_variable_Arctic(double T, double Lambda) nogil

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

cdef inline double lambda_Hu2010(double T) nogil:
    #Equation 1 and 2 from Hu et al. (2010)
    cdef:
        double p
        double TC = T - 273.15 #Convert to Celcius

    p = 5.3608 + 0.4025 * TC + 0.08387 * TC * TC + 0.007182 * TC * TC * TC + \
        2.39e-4 * TC * TC * TC * TC + 2.87e-6 * TC * TC * TC * TC * TC
    return 1.0 / (1.0 + exp(-p))

cdef inline double lambda_Cesana(double T) nogil:
    #Emperical polynomial derived from Greg Cesana based on CALIPSO-GOCCP for the Arctic ocean
    #Night time only observations of cloud phase
    cdef:
        double TC = T - 273.15
        double p

    if -40.0 <= TC < 0.0:
        p = 0.4847323 + 0.0313655*TC + 0.001268696*TC*TC - 0.0000225937*TC*TC*TC - 6.731156e-7*TC*TC*TC*TC
    elif TC >= 0.0:
        p = 0.0
    else:
        p = 1.0

    return 1.0 - p


cdef inline double lambda_logistic(double T) nogil:
    #Cubic logistic function fitted to the Hu et al. (2009) expression
    #(Temperature is in Kelvin)
    cdef:
        double k = 0.32928
        double Tlf = 241.92
        double Twarm = 273.15
        double Tcold = 235.0
        double Lambda = 0.0

    if Tcold <= T <= Twarm:
        Lambda = 1.0/pow((1.0 + exp(-k*(T - Tlf))), 3)
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


cdef class Microphysics_Arctic_1M:


    cdef public:
        str thermodynamics_type

    cdef:
        double ccn
        double n0_ice_input
        Py_ssize_t order

        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil

        ClausiusClapeyron CC

        double [:] evap_rate
        double [:] precip_rate
        double [:] melt_rate


    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef ice_stats(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

