cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
cimport Kinematics
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats

cdef class SGS:
    cdef:
        object scheme

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV,Kinematics.Kinematics Ke, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                   PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)



cdef class UniformViscosity:

    cdef:
        double const_viscosity
        double const_diffusivity
        bint is_init 

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                   PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class Smagorinsky:
    cdef:
        double cs
        double prt

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke,  ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                   PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)



cdef class TKE:
    cdef:
        double ck
        double cn
        ParallelMPI.Pencil Z_Pencil

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                 PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
                   PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


