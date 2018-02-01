cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI

cdef class Forcing:
    cdef:
        object scheme
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingNone:
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
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
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingSullivanPatton:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingGabls:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
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
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
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
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingIsdac:
    cdef:
        double [:] dtdt
        double [:] dqtdt
        double [:] initial_u
        double [:] initial_v
        double [:] initial_entropy
        double [:] initial_qt
        double [:] nudge_coeff_velocities
        double [:] nudge_coeff_scalars
        double [:] w_half
        # double [:] ls_adv_Q
        double [:] ls_adv_qt
        double [:] ls_adv_t
        double divergence
        bint merra2

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingIsdacCC:
    cdef:

        double [:] initial_u
        double [:] initial_v
        double [:] initial_entropy
        double [:] initial_qt
        double [:] nudge_coeff_velocities
        double [:] nudge_coeff_scalars
        double [:] w_half

        double divergence
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingMpace:
    cdef:
        double divergence
        double [:] subsidence
        double [:] dtdt
        double [:] dqtdt
        double [:] u0
        double [:] v0
        double [:] nudge_coeff_velocities

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingSheba:
    cdef:

        double [:] subsidence
        double [:] dtdt
        double [:] dqtdt
        double [:] u0
        double [:] v0
        double [:] nudge_coeff_velocities

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, namelist)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)