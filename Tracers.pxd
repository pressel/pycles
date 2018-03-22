cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI

from NetCDFIO cimport NetCDFIO_Stats
import cython

from libc.math cimport sqrt, log, fabs,atan, exp, fmax
cimport numpy as np
import numpy as np
include "parameters.pxi"

import cython



cdef class TracersNone:

    cpdef initialize(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef update_cleanup(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)



cdef class UpdraftTracers:
    cdef:
        bint lcl_tracers
        Py_ssize_t index_lcl
        dict tracer_dict

    cpdef initialize(self, Grid.Grid Gr,  PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef update_cleanup(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

