from NetCDFIO cimport NetCDFIO_Stats
cimport Grid
cimport ParallelMPI
cimport ReferenceState
cimport Restart

cdef extern from "prognostic_variables.h":
    struct VelocityDofs:
        Py_ssize_t u
        Py_ssize_t v
        Py_ssize_t w

cdef extern from "prognostic_variables.h":
        void build_buffer(Py_ssize_t nv, Py_ssize_t dim, Py_ssize_t s ,Grid.DimStruct *dims, double* values, double* buffer)
        void buffer_to_values(Py_ssize_t dim, Py_ssize_t s, Grid.DimStruct *dims, double* values, double* buffer)
        void set_bcs(Py_ssize_t dim, Py_ssize_t s, double bc_factor,  Grid.DimStruct *dims, double* values)
        void set_to_zero(Py_ssize_t nv, Grid.DimStruct *dims, double* values )

cdef class PrognosticVariables:
    cdef:
        dict name_index
        dict units
        list index_name
        Py_ssize_t nv
        Py_ssize_t nv_scalars
        Py_ssize_t nv_velocities
        cdef double [:] bc_type
        cdef long [:] var_type
        cdef double [:] values
        cdef double [:] tendencies
        cdef long [:] velocity_directions
        list velocity_names_directional

    cpdef add_variable(self,name,units,bc_type,var_type,ParallelMPI.ParallelMPI Pa)
    cpdef initialize(self,Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cdef:
        void update_all_bcs(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef Update_all_bcs(self,Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef debug(self, Grid.Grid Gr, ReferenceState.ReferenceState RS ,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef set_velocity_direction(self,name,Py_ssize_t direction,ParallelMPI.ParallelMPI Pa)
    cdef inline Py_ssize_t get_nv(self, str variable_name):
        return self.name_index[variable_name]
    cdef inline Py_ssize_t get_varshift(self, Grid.Grid Gr, str variable_name):
        return self.name_index[variable_name] * Gr.dims.npg
    cpdef get_variable_array(self,name,Grid.Grid Gr)
    cpdef get_tendency_array(self,name,Grid.Grid Gr)
    cpdef tend_nan(self,PA,message)
    cpdef val_nan(self,PA,message)
    cpdef val_bounds(self,var_name,Grid.Grid Gr)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef restart(self, Grid.Grid Gr, Restart.Restart Re)
    cpdef init_from_restart(self, Grid.Grid Gr, Restart.Restart Re)