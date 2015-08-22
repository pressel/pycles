from NetCDFIO cimport NetCDFIO_Stats
cimport Grid
cimport ParallelMPI

cdef extern from "prognostic_variables.h":
    struct VelocityDofs:
        int u
        int v
        int w

cdef extern from "prognostic_variables.h":
        void build_buffer(int nv, int dim, int s ,Grid.DimStruct *dims, double* values, double* buffer)
        void buffer_to_values(int dim, int s, Grid.DimStruct *dims, double* values, double* buffer)
        void set_bcs(int dim, int s, double bc_factor,  Grid.DimStruct *dims, double* values)
        void set_to_zero(int nv, Grid.DimStruct *dims, double* values )

cdef class PrognosticVariables:
    cdef:
        dict name_index
        dict units
        int nv
        int nv_scalars
        int nv_velocities
        cdef double [:] bc_type
        cdef long [:] var_type
        cdef double [:] values
        cdef double [:] tendencies
        cdef long [:] velocity_directions

    cpdef add_variable(self,name,units,bc_type,var_type,ParallelMPI.ParallelMPI Pa)
    cpdef initialize(self,Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cdef:
        void update_all_bcs(self, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef Update_all_bcs(self,Grid.Grid Gr, ParallelMPI.ParallelMPI Pa)
    cpdef set_velocity_direction(self,name,int direction,ParallelMPI.ParallelMPI Pa)
    cdef inline int get_nv(self, str variable_name):
        return self.name_index[variable_name]
    cdef inline int get_varshift(self, Grid.Grid Gr, str variable_name):
        return self.name_index[variable_name] * Gr.dims.npg
    cpdef get_variable_array(self,name,Grid.Grid Gr)
    cpdef get_tendency_array(self,name,Grid.Grid Gr)
    cpdef tend_nan(self,PA,message)
    cpdef val_nan(self,PA,message)
    cpdef val_bounds(self,var_name,Grid.Grid Gr)
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
