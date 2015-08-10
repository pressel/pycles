cimport ParallelMPI
cimport Grid
cdef class DiagnosticVariables:
    cdef:
        dict name_index
        dict units
        int nv
        double [:] values
        double [:] bc_type

        void communicate_variable(self,Grid.Grid Gr,ParallelMPI.ParallelMPI PM, long nv)

    cpdef add_variables(self, name, units, bc_type, ParallelMPI.ParallelMPI Pa)

    cpdef initialize(self,Grid.Grid Gr)

    cpdef get_variable_array(self,name,Grid.Grid Gr)

    cdef inline int get_nv(self, str variable_name):
        return self.name_index[variable_name]

    cdef inline int get_varshift(self, Grid.Grid Gr, str variable_name):
        return self.name_index[variable_name] * Gr.dims.npg

    cpdef val_nan(self,PA,message)