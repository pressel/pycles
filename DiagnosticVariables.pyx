cimport ParallelMPI
cimport Grid
import numpy as np
cimport numpy as np
cimport ParallelMPI
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport mpi4py.mpi_c as mpi

cdef extern from "prognostic_variables.C":
        void build_buffer(int nv, int dim, int s ,Grid.DimStruct *dims, double* values, double* buffer)

cdef extern from "prognostic_variables.C":
        void buffer_to_values(int dim, int s, Grid.DimStruct *dims, double* values, double* buffer)

cdef extern from "prognostic_variables.C":
        void set_bcs( int dim, int s, double bc_factor,  Grid.DimStruct *dims, double* values)


cdef class DiagnosticVariables:
    def __init__(self):
        self.name_index = {}
        self.units = {}
        self.nv = 0
        self.bc_type = np.array([],dtype=np.double,order='c')


    cpdef add_variables(self, name, units,bc_type,  ParallelMPI.ParallelMPI Pa):
        self.name_index[name] = self.nv
        self.units[name] = units


        #Add bc type to array
        if bc_type == "sym":
            self.bc_type = np.append(self.bc_type,[1.0])
        elif bc_type =="asym":
            self.bc_type = np.append(self.bc_type,[-1.0])
        else:
            Pa.root_print("Not a valid bc_type. Killing simulation now!")
            Pa.kill()



        self.nv = len(self.name_index.keys())

        return

    cpdef initialize(self,Grid.Grid Gr):

        self.values = np.zeros((self.nv*Gr.dims.npg),dtype=np.double,order='c')

        return

    cdef void communicate_variable(self, Grid.Grid Gr, ParallelMPI.ParallelMPI PM, long nv):

        cdef:
            double*  send_buffer
            double*  recv_buffer
            long d, s
            long var_shift, buffer_var_shift
            long [:] shift = np.array([-1,1],dtype=np.int,order='c')
            int ierr, source_rank, dest_rank
            mpi.MPI_Status status

        ierr = mpi.MPI_Comm_rank(PM.cart_comm_world,&source_rank)


        #print(self.name_index)

        var_shift = (nv) * Gr.dims.npg
        for d in xrange(Gr.dims.dims):
            buffer_var_shift = 0
            #Allocate memory for send and recv buffers.
            send_buffer = <double*> PyMem_Malloc(Gr.dims.nbuffer[d] * sizeof(double))
            recv_buffer = <double*> PyMem_Malloc(Gr.dims.nbuffer[d] * sizeof(double))
            for s in shift:

                #Since we are only sending one variable at a time the first argument in buld_buffer should be 0
                # let's clean this up later
                build_buffer(0, d, s,&Gr.dims,&self.values[var_shift],&send_buffer[0])

                #Determine the MPI shifts
                ierr = mpi.MPI_Cart_shift(PM.cart_comm_world,d,s,&source_rank,&dest_rank)


                #Do send and recv given shift
                ierr = mpi.MPI_Sendrecv(&send_buffer[0],Gr.dims.nbuffer[d],mpi.MPI_DOUBLE,dest_rank,0,
                                            &recv_buffer[0],Gr.dims.nbuffer[d],
                                            mpi.MPI_DOUBLE,source_rank,0,PM.cart_comm_world,&status)

                #If communicated values are to be used copy them into the correct location
                if source_rank >= 0:
                    buffer_to_values(d, s,&Gr.dims,&self.values[var_shift],&recv_buffer[0])
                #If communicated values are not be used, set numerical boundary consitions
                else:
                    set_bcs(d,s,self.bc_type[nv],&Gr.dims,&self.values[var_shift])


            #Important: Free memory associated with memory buffer to prevent memory leak
            PyMem_Free(send_buffer)
            PyMem_Free(recv_buffer)


        return

    cpdef get_variable_array(self,name,Grid.Grid Gr):
        index = self.name_index[name]
        view = np.array(self.values).view()
        print self.nv*Gr.dims.nlg[0]*Gr.dims.nlg[1]*Gr.dims.nlg[2], np.shape(self.values)
        view.shape = (self.nv,Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2])
        return view[index,:,:,:]