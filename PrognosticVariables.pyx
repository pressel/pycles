#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import numpy as np
cimport numpy as np
cimport mpi4py.mpi_c as mpi
from NetCDFIO cimport NetCDFIO_Stats

cimport Grid
cimport ParallelMPI
cimport ReferenceState


cdef class PrognosticVariables:
    def __init__(self, Grid.Grid Gr):
        self.name_index = {}
        self.index_name = []
        self.units = {}
        self.nv = 0
        self.nv_scalars = 0
        self.nv_velocities = 0
        self.bc_type = np.array([],dtype=np.double,order='c')
        self.var_type = np.array([],dtype=np.int,order='c')
        self.velocity_directions = np.zeros((Gr.dims.dims,),dtype=np.int,order='c')
        return

    cpdef add_variable(self,name,units,bc_type,var_type,ParallelMPI.ParallelMPI Pa):

        #Store names and units
        self.name_index[name] = self.nv
        self.index_name.append(name)
        self.units[name] = units
        self.nv = len(self.name_index.keys())

        #Add bc type to array
        if bc_type == "sym":
            self.bc_type = np.append(self.bc_type,[1.0])
        elif bc_type =="asym":
            self.bc_type = np.append(self.bc_type,[-1.0])
        else:
            Pa.root_print("Not a valid bc_type. Killing simulation now!")
            Pa.kill()

        #Set the type of the variable being added 0=velocity; 1=scalars
        if var_type == "velocity":
            self.var_type = np.append(self.var_type,0)
            self.nv_velocities += 1
        elif var_type == "scalar":
            self.var_type = np.append(self.var_type,1)
            self.nv_scalars += 1
        else:
            Pa.root_print("Not a vaild var_type. Killing simulation now!")
            Pa.kill()

        return

    cpdef set_velocity_direction(self,name,Py_ssize_t direction,ParallelMPI.ParallelMPI Pa):
        try:
            self.velocity_directions[direction] = self.get_nv(name)
        except:
            Pa.root_print('problem setting velocity '+ name+' to direction '+ str(direction))
            Pa.root_print('Killing simulation now!')
            Pa.kill()
        return

    cpdef initialize(self,Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.values = np.empty((self.nv*Gr.dims.npg),dtype=np.double,order='c')
        self.tendencies = np.zeros((self.nv*Gr.dims.npg),dtype=np.double,order='c')

        #Add prognostic variables to Statistics IO
        Pa.root_print('Setting up statistical output files for Prognostic Variables')
        for var_name in self.name_index.keys():
            #Add mean profile
            NS.add_profile(var_name+'_mean',Gr,Pa)

            if var_name == 'u' or var_name == 'v':
                NS.add_profile(var_name+'_translational_mean',Gr,Pa)

            #Add mean of squares profile
            NS.add_profile(var_name+'_mean2',Gr,Pa)
            #Add mean of cubes profile
            NS.add_profile(var_name+'_mean3',Gr,Pa)
            #Add max profile
            NS.add_profile(var_name+'_max',Gr,Pa)
            #Add min profile
            NS.add_profile(var_name+'_min',Gr,Pa)
            #Add max ts
            NS.add_ts(var_name+'_max',Gr,Pa)
            #Add min ts
            NS.add_ts(var_name+'_min',Gr,Pa)
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS ,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t var_shift
            double [:] tmp

        for var_name in self.name_index.keys():
            var_shift = self.get_varshift(Gr,var_name)


            #Compute and write mean
            tmp = Pa.HorizontalMean(Gr,&self.values[var_shift])
            NS.write_profile(var_name + '_mean',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

            # Also output the velocities with the translational velocity included
            if var_name == 'u':
                NS.write_profile(var_name + '_translational_mean',np.array(tmp[Gr.dims.gw:-Gr.dims.gw]) + RS.u0,Pa)
            elif var_name == 'v':
                NS.write_profile(var_name + '_translational_mean',np.array(tmp[Gr.dims.gw:-Gr.dims.gw]) + RS.v0,Pa)


            #Compute and write mean of squres
            tmp = Pa.HorizontalMeanofSquares(Gr,&self.values[var_shift],&self.values[var_shift])
            NS.write_profile(var_name + '_mean2',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
            #Compute and write mean of cubes
            tmp = Pa.HorizontalMeanofCubes(Gr,&self.values[var_shift],&self.values[var_shift],&self.values[var_shift])
            NS.write_profile(var_name + '_mean3',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)

            #Compute and write maxes
            tmp = Pa.HorizontalMaximum(Gr,&self.values[var_shift])
            NS.write_profile(var_name + '_max',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
            NS.write_ts(var_name+'_max',np.amax(tmp[Gr.dims.gw:-Gr.dims.gw]),Pa)

            #Compute and write mins
            tmp = Pa.HorizontalMinimum(Gr,&self.values[var_shift])
            NS.write_profile(var_name + '_min',tmp[Gr.dims.gw:-Gr.dims.gw],Pa)
            NS.write_ts(var_name+'_min',np.amin(tmp[Gr.dims.gw:-Gr.dims.gw]),Pa)

        return

    cdef void update_all_bcs(self,Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):

        cdef double* send_buffer
        cdef double* recv_buffer
        cdef double a =0
        cdef double b = 0
        cdef Py_ssize_t [:] shift = np.array([-1,1],dtype=np.int,order='c')
        cdef Py_ssize_t d, i, s
        cdef Py_ssize_t ierr
        cdef int dest_rank, source_rank
        cdef mpi.MPI_Status status

        #Get this processors rank in the cart_comm_world communicator
        ierr = mpi.MPI_Comm_rank(Pa.cart_comm_world,&source_rank)
        cdef Py_ssize_t j,k,var_shift,ishift, jshift, buffer_var_shift

        #Loop over dimensions sending buffers for each
        for d in xrange(Gr.dims.dims):

            #Allocate memory for send buffer using python memory manager for safety
            send_buffer = <double*> PyMem_Malloc(self.nv * Gr.dims.nbuffer[d] * sizeof(double))
            recv_buffer = <double*> PyMem_Malloc(self.nv * Gr.dims.nbuffer[d] * sizeof(double))
            #Loop over shifts (this should only be -1 or 1)
            for s in shift:
                #Now loop over variables and store in send buffer
                for i in xrange(self.nv):
                    buffer_var_shift = Gr.dims.nbuffer[d] * i
                    var_shift = i * Gr.dims.npg
                    build_buffer(i, d, s,&Gr.dims,&self.values[0],&send_buffer[0])

                #Compute the mpi shifts (lower and upper) in the world communicator for dimeniosn d
                ierr = mpi.MPI_Cart_shift(Pa.cart_comm_world,d,s,&source_rank,&dest_rank)

                ierr = mpi.MPI_Sendrecv(&send_buffer[0],self.nv*Gr.dims.nbuffer[d],mpi.MPI_DOUBLE,dest_rank,0,
                                            &recv_buffer[0],self.nv*Gr.dims.nbuffer[d],
                                            mpi.MPI_DOUBLE,source_rank,0,Pa.cart_comm_world,&status)

                for i in xrange(self.nv):
                    buffer_var_shift = Gr.dims.nbuffer[d] * i
                    var_shift = i * Gr.dims.npg
                    if source_rank >= 0:
                        buffer_to_values(d, s,&Gr.dims,&self.values[var_shift],&recv_buffer[buffer_var_shift])
                    else:
                        set_bcs(d,s,self.bc_type[i],&Gr.dims,&self.values[var_shift])

            #Important: Free memory associated with memory buffer to prevent memory leak
            PyMem_Free(send_buffer)
            PyMem_Free(recv_buffer)
        return

    cpdef Update_all_bcs(self,Grid.Grid Gr, ParallelMPI.ParallelMPI Pa ):
          self.update_all_bcs(Gr, Pa)
          return

    cpdef get_variable_array(self,name,Grid.Grid Gr):
        index = self.name_index[name]
        view = np.array(self.values).view()
        view.shape = (self.nv,Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2])
        return view[index,:,:,:]

    cpdef get_tendency_array(self,name,Grid.Grid Gr):
        index = self.name_index[name]
        view = np.array(self.tendencies).view()
        view.shape = (self.nv,Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2])
        return view[index,:,:,:]

    cpdef tend_nan(self,PA,message):
        if np.isnan(self.tendencies).any():
            print('Nans found in tendencies')
            print(message)
            PA.kill()
        return

    cpdef val_nan(self,PA,message):
        if np.isnan(self.values).any():
            print('Nans found in Prognostic Variables values')
            print(message)
            PA.kill()
        return

    cpdef val_bounds(self,var_name,Grid.Grid Gr):
        var_array = self.get_variable_array(var_name, Gr)
        return np.amin(var_array), np.amax(var_array)
