#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport mpi4py.libmpi as mpi
cimport Grid
from time import time
import sys

import numpy as np
cimport numpy as np
import cython
from libc.math cimport fmin, fmax
cdef class ParallelMPI:
    def __init__(self,namelist):
        '''
        Initializes the ParallelMPI class. Calls MPI init. Sets-up MPI cartesian topologies and sub-topologies.
        :param namelist: Namelist dictionary.
        :return:
        '''

        cdef:
            int is_initialized
            int ierr = 0

        #Check to see if MPI_Init has been called if not do so
        ierr = mpi.MPI_Initialized(&is_initialized)
        if not is_initialized:
            from mpi4py import MPI
        self.comm_world =  mpi.MPI_COMM_WORLD
        ierr = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD, &self.rank)
        ierr = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD, &self.size)

        cdef:
            int [3] cart_dims
            int [3] cyclic
            int ndims = 3
            int reorder = 1

        cart_dims[0] = namelist['mpi']['nprocx']
        cart_dims[1] = namelist['mpi']['nprocy']
        cart_dims[2] = namelist['mpi']['nprocz']

        #Check to make sure that cart dimensions are consistent with MPI global size
        if cart_dims[0] * cart_dims[1] * cart_dims[2] != self.size:
            self.root_print('MPI global size: ' + str(self.size) +
                            'does not equal nprocx * nprocy * nprocz: '
                            + str(cart_dims[0] * cart_dims[1] * cart_dims[2]))
            self.root_print('Killing simulation NOW!')
            self.kill()

        cyclic[0] = 1
        cyclic[1] = 1
        cyclic[2] = 0

        #Create the cartesian world commmunicator
        ierr = mpi.MPI_Cart_create(self.comm_world,ndims, cart_dims, cyclic, reorder,&self.cart_comm_world)
        self.barrier()

        #Create the cartesian sub-communicators
        self.create_sub_communicators()
        self.barrier()

        return

    cpdef root_print(self,txt_output):
        '''
        Print only from the root process.
        :param txt_output: Output
        :return:
        '''
        if self.rank==0:
            print(txt_output)
        return

    cpdef kill(self):
        '''
        Call MPI_Abort.
        :return:
        '''
        cdef int ierr = 0
        self.root_print("Terminating MPI!")
        ierr = mpi.MPI_Abort(self.comm_world,1)
        sys.exit()
        return

    cdef void barrier(self):
        '''
        Call MPI_Barrier on global MPI communicator.
        :return:
        '''
        mpi.MPI_Barrier(self.comm_world)
        return

    cdef void create_sub_communicators(self):
        '''
        :return: Sets up cartesian sub topologies from cart_comm_world.
        '''
        cdef:
            int ierr = 0
            int [3] remains

        #Create the sub-communicator where x-dimension remains
        remains[0] = 1
        remains[1] = 0
        remains[2] = 0
        ierr = mpi.MPI_Cart_sub(self.cart_comm_world,remains, &self.cart_comm_sub_x)
        ierr =  mpi.MPI_Comm_size(self.cart_comm_sub_x, &self.sub_x_size)
        ierr =  mpi.MPI_Comm_rank(self.cart_comm_sub_x, &self.sub_x_rank)

        #Create the sub-communicator where the y-dimension remains
        remains[0] = 0
        remains[1] = 1
        remains[2] = 0
        ierr = mpi.MPI_Cart_sub(self.cart_comm_world,remains, &self.cart_comm_sub_y)
        ierr =  mpi.MPI_Comm_size(self.cart_comm_sub_y, &self.sub_y_size)
        ierr =  mpi.MPI_Comm_rank(self.cart_comm_sub_y, &self.sub_y_rank)

        #Create the sub communicator where the z-dimension remains
        remains[0] = 0
        remains[1] = 0
        remains[2] = 1
        ierr = mpi.MPI_Cart_sub(self.cart_comm_world,remains, &self.cart_comm_sub_z)
        ierr =  mpi.MPI_Comm_size(self.cart_comm_sub_z, &self.sub_z_size)
        ierr =  mpi.MPI_Comm_rank(self.cart_comm_sub_z, &self.sub_z_rank)

        #Create the sub communicator where x and y-dimension still remains
        remains[0] = 1
        remains[1] = 1
        remains[2] = 0
        ierr = mpi.MPI_Cart_sub(self.cart_comm_world,remains, &self.cart_comm_sub_xy)


        return

    cdef float domain_scalar_sum(self, float local_value):
        '''
        Compute the sum over all mpi ranks of a single scalar of type double.
        :param local_value: the value to be summed over the ranks
        :return: sum of local values on all processes
        '''

        cdef:
            float global_sum

        mpi.MPI_Allreduce(&local_value, &global_sum,1,mpi.MPI_FLOAT,mpi.MPI_SUM,self.comm_world)

        return global_sum



    cdef float domain_scalar_max(self, float local_value):
        '''
        Compute the maximum over all mpi ranks of a single scalar of type double.
        :param local_value: the value to be maxed over the ranks
        :return: maximum of local values on all processes
        '''

        cdef:
            float global_max

        mpi.MPI_Allreduce(&local_value, &global_max,1,mpi.MPI_FLOAT,mpi.MPI_MAX,self.comm_world)

        return global_max

    cdef float domain_scalar_min(self, float local_value):
        '''
        Compute the minimum over all mpi ranks of a single scalar of type double.
        :param local_value: the value to be min-ed over the ranks
        :return: sum of local values on all processes
        '''

        cdef:
            float  global_min

        mpi.MPI_Allreduce(&local_value, &global_min,1,mpi.MPI_FLOAT,mpi.MPI_MIN,self.comm_world)

        return global_min

    cdef float [:] domain_vector_sum(self, float [:] local_vector, Py_ssize_t n):
        '''
        Compute the sum over all mpi ranks of a vector of type double.
        :param local_vector: the value to be summed over the ranks
        :return: sum of local vectors on all processes
        '''

        cdef:
            float [:] global_sum = np.empty((n,),dtype=np.float32,order='c')

        mpi.MPI_Allreduce(&local_vector[0], &global_sum[0],n,mpi.MPI_FLOAT,mpi.MPI_SUM,self.comm_world)

        return global_sum

    cdef float [:] HorizontalMean(self, Grid.Grid Gr, float *values):
        '''
        Compute the horizontal mean of the array pointed to by values.
        values should have dimension of Gr.dims.nlg[0] * Gr.dims.nlg[1]
        * Gr.dims.nlg[1].

        :param Gr: Grid class
        :param values1: pointer to array of type float containing first value in product
        :return: memoryview type float with dimension Gr.dims.nlg[2]
        '''

        cdef:
            float [:] mean_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] mean = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            int i,j,k,ijk
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int kmin = 0
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int kmax = Gr.dims.nlg[2]
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]
            int ishift, jshift
            float n_horizontal_i = 1.0/np.float32(Gr.dims.n[1]*Gr.dims.n[0])

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        mean_local[k] += values[ijk]


        #Here we call MPI_Allreduce on the sub_xy communicator as we only need communication among
        #processes with the the same vertical rank

        mpi.MPI_Allreduce(&mean_local[0],&mean[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.cart_comm_sub_xy)

        for i in xrange(Gr.dims.nlg[2]):
            mean[i] = mean[i]*n_horizontal_i

        return mean

    cdef float [:] HorizontalMeanofSquares(self, Grid.Grid Gr, const float *values1, const float *values2):
        '''
        Compute the horizontal mean of the product of two variables (values1 and values2). values1 and values2 are
        passed in as pointers of type double. These should have dimension of Gr.dims.nlg[0] * Gr.dims.nlg[1]
        * Gr.dims.nlg[1].

        :param Gr: Grid class
        :param values1: pointer to array of type float containing first value in product
        :param values2: pointer to array of type float containing second value in product
        :return: memoryview type float with dimension Gr.dims.nlg[2]
        '''

        cdef:
            float [:] mean_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] mean = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            int i,j,k,ijk
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int kmin = 0
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int kmax = Gr.dims.nlg[2]
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]
            int ishift, jshift
            float n_horizontal_i = 1.0/np.float32(Gr.dims.n[1]*Gr.dims.n[0])

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        mean_local[k] += values1[ijk]*values2[ijk]



        #Here we call MPI_Allreduce on the sub_xy communicator as we only need communication among
        #processes with the the same vertical rank

        mpi.MPI_Allreduce(&mean_local[0],&mean[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.cart_comm_sub_xy)

        for i in xrange(Gr.dims.nlg[2]):
            mean[i] = mean[i]*n_horizontal_i
        return mean

    cdef float [:] HorizontalMeanofCubes(self,Grid.Grid Gr,const float *values1,const float *values2, const float *values3):

        cdef:
            float [:] mean_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] mean = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            int i,j,k,ijk
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int kmin = 0
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int kmax = Gr.dims.nlg[2]
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]
            int ishift, jshift
            float n_horizontal_i = 1.0/np.float32(Gr.dims.n[1]*Gr.dims.n[0])

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        mean_local[k] += values1[ijk]*values2[ijk]*values3[ijk]
        #Here we call MPI_Allreduce on the sub_xy communicator as we only need communication among
        #processes with the the same vertical rank

        mpi.MPI_Allreduce(&mean_local[0],&mean[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.cart_comm_sub_xy)

        for i in xrange(Gr.dims.nlg[2]):
            mean[i] = mean[i]*n_horizontal_i

        return mean

    cdef float [:] HorizontalMaximum(self, Grid.Grid Gr, float *values):
        cdef:
            float [:] max_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] max = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            int i,j,k,ijk
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int kmin = 0
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int kmax = Gr.dims.nlg[2]
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]
            int ishift, jshift
            float n_horizontal_i = 1.0/np.float32(Gr.dims.n[1]*Gr.dims.n[0])

        with nogil:
            for k in xrange(kmin,kmax):
                max_local[k] = -9e12

            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        max_local[k] = fmax(max_local[k],values[ijk])

        mpi.MPI_Allreduce(&max_local[0],&max[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_MAX,self.cart_comm_sub_xy)
        return max

    cdef float [:] HorizontalMinimum(self, Grid.Grid Gr, float *values):
        cdef:
            float [:] min_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] min = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            int i,j,k,ijk
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int kmin = 0
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int kmax = Gr.dims.nlg[2]
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]
            int ishift, jshift
            float n_horizontal_i = 1.0/np.float32(Gr.dims.n[1]*Gr.dims.n[0])

        with nogil:
            for k in xrange(kmin,kmax):
                min_local[k] = 9e12

            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        min_local[k] = fmin(min_local[k],values[ijk])

        mpi.MPI_Allreduce(&min_local[0],&min[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_MIN,self.cart_comm_sub_xy)
        return min

    cdef float HorizontalMeanSurface(self,Grid.Grid Gr,float *values):
        # Some assumptions for using this function:
        #--the <values> array is defined for all processors
        #--<values> = 0 on all processors for which zrank !=0
        # this is necessary to ensure that the root processor has the correct mean

        cdef:
            float mean_local = 0.0
            float mean = 0.0
            int i,j,ij
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int gw = Gr.dims.gw
            int istride_2d = Gr.dims.nlg[1]
            int ishift, jshift
            float n_horizontal_i = 1.0/np.float32(Gr.dims.n[1]*Gr.dims.n[0])

        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride_2d
                for j in xrange(jmin,jmax):
                    ij = ishift + j
                    mean_local += values[ij]



        mpi.MPI_Allreduce(&mean_local,&mean,1,
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.comm_world)

        mean = mean*n_horizontal_i


        return mean




    cdef float [:] HorizontalMeanConditional(self,Grid.Grid Gr,float *values, float *mask):
        '''
        This function computes horizontal means given a binary conditional. For example, it can be used to compute
        mean profiles within cloudy air. The mask must be pre-computed.
        :param Gr: Grid class
        :param values: variable array to be averaged. Contains ghost points
        :param mask: array of 1's (condition=true) and 0's (condition = false). Contains ghost points, but mask values
                        of ghost points do not have to be correct (they are not used in this routine).
        :return: vertical profile of the conditional average of array values
        '''

        cdef:
            float [:] mean_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] mean = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] sum_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] sum = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')

            int i,j,k,ijk
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int kmin = 0
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int kmax = Gr.dims.nlg[2]
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]
            int ishift, jshift


        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        mean_local[k] += values[ijk]*mask[ijk]
                        sum_local[k] += mask[ijk]


        #Here we call MPI_Allreduce on the sub_xy communicator as we only need communication among
        #processes with the the same vertical rank

        mpi.MPI_Allreduce(&mean_local[0],&mean[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.cart_comm_sub_xy)

        mpi.MPI_Allreduce(&sum_local[0],&sum[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.cart_comm_sub_xy)

        for i in xrange(Gr.dims.nlg[2]):
            mean[i] = mean[i]/np.maximum(sum[i], 1.0)

        return mean


    cdef float [:] HorizontalMeanofSquaresConditional(self,Grid.Grid Gr,float *values1,float *values2, float *mask):
        '''
        This function computes horizontal means of the product of two array given a binary conditional.
        For example, it can be used to compute mean-square profiles within cloudy air. The mask must be pre-computed.
        :param Gr: Grid class
        :param values1: 1st of the variable arrays to be multiplied. Contains ghost points
        :param values2: 2nd of the variable arrays to be multiplied. Contains ghost points
        :param mask: array of 1's (condition=true) and 0's (condition = false). Contains ghost points, but mask values
                        of ghost points do not have to be correct (they are not used in this routine).
        :return: vertical profile of the conditional average of array values
        '''



        cdef:
            float [:] mean_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] mean = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] sum_local = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')
            float [:] sum = np.zeros(Gr.dims.nlg[2],dtype=np.float32,order='c')

            int i,j,k,ijk
            int imin = Gr.dims.gw
            int jmin = Gr.dims.gw
            int kmin = 0
            int imax = Gr.dims.nlg[0] - Gr.dims.gw
            int jmax = Gr.dims.nlg[1] - Gr.dims.gw
            int kmax = Gr.dims.nlg[2]
            int istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            int jstride = Gr.dims.nlg[2]
            int ishift, jshift


        with nogil:
            for i in xrange(imin,imax):
                ishift = i * istride
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        mean_local[k] += values1[ijk]*values2[ijk]*mask[ijk]
                        sum_local[k] += mask[ijk]


        #Here we call MPI_Allreduce on the sub_xy communicator as we only need communication among
        #processes with the the same vertical rank

        mpi.MPI_Allreduce(&mean_local[0],&mean[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.cart_comm_sub_xy)

        mpi.MPI_Allreduce(&sum_local[0],&sum[0],Gr.dims.nlg[2],
                          mpi.MPI_FLOAT,mpi.MPI_SUM,self.cart_comm_sub_xy)

        for i in xrange(Gr.dims.nlg[2]):
            mean[i] = mean[i]/np.maximum(sum[i], 1.0)

        return mean

cdef class Pencil:


    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ParallelMPI Pa, int dim):

        self.dim = dim
        self.n_local_values = Gr.dims.npl

        cdef:
            int remainder = 0
            int i

        if dim==0:
            self.size = Pa.sub_x_size
            self.rank = Pa.sub_x_rank
            self.n_total_pencils = Gr.dims.nl[1] * Gr.dims.nl[2]
            self.pencil_length = Gr.dims.n[0]
        elif dim==1:
            self.size = Pa.sub_y_size
            self.rank = Pa.sub_y_rank
            self.n_total_pencils = Gr.dims.nl[0] * Gr.dims.nl[2]
            self.pencil_length = Gr.dims.n[1]
        elif dim==2:
            self.size = Pa.sub_z_size
            self.rank = Pa.sub_z_rank
            self.n_total_pencils = Gr.dims.nl[0] * Gr.dims.nl[1]
            self.pencil_length = Gr.dims.n[2]
        else:
            Pa.root_print('Pencil dim='+ str(dim) + 'not valid')
            Pa.root_print('Killing simuulation')
            Pa.kill()

        remainder =  self.n_total_pencils%self.size
        self.n_pencil_map = np.empty((self.size,),dtype=np.int,order='c')
        self.n_pencil_map[:] = self.n_total_pencils//self.size
        for i in xrange(self.size):
            if i < remainder:
                self.n_pencil_map[i] += 1

        self.n_local_pencils = self.n_pencil_map[self.rank]               #Number of pencils locally
        self.nl_map = np.empty((self.size),dtype=np.int,order='c')        #Number of local grid points in pencild dir
        self.send_counts = np.empty((self.size),dtype=np.intc,order='c')  #Total number of points to send to each rank
        self.recv_counts = np.empty((self.size),dtype=np.intc,order='c')  #Total numer of points to recv from each rank
        self.rdispls = np.zeros((self.size),dtype=np.intc,order='c')      #Where to put received points
        self.sdispls = np.zeros((self.size),dtype=np.intc,order='c')      #Where to get sent points

        #Now need to communicate number of local points on each process
        if self.dim==0:
            #Gather the number of points on in direction dim for each rank
            mpi.MPI_Allgather(&Gr.dims.nl[0],1,mpi.MPI_LONG,&self.nl_map[0],1,mpi.MPI_LONG,Pa.cart_comm_sub_x)

            #Now compute the send counts
            for i in xrange(self.size):
                self.send_counts[i] = Gr.dims.nl[0] * self.n_pencil_map[i]
                self.recv_counts[i] = self.n_local_pencils * self.nl_map[i]

        elif self.dim==1:
            mpi.MPI_Allgather(&Gr.dims.nl[1],1,mpi.MPI_LONG,&self.nl_map[0],1,mpi.MPI_LONG,Pa.cart_comm_sub_y)
            #Now compute the send counts
            for i in xrange(self.size):
                self.send_counts[i] = Gr.dims.nl[1] * self.n_pencil_map[i]
                self.recv_counts[i] = self.n_local_pencils * self.nl_map[i]
        else:
            mpi.MPI_Allgather(&Gr.dims.nl[2],1,mpi.MPI_LONG,&self.nl_map[0],1,mpi.MPI_LONG,Pa.cart_comm_sub_z)
            #Now compute the send counts
            for i in xrange(self.size):
                self.send_counts[i] = Gr.dims.nl[2] * self.n_pencil_map[i]
                self.recv_counts[i] = self.n_local_pencils * self.nl_map[i]

        #Compute the send and receive displacments
        for i in xrange(self.size-1):
            self.sdispls[i+1] = self.sdispls[i] + self.send_counts[i]
            self.rdispls[i+1] = self.rdispls[i] + self.recv_counts[i]

        Pa.barrier()
        return

    cdef float [:,:] forward_double(self, Grid.DimStruct *dims, ParallelMPI Pa ,float *data):

        cdef:
            float [:] local_transpose = np.empty((dims.npl,),dtype=np.float32,order='c')
            float [:] recv_buffer = np.empty((self.n_local_pencils * self.pencil_length),dtype=np.float32,order='c')
            float [:,:] pencils = np.empty((self.n_local_pencils,self.pencil_length),dtype=np.float32,order='c')

        #Build send buffer
        self.build_buffer_double(dims, data, &local_transpose[0])

        if(self.size > 1):
            #Do all to all communication
            if self.dim == 0:
                mpi.MPI_Alltoallv(&local_transpose[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_FLOAT,
                            &recv_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_FLOAT,Pa.cart_comm_sub_x)
            elif self.dim==1:
                mpi.MPI_Alltoallv(&local_transpose[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_FLOAT,
                            &recv_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_FLOAT,Pa.cart_comm_sub_y)
            else:
                mpi.MPI_Alltoallv(&local_transpose[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_FLOAT,
                            &recv_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_FLOAT,Pa.cart_comm_sub_z)

            self.unpack_buffer_double(dims,&recv_buffer[0],pencils)

        else:
            self.unpack_buffer_double(dims,&local_transpose[0],pencils)


        return pencils

    cdef void build_buffer_double(self, Grid.DimStruct *dims, float *data, float *local_transpose ):
        '''
            A method to build a send buffer for Pencils of type double. The function has no return value but does
            have side effects the memory pointed to by *local_transpose.

        :param dims: pointer to dims structure
        :param data: pointer to 1D array
        :param local_transpose: pointer to the transposed data ready for Pencil communication.
        :return:
        '''

        cdef:
            long imin = dims.gw
            long jmin = dims.gw
            long kmin = dims.gw
            long imax = dims.nlg[0] - dims.gw
            long jmax = dims.nlg[1] - dims.gw
            long kmax = dims.nlg[2] - dims.gw
            long istride, jstride, kstride
            long istride_nogw, jstride_nogw, kstride_nogw
            long ishift, jshift, kshift
            long ishift_nogw, jshift_nogw, kshift_nogw
            long i,j,k,ijk,ijk_no_gw

        '''
           Determine the strides, first for the un-transposed data (including ghost points), and then for the transposed
                data. In the case of the transposed data, the strides are such that the fastest changing 3D index is in
                then self.dim direction.
        '''
        if self.dim == 0:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = 1
            jstride_nogw = dims.nl[0]
            kstride_nogw = dims.nl[0] * dims.nl[1]
        elif self.dim ==1:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1]
            jstride_nogw = 1
            kstride_nogw = dims.nl[0] * dims.nl[1]
        else:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1] * dims.nl[2]
            jstride_nogw = dims.nl[2]
            kstride_nogw = 1

        '''
            Transpose the data given the strides above. The indicies i, j, k are for the un-transposed data including
            ghost points. For the transposed data, excluding ghost points we must stubrtact gw.
        '''
        with nogil:
            for i in xrange(imin,imax):
                ishift = i*istride
                ishift_nogw = (i-dims.gw) * istride_nogw
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    jshift_nogw = (j-dims.gw) * jstride_nogw
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        ijk_no_gw = ishift_nogw + jshift_nogw+ (k-dims.gw)*kstride_nogw
                        local_transpose[ijk_no_gw] = data[ijk]
        return

    cdef void unpack_buffer_double(self,Grid.DimStruct *dims, float *recv_buffer, float  [:,:] pencils):

        cdef:
            long m, p, i
            long nl_shift, count


        #Loop over the number of processors in the rank
        count = 0
        for m in xrange(self.size):

            if m == 0:
                nl_shift = 0
            else:
                nl_shift += self.nl_map[m-1]

            #Loop over the number of local pencils
            with nogil:
                for p in xrange(self.n_local_pencils):
                    #Now loop over the number of points in each pencil from the m-th processor
                    for i in xrange(self.nl_map[m]):
                        pencils[p,nl_shift + i] = recv_buffer[count]
                        count += 1
        return

    cdef void reverse_double(self, Grid.DimStruct *dims, ParallelMPI Pa, float [:,:] pencils, float *data):

        cdef:
            float [:] send_buffer = np.empty(self.n_local_pencils * self.pencil_length,dtype=np.float32,order='c')
            float [:] recv_buffer = np.empty(dims.npl,dtype=np.float32,order='c')

        #This is exactly the inverse operation to forward_float so that the send_counts can be used as the recv_counts
        #and vice versa

        self.reverse_build_buffer_double(dims,pencils,&send_buffer[0])

        if(self.size > 1):
            #Do all to all communication
            if self.dim == 0:
                mpi.MPI_Alltoallv(&send_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_FLOAT,
                            &recv_buffer[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_FLOAT,Pa.cart_comm_sub_x)
            elif self.dim==1:
                mpi.MPI_Alltoallv(&send_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_FLOAT,
                            &recv_buffer[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_FLOAT,Pa.cart_comm_sub_y)
            else:
                mpi.MPI_Alltoallv(&send_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_FLOAT,
                            &recv_buffer[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_FLOAT,Pa.cart_comm_sub_z)

            self.reverse_unpack_buffer_double(dims,&recv_buffer[0],&data[0])

        else:
            self.reverse_unpack_buffer_double(dims,&send_buffer[0],&data[0])

        return

    cdef void reverse_build_buffer_double(self, Grid.DimStruct *dims, float [:,:] pencils, float *send_buffer):
        cdef:
            long m, p, i
            long nl_shift, count
        #Loop over the number of processors in the rank
        count = 0
        for m in xrange(self.size):

            if m == 0:
                nl_shift = 0
            else:
                nl_shift += self.nl_map[m-1]

            #Loop over the number of local pencils
            with nogil:
                for p in xrange(self.n_local_pencils):
                    #Now loop over the number of points in each pencil from the m-th processor
                    for i in xrange(self.nl_map[m]):
                        send_buffer[count] = pencils[p,nl_shift + i]
                        count += 1
        return

    cdef void reverse_unpack_buffer_double(self, Grid.DimStruct *dims, float *recv_buffer, float *data ):

        cdef:
            long imin = dims.gw
            long jmin = dims.gw
            long kmin = dims.gw
            long imax = dims.nlg[0] - dims.gw
            long jmax = dims.nlg[1] - dims.gw
            long kmax = dims.nlg[2] - dims.gw
            long istride, jstride, kstride
            long istride_nogw, jstride_nogw, kstride_nogw
            long ishift, jshift, kshift
            long ishift_nogw, jshift_nogw, kshift_nogw
            long i,j,k,ijk,ijk_no_gw

        if self.dim == 0:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = 1
            jstride_nogw = dims.nl[0]
            kstride_nogw = dims.nl[0] * dims.nl[1]
        elif self.dim ==1:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1]
            jstride_nogw = 1 #dims.nl[0]
            kstride_nogw = dims.nl[0] * dims.nl[1]
        else:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1] * dims.nl[2]
            jstride_nogw = dims.nl[2]
            kstride_nogw = 1


        #Build the local buffer
        with nogil:
            for i in xrange(imin,imax):
                ishift = i*istride
                ishift_nogw = (i-dims.gw) * istride_nogw
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    jshift_nogw = (j-dims.gw) * jstride_nogw
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        ijk_no_gw = ishift_nogw + jshift_nogw+ (k-dims.gw)*kstride_nogw
                        data[ijk] = recv_buffer[ijk_no_gw]
        return

    cdef void build_buffer_complex(self, Grid.DimStruct *dims, float complex *data, float complex *local_transpose ):

        cdef:
            long imin = dims.gw
            long jmin = dims.gw
            long kmin = dims.gw
            long imax = dims.nlg[0] - dims.gw
            long jmax = dims.nlg[1] - dims.gw
            long kmax = dims.nlg[2] - dims.gw
            long istride, jstride, kstride
            long istride_nogw, jstride_nogw, kstride_nogw
            long ishift, jshift, kshift
            long ishift_nogw, jshift_nogw, kshift_nogw
            long i,j,k,ijk,ijk_no_gw

        if self.dim == 0:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = 1
            jstride_nogw = dims.nl[0]
            kstride_nogw = dims.nl[0] * dims.nl[1]
        elif self.dim ==1:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1]
            jstride_nogw = 1 #dims.nl[0]
            kstride_nogw = dims.nl[0] * dims.nl[1]
        else:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1] * dims.nl[2]
            jstride_nogw = dims.nl[2]
            kstride_nogw = 1

        #Build the local buffer
        with nogil:
            for i in xrange(imin,imax):
                ishift = i*istride
                ishift_nogw = (i-dims.gw) * istride_nogw
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    jshift_nogw = (j-dims.gw) * jstride_nogw
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        ijk_no_gw = ishift_nogw + jshift_nogw+ (k-dims.gw)*kstride_nogw
                        local_transpose[ijk_no_gw] = data[ijk]
        return

    cdef void unpack_buffer_complex(self,Grid.DimStruct *dims, float complex *recv_buffer, float complex  [:,:] pencils):

        cdef:
            long m, p, i
            long nl_shift, count

        #Loop over the number of processors in the rank
        count = 0
        for m in xrange(self.size):

            if m == 0:
                nl_shift = 0
            else:
                nl_shift += self.nl_map[m-1]

            #Loop over the number of local pencils
            with nogil:
                for p in xrange(self.n_local_pencils):
                    #Now loop over the number of points in each pencil from the m-th processor
                    for i in xrange(self.nl_map[m]):
                        pencils[p,nl_shift + i] = recv_buffer[count]
                        count += 1
        return

    cdef float complex [:,:] forward_complex(self, Grid.DimStruct *dims, ParallelMPI Pa ,float complex *data):

        cdef:
            float complex [:] local_transpose = np.empty((dims.npl,),dtype=np.complex64,order='c')
            float complex [:] recv_buffer = np.empty((self.n_local_pencils * self.pencil_length),dtype=np.complex64,order='c')
            float complex [:,:] pencils = np.empty((self.n_local_pencils,self.pencil_length),dtype=np.complex64,order='c')

        #Build send buffer
        self.build_buffer_complex(dims, data, &local_transpose[0])

        if(self.size > 1):
            #Do all to all communication
            if self.dim == 0:
                mpi.MPI_Alltoallv(&local_transpose[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_COMPLEX,
                            &recv_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_COMPLEX,Pa.cart_comm_sub_x)
            elif self.dim==1:
                mpi.MPI_Alltoallv(&local_transpose[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_COMPLEX,
                            &recv_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_COMPLEX,Pa.cart_comm_sub_y)
            else:
                mpi.MPI_Alltoallv(&local_transpose[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_COMPLEX,
                            &recv_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_COMPLEX,Pa.cart_comm_sub_z)


            self.unpack_buffer_complex(dims,&recv_buffer[0],pencils)

        else:
            self.unpack_buffer_complex(dims,&local_transpose[0],pencils)

        return pencils

    cdef void reverse_complex(self, Grid.DimStruct *dims, ParallelMPI Pa, float complex [:,:] pencils, float complex *data):

        cdef:
            float complex [:] send_buffer = np.empty(self.n_local_pencils * self.pencil_length,dtype=np.complex64,order='c')
            float complex [:] recv_buffer = np.empty(dims.npl,dtype=np.complex64,order='c')

        #This is exactly the inverse operation to forward_float so that the send_counts can be used as the recv_counts
        #and vice versa
        self.reverse_build_buffer_complex(dims,pencils,&send_buffer[0])
        if(self.size > 1):
            #Do all to all communication
            if self.dim == 0:
                mpi.MPI_Alltoallv(&send_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_COMPLEX,
                            &recv_buffer[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_COMPLEX,Pa.cart_comm_sub_x)
            elif self.dim==1:
                mpi.MPI_Alltoallv(&send_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_COMPLEX,
                            &recv_buffer[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_COMPLEX,Pa.cart_comm_sub_y)
            else:

                mpi.MPI_Alltoallv(&send_buffer[0], &self.recv_counts[0], &self.rdispls[0],mpi.MPI_COMPLEX,
                            &recv_buffer[0], &self.send_counts[0], &self.sdispls[0],mpi.MPI_COMPLEX,Pa.cart_comm_sub_z)
            self.reverse_unpack_buffer_complex(dims,&recv_buffer[0],data)
        else:
            self.reverse_unpack_buffer_complex(dims,&send_buffer[0],data)

        return


    cdef void reverse_build_buffer_complex(self, Grid.DimStruct *dims, float complex [:,:] pencils, float complex *send_buffer):
        cdef:
            long m, p, i
            long nl_shift, count

        #Loop over the number of processors in the rank
        count = 0
        for m in xrange(self.size):

            if m == 0:
                nl_shift = 0
            else:
                nl_shift += self.nl_map[m-1]

            #Loop over the number of local pencils
            with nogil:
                for p in xrange(self.n_local_pencils):
                    #Now loop over the number of points in each pencil from the m-th processor
                    for i in xrange(self.nl_map[m]):
                        send_buffer[count] = pencils[p,nl_shift + i]
                        count += 1
        return

    cdef void reverse_unpack_buffer_complex(self, Grid.DimStruct *dims, float complex *recv_buffer, float complex *data ):

        cdef:
            long imin = dims.gw
            long jmin = dims.gw
            long kmin = dims.gw
            long imax = dims.nlg[0] - dims.gw
            long jmax = dims.nlg[1] - dims.gw
            long kmax = dims.nlg[2] - dims.gw
            long istride, jstride, kstride
            long istride_nogw, jstride_nogw, kstride_nogw
            long ishift, jshift, kshift
            long ishift_nogw, jshift_nogw, kshift_nogw

            long i,j,k,ijk,ijk_no_gw

        if self.dim == 0:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = 1
            jstride_nogw = dims.nl[0]
            kstride_nogw = dims.nl[0] * dims.nl[1]
        elif self.dim ==1:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1]
            jstride_nogw = 1 #dims.nl[0]
            kstride_nogw = dims.nl[0] * dims.nl[1]
        else:
            istride = dims.nlg[1] * dims.nlg[2]
            jstride = dims.nlg[2]
            kstride = 1

            istride_nogw = dims.nl[1] * dims.nl[2]
            jstride_nogw = dims.nl[2]
            kstride_nogw = 1

        #Build the local buffer
        with nogil:
            for i in xrange(imin,imax):
                ishift = i*istride
                ishift_nogw = (i-dims.gw) * istride_nogw
                for j in xrange(jmin,jmax):
                    jshift = j * jstride
                    jshift_nogw = (j-dims.gw) * jstride_nogw
                    for k in xrange(kmin,kmax):
                        ijk = ishift + jshift + k
                        ijk_no_gw = ishift_nogw + jshift_nogw+ (k-dims.gw)*kstride_nogw
                        data[ijk] = recv_buffer[ijk_no_gw]
        return

