#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport mpi4py.mpi_c as mpi
cimport ParallelMPI
cimport Restart
cimport numpy as np
import numpy as np
import time
cdef class Grid:
    '''
    A class for storing information about the LES grid.
    '''
    def __init__(self,namelist,Parallel):
        '''

        :param namelist: Namelist dictionary
        :param Parallel: ParallelMPI class
        :return:
        '''

        self.dims.dims = namelist['grid']['dims']

        #Get the grid spacing
        self.dims.dx[0] = namelist['grid']['dx']
        self.dims.dx[1] = namelist['grid']['dy']
        self.dims.dx[2] = namelist['grid']['dz']

        #Set the inverse grid spacing
        self.dims.dxi[0] = 1.0/self.dims.dx[0]
        self.dims.dxi[1] = 1.0/self.dims.dx[1]
        self.dims.dxi[2] = 1.0/self.dims.dx[2]

        #Get the grid dimensions and ghost points
        self.dims.gw = namelist['grid']['gw']
        self.dims.n[0] = namelist['grid']['nx']
        self.dims.n[1] = namelist['grid']['ny']
        self.dims.n[2] = namelist['grid']['nz']

        #Compute the global and local dims
        self.compute_global_dims()
        self.compute_local_dims(Parallel)
        self.compute_coordinates()

        return

    cdef inline void compute_global_dims(self):
        '''
        Compute the dimensions of the global of the domain, including ghost points and store the to self.dims.
        :return:
        '''
        cdef int i
        with nogil:
            for i in range(self.dims.dims):
                self.dims.ng[i] = self.dims.n[i] + 2*self.dims.gw
        return

    cdef inline void compute_local_dims(self,ParallelMPI.ParallelMPI Parallel):
        '''
        This function computes the local dimensions of the 3D array owned by each processor. No assumption is made
        about the number of cores evenly dividing the number of global grid points in each directions. If the number of
        grid points is not evenly divisible, we tack one additional point from the remainder onto each rank less the
        the remainder.
        :param Parallel:
        :return:
        '''
        cdef:
            int i
            int ierr = 0
            int maxdims = 3
            int [3] mpi_dims
            int [3] mpi_periods
            int [3] mpi_coords
            int remainder = 0

        ierr = mpi.MPI_Cart_get(Parallel.cart_comm_world,maxdims,mpi_dims,mpi_periods,mpi_coords)
        for i in xrange(3):  #Here we loop over all three dimensions even if they are empty
            self.dims.nl[i] = self.dims.n[i]//mpi_dims[i]
            remainder = self.dims.n[i]%mpi_dims[i]
            if remainder > 0 and mpi_coords[i] < remainder:
                self.dims.nl[i] += 1
            self.dims.nlg[i] = self.dims.nl[i] + 2 * self.dims.gw

        #Now compute the high and lo indicies for this processor
        for i in xrange(3):
            npts = 0
            nptsg = 0
            proc = 0


            while proc <= mpi_coords[i]:
                self.dims.indx_lo[i] = npts
                self.dims.indx_lo_g[i] = nptsg
                npts += self.dims.n[i]//mpi_dims[i]
                nptsg +=  self.dims.n[i]//mpi_dims[i] + 2 * self.dims.gw
                remainder = self.dims.n[i]%mpi_dims[i]
                if remainder >0 and proc  < remainder:
                    npts += 1
                    nptsg +=  1
                proc += 1

        self.dims.npd = np.max([self.dims.n[0],1])*np.max([self.dims.n[1],1])*np.max([self.dims.n[2],1])
        self.dims.npl = self.dims.nl[0] * self.dims.nl[1] * self.dims.nl[2]
        self.dims.npg = self.dims.nlg[0] * self.dims.nlg[1] * self.dims.nlg[2]


        #Compute the number of ghostpoint for mpi_buffers
        self.dims.nbuffer[0] = self.dims.gw * np.max([self.dims.nlg[1] * self.dims.nlg[2],
                                                      self.dims.nlg[1], self.dims.nlg[2] ])
        self.dims.nbuffer[1] = self.dims.gw * np.max([self.dims.nlg[0] * self.dims.nlg[2],
                                                      self.dims.nlg[0], self.dims.nlg[2] ])
        self.dims.nbuffer[2] = self.dims.gw * np.max([self.dims.nlg[0] * self.dims.nlg[1],
                                                      self.dims.nlg[0], self.dims.nlg[1] ])


        self.dims.ghosted_stride[0] = np.max([self.dims.nlg[1] * self.dims.nlg[2], self.dims.nlg[1], self.dims.nlg[2]])
        self.dims.ghosted_stride[1] = np.max([1, self.dims.nlg[0]])
        self.dims.ghosted_stride[2] = 1

        return

    cdef void compute_coordinates(self):
        '''
        Compute the dimensional (with units) of meters coordiantes. x_half, y_half and z_half are
        the grid cell center and x,y,z are at the grid cell edges.
        :return:
        '''


        self.x_half = np.empty((self.dims.n[0]+2*self.dims.gw),dtype=np.double,order='c')
        self.x = np.empty((self.dims.n[0]+2*self.dims.gw),dtype=np.double,order='c')

        self.y_half = np.empty((self.dims.n[1]+2*self.dims.gw),dtype=np.double,order='c')
        self.y = np.empty((self.dims.n[1]+2*self.dims.gw),dtype=np.double,order='c')

        self.z_half = np.empty((self.dims.n[2]+2*self.dims.gw),dtype=np.double,order='c')
        self.z = np.empty((self.dims.n[2]+2*self.dims.gw),dtype=np.double,order='c')

        cdef int i, count = 0
        for i in xrange(-self.dims.gw,self.dims.n[2]+self.dims.gw,1):
            self.z[count] = (i + 1) * self.dims.dx[2]
            self.z_half[count] = (i+0.5)*self.dims.dx[2]
            count += 1

        count = 0
        for i in xrange(-self.dims.gw,self.dims.n[0]+self.dims.gw,1):
            self.x[count] = (i + 1) * self.dims.dx[0]
            self.x_half[count] = (i+0.5)*self.dims.dx[0]
            count += 1

        count = 0
        for i in xrange(-self.dims.gw,self.dims.n[1]+self.dims.gw,1):
            self.y[count] = (i + 1) * self.dims.dx[1]
            self.y_half[count] = (i+0.5)*self.dims.dx[1]
            count += 1


        #Extract just the local components of the height coordinate
        self.zl = self.extract_local_ghosted(self.z,2)
        self.zl_half = self.extract_local_ghosted(self.zl,2)

        #Extract just the local components of the height coordinate
        self.xl = self.extract_local_ghosted(self.x,0)
        self.xl_half = self.extract_local_ghosted(self.xl,0)

        #Extract just the local components of the height coordinate
        self.yl = self.extract_local_ghosted(self.y,1)
        self.yl_half = self.extract_local_ghosted(self.yl,1)

        return

    cpdef extract_local(self,double [:] global_array, int dim):
        pass

    cpdef extract_local_ghosted(self,double [:] global_array, int dim):
        cdef int start = self.dims.indx_lo_g[dim]
        cdef int end = self.dims.indx_lo_g[dim] + self.dims.nlg[dim]
        #Force a copy with the return statement
        return np.array(global_array[start:end],dtype=np.double)

    cpdef restart(self, Restart.Restart Re):
        Re.restart_data['Gr'] = {}
        Re.restart_data['Gr']['dims'] = self.dims.dims
        Re.restart_data['Gr']['n'] = np.array([self.dims.n[0],
                                               self.dims.n[1],
                                               self.dims.n[2]])
        Re.restart_data['Gr']['ng'] =  np.array([self.dims.ng[0],
                                                 self.dims.ng[1],
                                                 self.dims.ng[2]])
        Re.restart_data['Gr']['nl'] = np.array([self.dims.nl[0],
                                                self.dims.nl[1],
                                                self.dims.nl[2]])
        Re.restart_data['Gr']['nlg'] = np.array([self.dims.nlg[0],
                                                  self.dims.nlg[1],
                                                  self.dims.nlg[2]])
        Re.restart_data['Gr']['indx_lo_g'] = np.array([self.dims.indx_lo_g[0],
                                                  self.dims.indx_lo_g[1],
                                                  self.dims.indx_lo_g[2]])
        Re.restart_data['Gr']['indx_lo'] = np.array([self.dims.indx_lo[0],
                                                  self.dims.indx_lo[1],
                                                  self.dims.indx_lo[2]])
        Re.restart_data['Gr']['npd'] = self.dims.npd
        Re.restart_data['Gr']['npl'] = self.dims.npl
        Re.restart_data['Gr']['npg'] = self.dims.npg
        Re.restart_data['Gr']['gw'] = self.dims.gw
        Re.restart_data['Gr']['nbuffer']  = np.array([self.dims.nbuffer[0],
                                                  self.dims.nbuffer[1],
                                                  self.dims.nbuffer[2]])
        Re.restart_data['Gr']['nbuffer']  = np.array([self.dims.ghosted_stride[0],
                                                  self.dims.ghosted_stride[1],
                                                  self.dims.ghosted_stride[2]])
        Re.restart_data['Gr']['dx'] = np.array([self.dims.dx[0],
                                                self.dims.dx[1],
                                                self.dims.dx[2]])
        Re.restart_data['Gr']['dxi'] = np.array([self.dims.dxi[0],
                                                self.dims.dxi[1],
                                                self.dims.dxi[2]])

        return