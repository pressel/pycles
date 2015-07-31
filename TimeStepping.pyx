cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport Grid as Grid

import numpy as np
cimport numpy as np

import cython

cdef class TimeStepping:

    def __init__(self):
        pass


    cpdef initialize(self,namelist,PrognosticVariables.PrognosticVariables PV, ParallelMPI.ParallelMPI Pa):

        #Get the time stepping potions from the name list
        try:
            self.ts_type = namelist['time_stepping']['ts_type']
        except:
            Pa.root_print('ts_type not given in namelist')
            Pa.root_print('Killing simulation now')
            Pa.kill()


        try:
            self.dt = namelist['time_stepping']['dti']
        except:
            Pa.root_print('dti  (initial time step) not given in namelist so taking defualt value dti = 1.0')
            self.dt = 1.0


        try:
            self.t = namelist['time_stepping']['t']
        except:
            Pa.root_print('t (initial time) not given in namelist so taking default value t = 0')
            self.t = 0.0



        #Now initialize the correct time stepping routine
        if self.ts_type == 2:
            self.initialize_second(PV)
        elif self.ts_type == 3:
            self.initialize_third(PV)
        else:
            Pa.root_print('Invalid ts_type: ' + str(self.ts_type))
            Pa.root_print('Killing simulation now')
            Pa.kill()


        return


    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
        if self.ts_type == 2:
            self.update_second(Gr,PV)
        if self.ts_type == 3:
            self.update_third(Gr,PV)


        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update_second(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):

        cdef:
            int i


        if self.rk_step == 0:
            for i in xrange(Gr.dims.npg*PV.nv):
                self.value_copies[0,i] = PV.values[i]
                PV.values[i] += PV.tendencies[i]*self.dt
                PV.tendencies[i] = 0.0
        else:
            for i in xrange(Gr.dims.npg*PV.nv):
                PV.values[i] = 0.5 * (self.value_copies[0,i] + PV.values[i] + PV.tendencies[i] * self.dt)
                PV.tendencies[i] = 0.0

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef update_third(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):

        cdef:
            int i


        if self.rk_step == 0:
            for i in xrange(Gr.dims.npg*PV.nv):
                self.value_copies[0,i] = PV.values[i]
                PV.values[i] += PV.tendencies[i]*self.dt
                PV.tendencies[i] = 0.0
        elif self.rk_step == 1:
            for i in xrange(Gr.dims.npg*PV.nv):
                PV.values[i] = 0.75 * self.value_copies[0,i] +  0.25*(PV.values[i] + PV.tendencies[i]*self.dt)
                PV.tendencies[i] = 0.0
        else:
            for i in xrange(Gr.dims.npg*PV.nv):
                PV.values[i] = (1.0/3.0) * self.value_copies[0,i] + (2.0/3.0)*(PV.values[i] + PV.tendencies[i]*self.dt)
                PV.tendencies[i] = 0.0
        return


    cdef void initialize_second(self,PrognosticVariables.PrognosticVariables PV):

        self.rk_step = 0
        self.n_rk_steps = 2

        #Now initialize storage
        self.value_copies = np.zeros((1,PV.values.shape[0]),dtype=np.double,order='c')
        self.tendency_copies = None

        return


    cdef void initialize_third(self,PrognosticVariables.PrognosticVariables PV):

        self.rk_step = 0
        self.n_rk_steps = 3

        #Now initialize storage
        self.value_copies = np.zeros((1,PV.values.shape[0]),dtype=np.double,order='c')
        self.tendency_copies = None

        return
