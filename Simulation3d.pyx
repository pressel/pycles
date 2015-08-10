cimport ParallelMPI
cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ScalarAdvection
cimport MomentumAdvection
cimport SGS
cimport ScalarDiffusion
cimport MomentumDiffusion
cimport ReferenceState
cimport PressureSolver
cimport TimeStepping
cimport Kinematics
cimport NetCDFIO

print 'Here'
from Initialization import InitializationFactory
from Microphysics import MicrophysicsFactory
from Thermodynamics cimport LatentHeat
from Thermodynamics import ThermodynamicsFactory
import time

cdef inline double L_const(double T) nogil:
    return T + T*2.0 / T

class Simulation3d:
    def __init__(self,namelist):
        return


    def initialize(self,namelist):
        self.Parallel = ParallelMPI.ParallelMPI(namelist)
        self.Grid = Grid.Grid(namelist,self.Parallel)
        self.PV = PrognosticVariables.PrognosticVariables(self.Grid)
        self.Ke = Kinematics.Kinematics()

        self.DV = DiagnosticVariables.DiagnosticVariables()
        self.Pr = PressureSolver.PressureSolver()

        self.SA = ScalarAdvection.ScalarAdvection(namelist, self.Parallel)
        self.MA = MomentumAdvection.MomentumAdvection(namelist, self.Parallel)
        self.SGS = SGS.SGS()
        self.SD = ScalarDiffusion.ScalarDiffusion(self.DV,self.Parallel)
        self.MD = MomentumDiffusion.MomentumDiffusion(self.DV,self.Parallel)

        self.LH = LatentHeat(namelist, self.Parallel)
        self.Micro = MicrophysicsFactory(namelist,self.LH,self.Parallel)
        self.Thermo = ThermodynamicsFactory(namelist,self.Micro,self.LH,self.Parallel)

        self.Reference = ReferenceState.ReferenceState(self.Grid)
        self.StatsIO  = NetCDFIO.NetCDFIO_Stats()
        self.FieldsIO = NetCDFIO.NetCDFIO_Fields()



        self.TS = TimeStepping.TimeStepping()

        #Add new prognostic variables
        self.PV.add_variable('u','m/s',"sym","velocity",self.Parallel)
        self.PV.set_velocity_direction('u',0,self.Parallel)
        self.PV.add_variable('v','m/s',"sym","velocity",self.Parallel)
        self.PV.set_velocity_direction('v',1,self.Parallel)
        self.PV.add_variable('w','m/s',"asym","velocity",self.Parallel)
        self.PV.set_velocity_direction('w',2,self.Parallel)

        self.StatsIO.initialize(namelist, self.Grid, self.Parallel)
        self.FieldsIO.initialize(namelist,self.Parallel)
        self.Thermo.initialize(self.Grid,self.PV,self.DV,self.Parallel)


        self.PV.initialize(self.Grid)

        self.Ke.initialize(self.Grid)
        self.SA.initialize(self.Grid,self.PV)
        self.MA.initialize(self.Grid,self.PV)
        self.SGS.initialize(self.Grid)
        self.SD.initialize(self.Grid,self.PV,self.DV,self.Parallel)
        self.MD.initialize(self.Grid,self.PV,self.DV,self.Parallel)

        self.TS.initialize(namelist,self.PV,self.Parallel)


        SetInitialConditions = InitializationFactory(namelist)
        SetInitialConditions(self.Grid,self.PV,self.Reference,self.Thermo)


        self.Pr.initialize(namelist,self.Grid,self.Reference,self.DV,self.Parallel)
        self.DV.initialize(self.Grid)

        return



    def run(self):
        cdef PrognosticVariables.PrognosticVariables PV_ = self.PV
        cdef DiagnosticVariables.DiagnosticVariables DV_ = self.DV
        PV_.Update_all_bcs(self.Grid,self.Parallel)

        cdef LatentHeat LH_ = self.LH
        cdef Grid.Grid GR_ = self.Grid
        cdef ParallelMPI.ParallelMPI PA_ = self.Parallel

        cdef int rk_step
        while (self.TS.t < self.TS.t_max):

            for self.TS.rk_step in xrange(self.TS.n_rk_steps):
                self.Ke.update(self.Grid,PV_)
                self.Thermo.update(self.Grid,self.Reference,PV_,DV_)
                self.SA.update_cython(self.Grid,self.Reference,PV_,self.Parallel)
                self.MA.update(self.Grid,self.Reference,PV_,self.Parallel)
                self.SGS.update(self.Grid,self.Reference,self.DV,self.PV)
                #self.SD.update(self.Grid,self.Reference,self.PV,self.DV)
                #self.MD.update(self.Grid,self.Reference,self.PV,self.DV,self.Ke)
                time1 = time.time()
                self.FieldsIO.update(self.Grid, self.PV, self.TS,self.Parallel)
                self.TS.update(self.Grid, self.PV, self.Parallel)
                time2 = time.time()
                PV_.Update_all_bcs(self.Grid,self.Parallel)
                self.Pr.update(self.Grid,self.Reference,self.DV,self.PV,self.Parallel)


            self.Parallel.root_print('T = ' + str(self.TS.t) + ' dt = ' + str(self.TS.dt) + ' cfl_max = ' + str(self.TS.cfl_max) + ' walltime = ' + str(time2 - time1) )

                #var_u = PV_.get_tendency_array('s',self.Grid)
                #var = DV_.get_variable_array('dynamic_pressure',self.Grid)
                #var_u = PV_.get_variable_array('u',self.Grid)
                #print 'symmetry', var[128+1,6,24],var[128+2,6,24],var[128+3,6,24],var[128+4,6,24]
                #print 'symmetry u end',var_u[128+1:128+4,6,24]

                #print(np.array(self.Reference.alpha0[:]),np.array(self.Reference.alpha0_half[:]))
            #self.Parallel.kill()
            #import pylab as plt
            #var = PV_.get_variable_array('s',self.Grid)
            #w = PV_.get_variable_array('w',self.Grid)
            #p = DV_.get_variable_array('dynamic_pressure',self.Grid)
            #print(np.max(w))
            #try:
            #    plt.figure(1)
            #    plt.contour(var[GR_.dims.gw:-GR_.dims.gw,7,GR_.dims.gw:-GR_.dims.gw].T,128)
            #    plt.colorbar()
            #    plt.savefig('./figs/'+str(10000 + i) + '.png')
            #    plt.close()
            #except:
            #    pass

        return


    def io(self):

        return


