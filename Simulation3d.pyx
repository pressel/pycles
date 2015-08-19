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
cimport Damping
cimport NetCDFIO

from libc.math cimport fmin

print 'Here'
from Initialization import InitializationFactory
from Microphysics import MicrophysicsFactory
from Thermodynamics cimport LatentHeat
from Thermodynamics import ThermodynamicsFactory
import time

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
        self.SGS = SGS.SGS(namelist)
        self.SD = ScalarDiffusion.ScalarDiffusion(self.DV,self.Parallel)
        self.MD = MomentumDiffusion.MomentumDiffusion(self.DV,self.Parallel)

        self.LH = LatentHeat(namelist, self.Parallel)
        self.Micro = MicrophysicsFactory(namelist,self.LH,self.Parallel)
        self.Thermo = ThermodynamicsFactory(namelist,self.Micro,self.LH,self.Parallel)

        self.Reference = ReferenceState.ReferenceState(self.Grid)
        self.StatsIO  = NetCDFIO.NetCDFIO_Stats()
        self.FieldsIO = NetCDFIO.NetCDFIO_Fields()
        self.Damping = Damping.Damping(namelist,self.Parallel)


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
        self.Thermo.initialize(self.Grid,self.PV,self.DV,self.StatsIO,self.Parallel)


        self.PV.initialize(self.Grid,self.StatsIO,self.Parallel)

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
        self.DV.initialize(self.Grid,self.StatsIO,self.Parallel)
        self.Damping.initialize(self.Grid)

        return



    def run(self):
        cdef PrognosticVariables.PrognosticVariables PV_ = self.PV
        cdef DiagnosticVariables.DiagnosticVariables DV_ = self.DV
        PV_.Update_all_bcs(self.Grid,self.Parallel)

        cdef LatentHeat LH_ = self.LH
        cdef Grid.Grid GR_ = self.Grid
        cdef ParallelMPI.ParallelMPI PA_ = self.Parallel

        cdef int rk_step

        #DO First Output
        self.Thermo.update(self.Grid,self.Reference,PV_,DV_)
        self.force_io()
        while (self.TS.t < self.TS.t_max):
            time1 = time.time()
            for self.TS.rk_step in xrange(self.TS.n_rk_steps):
                self.Ke.update(self.Grid,PV_)
                self.Thermo.update(self.Grid,self.Reference,PV_,DV_)
                self.SA.update_cython(self.Grid,self.Reference,PV_,self.Parallel)
                self.MA.update(self.Grid,self.Reference,PV_,self.Parallel)

                self.SGS.update(self.Grid,self.Reference,self.DV,self.PV, self.Ke)
                self.Damping.update(self.Grid,self.PV,self.Parallel)
                self.SD.update(self.Grid,self.Reference,self.PV,self.DV)
                self.MD.update(self.Grid,self.Reference,self.PV,self.DV,self.Ke)


                self.TS.update(self.Grid, self.PV, self.Parallel)
                PV_.Update_all_bcs(self.Grid,self.Parallel)
                self.Pr.update(self.Grid,self.Reference,self.DV,self.PV,self.Parallel)
                self.TS.adjust_timestep(self.Grid, self.PV, self.Parallel)
                self.io()

            time2 = time.time()

            self.Parallel.root_print('T = ' + str(self.TS.t) + ' dt = ' + str(self.TS.dt) + ' cfl_max = ' + str(self.TS.cfl_max) + ' walltime = ' + str(time2 - time1) )


        return


    def io(self):

        cdef:
            fields_dt = 0.0
            stats_dt = 0.0
            min_dt = 0.0

        if self.TS.t > 0 and self.TS.rk_step == self.TS.n_rk_steps - 1:

            #Adjust time step for output if necessary
            fields_dt = self.FieldsIO.last_output_time + self.FieldsIO.frequency  - self.TS.t
            stats_dt = self.StatsIO.last_output_time + self.StatsIO.frequency - self.TS.t
            if not fields_dt == 0.0 and not stats_dt == 0.0:
                min_dt = fmin(self.TS.dt,fmin(fields_dt,stats_dt))
            elif fields_dt == 0.0 and stats_dt == 0.0:
                min_dt = self.TS.dt
            elif fields_dt == 0.0:
                min_dt = fmin(self.TS.dt,stats_dt)
            else:
                min_dt = fmin(self.TS.dt,fields_dt)


            self.TS.dt = min_dt

            #If time to ouptut fields do output
            if self.FieldsIO.last_output_time + self.FieldsIO.frequency == self.TS.t:
                print 'Doing Ouput'
                self.FieldsIO.last_output_time = self.TS.t
                self.FieldsIO.update(self.Grid, self.PV, self.DV, self.TS, self.Parallel)
                self.FieldsIO.dump_prognostic_variables(self.Grid,self.PV)
                self.FieldsIO.dump_diagnostic_variables(self.Grid,self.DV)

            #If time to ouput stats do output
            if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t:
                self.StatsIO.last_output_time = self.TS.t
                self.StatsIO.write_simulation_time(self.TS.t, self.Parallel)
                self.PV.stats_io(self.Grid,self.StatsIO,self.Parallel)
                self.DV.stats_io(self.Grid,self.StatsIO,self.Parallel)
                self.Thermo.stats_io(self.Grid,self.PV,self.StatsIO,self.Parallel)

        return
    def force_io(self):

        #output stats here
        self.StatsIO.write_simulation_time(self.TS.t, self.Parallel)
        self.PV.stats_io(self.Grid,self.StatsIO,self.Parallel)
        self.DV.stats_io(self.Grid,self.StatsIO,self.Parallel)
        self.Thermo.stats_io(self.Grid,self.PV,self.StatsIO,self.Parallel)

        return

