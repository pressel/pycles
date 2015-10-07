import time
from Initialization import InitializationFactory
from Thermodynamics import ThermodynamicsFactory
from Microphysics import MicrophysicsFactory
from AuxiliaryStatistics import AuxiliaryStatisticsFactory
from libc.math cimport fmin
from Thermodynamics cimport LatentHeat
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
cimport Surface
cimport Forcing
cimport Radiation


class Simulation3d:

    def __init__(self, namelist):
        return

    def initialize(self, namelist):
        self.Pa = ParallelMPI.ParallelMPI(namelist)
        self.Gr = Grid.Grid(namelist, self.Pa)
        self.PV = PrognosticVariables.PrognosticVariables(self.Gr)
        self.Ke = Kinematics.Kinematics()
        self.DV = DiagnosticVariables.DiagnosticVariables()
        self.Pr = PressureSolver.PressureSolver()
        self.LH = LatentHeat(namelist, self.Pa)
        self.Micro = MicrophysicsFactory(namelist, self.LH, self.Pa)
        self.SA = ScalarAdvection.ScalarAdvection(namelist, self.Pa)
        self.MA = MomentumAdvection.MomentumAdvection(namelist, self.Pa)
        self.SGS = SGS.SGS(namelist)
        self.SD = ScalarDiffusion.ScalarDiffusion(namelist, self.LH, self.DV, self.Pa)
        self.MD = MomentumDiffusion.MomentumDiffusion(self.DV, self.Pa)
        self.Th = ThermodynamicsFactory(namelist, self.Micro, self.LH, self.Pa)
        self.Ref = ReferenceState.ReferenceState(self.Gr)
        self.Sur = Surface.Surface(namelist, self.LH, self.Pa)
        self.Fo = Forcing.Forcing(namelist, self.Pa)
        self.Ra = Radiation.Radiation(namelist, self.Pa)
        self.StatsIO = NetCDFIO.NetCDFIO_Stats()
        self.FieldsIO = NetCDFIO.NetCDFIO_Fields()
        self.Damping = Damping.Damping(namelist, self.Pa)
        self.TS = TimeStepping.TimeStepping()

        # Add new prognostic variables
        self.PV.add_variable('u', 'm/s', "sym", "velocity", self.Pa)
        self.PV.set_velocity_direction('u', 0, self.Pa)
        self.PV.add_variable('v', 'm/s', "sym", "velocity", self.Pa)
        self.PV.set_velocity_direction('v', 1, self.Pa)
        self.PV.add_variable('w', 'm/s', "asym", "velocity", self.Pa)
        self.PV.set_velocity_direction('w', 2, self.Pa)

        self.StatsIO.initialize(namelist, self.Gr, self.Pa)
        self.FieldsIO.initialize(namelist, self.Pa)
        self.Aux = AuxiliaryStatisticsFactory(namelist, self.Gr, self.StatsIO, self.Pa)
        self.Th.initialize(self.Gr, self.PV, self.DV, self.StatsIO, self.Pa)
        self.SGS.initialize(self.Gr,self.PV,self.StatsIO, self.Pa)
        self.PV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Ke.initialize(self.Gr, self.StatsIO, self.Pa)

        self.SA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.MA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.SD.initialize(self.Gr,self.PV,self.DV,self.StatsIO,self.Pa)
        self.MD.initialize(self.Gr,self.PV,self.DV,self.StatsIO, self.Pa)
        self.TS.initialize(namelist,self.PV,self.Pa)

        SetInitialConditions = InitializationFactory(namelist)
        SetInitialConditions(self.Gr, self.PV, self.Ref, self.Th, self.StatsIO, self.Pa)
        self.Sur.initialize(self.Gr, self.Ref, self.DV, self.StatsIO, self.Pa)

        self.Fo.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Ra.initialize(self.Gr,self.StatsIO,self.Pa)
        self.Pr.initialize(namelist, self.Gr, self.Ref, self.DV, self.Pa)
        self.DV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Damping.initialize(self.Gr)
        return

    def run(self):
        cdef PrognosticVariables.PrognosticVariables PV_ = self.PV
        cdef DiagnosticVariables.DiagnosticVariables DV_ = self.DV
        PV_.Update_all_bcs(self.Gr, self.Pa)
        cdef LatentHeat LH_ = self.LH
        cdef Grid.Grid GR_ = self.Gr
        cdef ParallelMPI.ParallelMPI PA_ = self.Pa
        cdef int rk_step
        # DO First Output
        self.Th.update(self.Gr, self.Ref, PV_, DV_)
        self.force_io()
        while (self.TS.t < self.TS.t_max):
            time1 = time.time()
            for self.TS.rk_step in xrange(self.TS.n_rk_steps):
                self.Ke.update(self.Gr,PV_)
                self.Th.update(self.Gr,self.Ref,PV_,DV_)
                self.SA.update(self.Gr,self.Ref,PV_,self.Pa)
                self.MA.update(self.Gr,self.Ref,PV_,self.Pa)
                self.Sur.update(self.Gr,self.Ref,self.PV, self.DV,self.Pa,self.TS)
                self.SGS.update(self.Gr,self.DV,self.PV, self.Ke,self.Pa)
                self.Damping.update(self.Gr,self.PV,self.Pa)
                self.SD.update(self.Gr,self.Ref,self.PV,self.DV)
                self.MD.update(self.Gr,self.Ref,self.PV,self.DV,self.Ke)

                self.Fo.update(self.Gr, self.Ref, self.PV, self.DV, self.Pa)
                self.Ra.update(self.Gr, self.Ref, self.PV, self.DV, self.Pa)
                self.TS.update(self.Gr, self.PV, self.Pa)
                PV_.Update_all_bcs(self.Gr, self.Pa)
                self.Pr.update(self.Gr, self.Ref, self.DV, self.PV, self.Pa)
                self.TS.adjust_timestep(self.Gr, self.PV, self.Pa)
                self.io()
                #PV_.debug(self.Gr,self.Ref,self.StatsIO,self.Pa)
            time2 = time.time()
            self.Pa.root_print('T = ' + str(self.TS.t) + ' dt = ' + str(self.TS.dt) +
                               ' cfl_max = ' + str(self.TS.cfl_max) + ' walltime = ' + str(time2 - time1))
        return

    def io(self):
        cdef:
            fields_dt = 0.0
            stats_dt = 0.0
            min_dt = 0.0

        if self.TS.t > 0 and self.TS.rk_step == self.TS.n_rk_steps - 1:
            # Adjust time step for output if necessary
            fields_dt = self.FieldsIO.last_output_time + \
                self.FieldsIO.frequency - self.TS.t
            stats_dt = self.StatsIO.last_output_time + \
                self.StatsIO.frequency - self.TS.t
            if not fields_dt == 0.0 and not stats_dt == 0.0:
                min_dt = fmin(self.TS.dt, fmin(fields_dt, stats_dt))
            elif fields_dt == 0.0 and stats_dt == 0.0:
                min_dt = self.TS.dt
            elif fields_dt == 0.0:
                min_dt = fmin(self.TS.dt, stats_dt)
            else:
                min_dt = fmin(self.TS.dt, fields_dt)
            self.TS.dt = min_dt

            # If time to ouptut fields do output
            if self.FieldsIO.last_output_time + self.FieldsIO.frequency == self.TS.t:
                self.Pa.root_print('Doing 3D FiledIO')
                self.FieldsIO.last_output_time = self.TS.t
                self.FieldsIO.update(
                    self.Gr, self.PV, self.DV, self.TS, self.Pa)
                self.FieldsIO.dump_prognostic_variables(self.Gr, self.PV)
                self.FieldsIO.dump_diagnostic_variables(self.Gr, self.DV, self.Pa)
                self.Pa.root_print('Finished Doing 3D FieldIO')

            # If time to ouput stats do output
            if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t:
                self.Pa.root_print('Doing StatsIO')
                self.StatsIO.last_output_time = self.TS.t
                self.StatsIO.write_simulation_time(self.TS.t, self.Pa)
                self.PV.stats_io(self.Gr, self.Ref, self.StatsIO, self.Pa)
                self.DV.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.Fo.stats_io(
                    self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
                self.Th.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
                self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke,self.StatsIO,self.Pa)
                self.SA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
                self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
                self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
                self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)
                self.Ke.stats_io(self.Gr,self.Ref,self.PV,self.StatsIO,self.Pa)
                self.Aux.stats_io(self.Gr, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)

                self.Pa.root_print('Finished Doing StatsIO')
        return

    def force_io(self):
        # output stats here
        self.StatsIO.write_simulation_time(self.TS.t, self.Pa)
        self.PV.stats_io(self.Gr, self.Ref, self.StatsIO, self.Pa)
        self.DV.stats_io(self.Gr, self.StatsIO, self.Pa)
        self.Fo.stats_io(
            self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Th.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
        self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke,self.StatsIO,self.Pa)
        self.SA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
        self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
        self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
        self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)
        self.Ke.stats_io(self.Gr,self.Ref,self.PV,self.StatsIO,self.Pa)
        self.Aux.stats_io(self.Gr, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
        return

