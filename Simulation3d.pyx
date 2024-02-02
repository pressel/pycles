import time
import numpy as np
import os
cimport numpy as np
from Initialization import InitializationFactory, AuxillaryVariables
from Thermodynamics import ThermodynamicsFactory
from Microphysics import MicrophysicsFactory
from Surface import SurfaceFactory
from Radiation import RadiationFactory
from SurfaceBudget import SurfaceBudgetFactory
from AuxiliaryStatistics import AuxiliaryStatistics
from ConditionalStatistics import ConditionalStatistics
from Thermodynamics cimport LatentHeat
from Tracers import TracersFactory
from PostProcessing import PostProcessing
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
cimport VisualizationOutput
cimport Forcing
cimport Radiation
cimport Restart
cimport Surface

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
        self.SA = ScalarAdvection.ScalarAdvection(namelist, self.LH, self.Pa)
        self.MA = MomentumAdvection.MomentumAdvection(namelist, self.Pa)
        self.SGS = SGS.SGS(namelist)
        self.SD = ScalarDiffusion.ScalarDiffusion(namelist, self.LH, self.DV, self.Pa)
        self.MD = MomentumDiffusion.MomentumDiffusion(self.DV, self.Pa)
        self.Th = ThermodynamicsFactory(namelist, self.Micro, self.LH, self.Pa)
        self.Ref = ReferenceState.ReferenceState(self.Gr)
        self.Sur = SurfaceFactory(namelist, self.LH, self.Pa)
        self.Fo = Forcing.Forcing(namelist, self.LH, self.Pa)
        self.Ra = RadiationFactory(namelist,self.LH, self.Pa)
        self.Budg = SurfaceBudgetFactory(namelist)
        self.StatsIO = NetCDFIO.NetCDFIO_Stats()
        self.FieldsIO = NetCDFIO.NetCDFIO_Fields()
        self.CondStatsIO = NetCDFIO.NetCDFIO_CondStats()
        self.Restart = Restart.Restart(namelist, self.Pa)
        self.VO = VisualizationOutput.VisualizationOutput(namelist, self.Pa)
        self.Damping = Damping.Damping(namelist, self.Pa)
        self.TS = TimeStepping.TimeStepping()
        self.Tr = TracersFactory(namelist)

        self.PP = PostProcessing(namelist)
        self.PP.initialize(namelist)

        # Add new prognostic variables
        self.PV.add_variable('u', 'm/s', 'u', 'u velocity component',"sym", "velocity", self.Pa)
        self.PV.set_velocity_direction('u', 0, self.Pa)
        self.PV.add_variable('v', 'm/s', 'v', 'v velocity component', "sym", "velocity", self.Pa)
        self.PV.set_velocity_direction('v', 1, self.Pa)
        self.PV.add_variable('w', 'm/s', 'w', 'w velocity component', "asym", "velocity", self.Pa)
        self.PV.set_velocity_direction('w', 2, self.Pa)

        AuxillaryVariables(namelist, self.PV, self.DV, self.Pa)


        self.StatsIO.initialize(namelist, self.Gr, self.Pa)
        self.FieldsIO.initialize(namelist, self.Pa)
        self.CondStatsIO.initialize(namelist, self.Gr, self.Pa)
        self.Aux = AuxiliaryStatistics(namelist)
        self.CondStats = ConditionalStatistics(namelist)
        self.Restart.initialize()

        self.VO.initialize()
        self.Th.initialize(self.Gr, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Micro.initialize(self.Gr, self.PV, self.DV, self.StatsIO, self.Pa)
        self.SGS.initialize(self.Gr,self.PV,self.StatsIO, self.Pa)
        self.Tr.initialize(self.Gr, self.PV, self.DV, self.StatsIO, self.Pa)
        self.PV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Ke.initialize(self.Gr, self.StatsIO, self.Pa)

        self.SA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.MA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.SD.initialize(self.Gr,self.PV,self.DV,self.StatsIO,self.Pa)
        self.MD.initialize(self.Gr,self.PV,self.DV,self.StatsIO, self.Pa)

        self.TS.initialize(namelist,self.PV,self.Pa)
        self.Sur.initialize(self.Gr, self.Ref,  self.StatsIO, self.Pa)

        if self.Restart.is_restart_run:
            self.Pa.root_print('This run is being restarted!')
            try:
                self.Restart.read(self.Pa)
            except:
                self.Pa.root_print('Could not read restart file')
                self.Pa.kill()

            self.TS.t = self.Restart.restart_data['TS']['t']
            self.TS.dt = self.Restart.restart_data['TS']['dt']
            self.Ref.init_from_restart(self.Gr, self.Restart)
            self.PV.init_from_restart(self.Gr, self.Restart, self.Pa)
            self.Sur.init_from_restart(self.Restart)
            self.StatsIO.last_output_time = self.Restart.restart_data['last_stats_output']
            self.CondStatsIO.last_output_time = self.Restart.restart_data['last_condstats_output']
            self.FieldsIO.last_output_time = self.Restart.restart_data['last_fields_output']
            self.Restart.last_restart_time = self.Restart.restart_data['last_restart_time']
            self.VO.last_vis_time = self.Restart.restart_data['last_vis_time']
            self.Restart.free_memory()
        else:
            self.Pa.root_print('This is not a restart run!')
            SetInitialConditions = InitializationFactory(namelist)
            SetInitialConditions(namelist,self.Gr, self.PV, self.Ref, self.Th, self.StatsIO, self.Pa, self.LH)
            del SetInitialConditions

        self.Pr.initialize(namelist, self.Gr, self.Ref, self.DV, self.Pa)
        self.DV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Fo.initialize(self.Gr, self.Ref, self.Th, self.StatsIO, self.Pa)
        self.Ra.initialize(self.Gr, self.StatsIO,self.Pa)
        self.Budg.initialize(self.Gr, self.StatsIO,self.Pa)
        self.Damping.initialize(self.Gr, self.Ref)
        self.Aux.initialize(namelist, self.Gr, self.PV, self.DV, self.StatsIO, self.Pa)
        self.CondStats.initialize(namelist, self.Gr, self.PV, self.DV, self.CondStatsIO, self.Pa)

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
        self.Ra.initialize_profiles(self.Gr, self.Ref, self.Th, self.DV, self.Sur, self.Pa)

        #Do IO if not a restarted run
        # if not self.Restart.is_restart_run:
        self.force_io()

        while (self.TS.t < self.TS.t_max):
            time1 = time.time()
            for self.TS.rk_step in xrange(self.TS.n_rk_steps):
                self.Ke.update(self.Gr,PV_)
                self.Th.update(self.Gr,self.Ref,PV_,DV_)
                self.Micro.update(self.Gr, self.Ref, self.Th, PV_, DV_, self.TS, self.Pa)
                self.Tr.update(self.Gr, self.Ref, PV_, DV_, self.TS, self.Pa)
                self.SA.update(self.Gr,self.Ref,PV_, DV_,  self.Pa)
                self.MA.update(self.Gr,self.Ref,PV_,self.Pa)
                self.Sur.update(self.Gr, self.Ref,self.PV, self.DV,self.Pa,self.TS)
                self.SGS.update(self.Gr,self.DV,self.PV, self.Ke, self.Sur,self.Pa)
                self.Damping.update(self.Gr, self.Ref,self.PV, self.DV, self.Pa)

                self.SD.update(self.Gr,self.Ref,self.PV,self.DV)
                self.MD.update(self.Gr,self.Ref,self.PV,self.DV,self.Ke)

                self.Fo.update(self.Gr, self.Ref, self.PV, self.DV, self.Pa)
                self.Ra.update(self.Gr, self.Ref, self.PV, self.DV, self.Sur, self.TS, self.Pa)
                self.Budg.update(self.Gr,self.Ra, self.Sur, self.TS, self.Pa)
                self.Tr.update_cleanup(self.Gr, self.Ref, PV_, DV_, self.Pa, self.TS)
                self.TS.update(self.Gr, self.PV, self.Pa)
                PV_.Update_all_bcs(self.Gr, self.Pa)
                self.Pr.update(self.Gr, self.Ref, self.DV, self.PV, self.Pa)
                self.TS.adjust_timestep(self.Gr, self.PV, self.DV,self.Pa)
                self.io()
                #PV_.debug(self.Gr,self.Ref,self.StatsIO,self.Pa)
            time2 = time.time()
            self.Pa.root_print('T = ' + str(self.TS.t) + ' dt = ' + str(self.TS.dt) +
                               ' cfl_max = ' + str(self.TS.cfl_max) + ' walltime = ' + str(time2 - time1))

        self.Restart.cleanup()

        return

    def io(self):
        cdef:
            double fields_dt = 0.0
            double stats_dt = 0.0
            double condstats_dt = 0.0
            double restart_dt = 0.0
            double vis_dt = 0.0
            double min_dt = 0.0

        if self.TS.t > 0 and self.TS.rk_step == self.TS.n_rk_steps - 1:
            # Adjust time step for output if necessary
            fields_dt = self.FieldsIO.last_output_time + self.FieldsIO.frequency - self.TS.t
            stats_dt = self.StatsIO.last_output_time + self.StatsIO.frequency - self.TS.t
            condstats_dt = self.CondStatsIO.last_output_time + self.CondStatsIO.frequency - self.TS.t
            restart_dt = self.Restart.last_restart_time + self.Restart.frequency - self.TS.t
            vis_dt = self.VO.last_vis_time + self.VO.frequency - self.TS.t


            dts = np.array([fields_dt, stats_dt, condstats_dt, restart_dt, vis_dt,
                            self.TS.dt, self.TS.dt_max, self.VO.frequency, self.Restart.frequency,
                            self.StatsIO.frequency, self.CondStatsIO.frequency, self.FieldsIO.frequency])



            self.TS.dt = np.amin(dts[dts > 0.0])
            # If time to ouptut fields do output
            if self.FieldsIO.last_output_time + self.FieldsIO.frequency == self.TS.t:
                self.Pa.root_print('Doing 3D FieldIO')
                self.Th.update(self.Gr, self.Ref, self.PV, self.DV)
                self.FieldsIO.last_output_time = self.TS.t
                self.FieldsIO.update(self.Gr, self.PV, self.DV, self.TS, self.Pa)
                self.FieldsIO.dump_prognostic_variables(self.Gr, self.PV)
                self.FieldsIO.dump_diagnostic_variables(self.Gr, self.DV, self.Pa)
                self.Pa.root_print('Finished Doing 3D FieldIO')

            # If time to ouput stats do output
            if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t:
                self.Pa.root_print('Doing StatsIO')
                self.StatsIO.last_output_time = self.TS.t
                self.StatsIO.open_files(self.Pa)
                self.StatsIO.write_simulation_time(self.TS.t, self.Pa)
                self.Micro.stats_io(self.Gr, self.Ref, self.Th, self.PV, self.DV, self.StatsIO, self.Pa) # do Micro.stats_io prior to DV.stats_io to get sedimentation velocity only in output
                self.PV.stats_io(self.Gr, self.Ref, self.StatsIO, self.Pa)

                self.DV.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.Fo.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
                self.Th.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)

                self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke,self.StatsIO,self.Pa)
                self.SA.stats_io(self.Gr, self.Ref,self.PV, self.StatsIO, self.Pa)
                self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
                self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
                self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)
                self.Ke.stats_io(self.Gr,self.Ref,self.PV,self.StatsIO,self.Pa)
                self.Tr.stats_io(self.Gr, self.PV, self.DV, self.TS, self.StatsIO, self.Pa)
                self.Ra.stats_io(self.Gr, self.Ref, self.DV, self.StatsIO, self.Pa)
                self.Budg.stats_io(self.Sur, self.StatsIO, self.Pa)
                self.Aux.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
                self.StatsIO.close_files(self.Pa)
                self.Pa.root_print('Finished Doing StatsIO')


            # If time to ouput stats do output
            if self.CondStatsIO.last_output_time + self.CondStatsIO.frequency == self.TS.t:
                self.Pa.root_print('Doing CondStatsIO')
                self.CondStatsIO.last_output_time = self.TS.t
                self.CondStatsIO.write_condstat_time(self.TS.t, self.Pa)

                self.CondStats.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.CondStatsIO, self.Pa)
                self.Pa.root_print('Finished Doing CondStatsIO')


            if self.VO.last_vis_time + self.VO.frequency == self.TS.t:
                self.VO.last_vis_time = self.TS.t
                self.VO.write(self.Gr, self.Ref, self.PV, self.DV, self.Pa)


            if self.Restart.output and self.Restart.last_restart_time + self.Restart.frequency == self.TS.t:
                self.Pa.root_print('Dumping Restart Files!')
                self.Restart.last_restart_time = self.TS.t
                self.Restart.restart_data['last_stats_output'] = self.StatsIO.last_output_time
                self.Restart.restart_data['last_fields_output'] = self.FieldsIO.last_output_time
                self.Restart.restart_data['last_condstats_output'] = self.CondStatsIO.last_output_time
                self.Restart.restart_data['last_vis_time'] = self.VO.last_vis_time
                self.Gr.restart(self.Restart)
                self.Sur.restart(self.Restart)
                self.Ref.restart(self.Gr, self.Restart)
                self.PV.restart(self.Gr, self.Restart)
                self.TS.restart(self.Restart)

                self.Restart.write(self.Pa)
                self.Pa.root_print('Finished Dumping Restart Files!')
                
        return

    def force_io(self):
        # output stats here

        self.Pa.root_print('Doing 3D FieldIO')
        self.Th.update(self.Gr, self.Ref, self.PV, self.DV)
        self.FieldsIO.update(self.Gr, self.PV, self.DV, self.TS, self.Pa)
        self.FieldsIO.dump_prognostic_variables(self.Gr, self.PV)
        self.FieldsIO.dump_diagnostic_variables(self.Gr, self.DV, self.Pa)
        self.Pa.root_print('Finished Doing 3D FieldIO')

        self.StatsIO.open_files(self.Pa)
        self.StatsIO.write_simulation_time(self.TS.t, self.Pa)
        self.PV.stats_io(self.Gr, self.Ref, self.StatsIO, self.Pa)

        self.DV.stats_io(self.Gr, self.StatsIO, self.Pa)
        self.Fo.stats_io(
            self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)

        self.Th.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Micro.stats_io(self.Gr, self.Ref, self.Th, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
        self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke ,self.StatsIO, self.Pa)
        self.SA.stats_io(self.Gr, self.Ref, self.PV, self.StatsIO, self.Pa)
        self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
        self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
        self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)
        self.Ke.stats_io(self.Gr, self.Ref, self.PV, self.StatsIO, self.Pa)
        self.Tr.stats_io(self.Gr, self.PV, self.DV, self.TS, self.StatsIO, self.Pa)
        self.Ra.stats_io(self.Gr, self.Ref, self.DV, self.StatsIO, self.Pa)
        self.Budg.stats_io(self.Sur, self.StatsIO, self.Pa)
        self.Aux.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
        self.StatsIO.close_files(self.Pa)
        return

    def postprocess(self):        
        self.PP.combine3d(self.Pa, self.Ref)
