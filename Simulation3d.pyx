import time
import numpy as np
cimport numpy as np
from Initialization import InitializationFactory, AuxillaryVariables
from Thermodynamics import ThermodynamicsFactory
from Microphysics import MicrophysicsFactory
from AuxiliaryStatistics import AuxiliaryStatistics
from ConditionalStatistics import ConditionalStatistics
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
cimport VisualizationOutput
cimport Surface
cimport Forcing
cimport Radiation
cimport Restart

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
        self.Sur = Surface.Surface(namelist, self.LH, self.Pa)
        self.Fo = Forcing.Forcing(namelist, self.Pa)
        self.Ra = Radiation.Radiation(namelist, self.Pa)
        self.StatsIO = NetCDFIO.NetCDFIO_Stats()
        self.FieldsIO = NetCDFIO.NetCDFIO_Fields()
        self.CondStatsIO = NetCDFIO.NetCDFIO_CondStats()
        self.Restart = Restart.Restart(namelist, self.Pa)
        self.VO = VisualizationOutput.VisualizationOutput(namelist, self.Pa)
        self.Damping = Damping.Damping(namelist, self.Pa)
        self.TS = TimeStepping.TimeStepping()

        # Add new prognostic variables
        self.PV.add_variable('u', 'm/s', "sym", "velocity", self.Pa)
        self.PV.set_velocity_direction('u', 0, self.Pa)
        self.PV.add_variable('v', 'm/s', "sym", "velocity", self.Pa)
        self.PV.set_velocity_direction('v', 1, self.Pa)
        self.PV.add_variable('w', 'm/s', "asym", "velocity", self.Pa)
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
        self.PV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Ke.initialize(self.Gr, self.StatsIO, self.Pa)

        self.SA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.MA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.SD.initialize(self.Gr,self.PV,self.DV,self.StatsIO,self.Pa)
        self.MD.initialize(self.Gr,self.PV,self.DV,self.StatsIO, self.Pa)

        self.TS.initialize(namelist,self.PV,self.Pa)

        # # __
        # print('Sim.initialise: before calling Initialization.pyx')
        # cdef PrognosticVariables.PrognosticVariables PV_ = self.PV
        # cdef:
        #     Py_ssize_t u_varshift = PV_.get_varshift(self.Gr,'u')
        #     Py_ssize_t v_varshift = PV_.get_varshift(self.Gr,'v')
        #     Py_ssize_t w_varshift = PV_.get_varshift(self.Gr,'w')
        #     Py_ssize_t s_varshift = PV_.get_varshift(self.Gr,'s')
        #     Py_ssize_t qt_varshift = PV_.get_varshift(self.Gr,'qt')
        # print(u_varshift, v_varshift, w_varshift, s_varshift, qt_varshift)
        # print(u_varshift - v_varshift, v_varshift - w_varshift, w_varshift - s_varshift, qt_varshift)
        # if np.isnan(PV_.values[u_varshift:v_varshift]).any():
        #     print('u: nan')
        # else:
        #     print('u: No nan')
        # if np.isnan(PV_.values[v_varshift:w_varshift]).any():
        #     print('v nan')
        # else:
        #     print('v: No nan')
        # if np.isnan(PV_.values[w_varshift:s_varshift]).any():
        #     print('w: nan')
        # else:
        #     print('w: No nan')
        # if np.isnan(PV_.values[s_varshift:qt_varshift]).any():
        #     print('s: nan')
        # else:
        #     print('s: No nan')
        # if np.isnan(PV_.values[w_varshift:s_varshift]).any():
        #     print('qt: nan')
        # else:
        #     print('qt: No nan')
        # # __

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
            self.PV.init_from_restart(self.Gr, self.Restart)
            self.StatsIO.last_output_time = self.Restart.restart_data['last_stats_output']
            self.CondStatsIO.last_output_time = self.Restart.restart_data['last_condstats_output']
            self.FieldsIO.last_output_time = self.Restart.restart_data['last_fields_output']
            self.Restart.last_restart_time = self.Restart.restart_data['last_restart_time']
            self.VO.last_vis_time = self.Restart.restart_data['last_vis_time']
            self.Restart.free_memory()
        else:
            self.Pa.root_print('This is not a restart run!')
            SetInitialConditions = InitializationFactory(namelist)
            SetInitialConditions(self.Gr, self.PV, self.Ref, self.Th, self.StatsIO, self.Pa)
            del SetInitialConditions

        # # __
        # print('Sim.initialise: after calling Initialization.pyx')
        # if np.isnan(PV_.values[u_varshift:v_varshift]).any():
        #     print('u: nan')
        # else:
        #     print('u: No nan')
        # if np.isnan(PV_.values[v_varshift:w_varshift]).any():
        #     print('v nan')
        # else:
        #     print('v: No nan')
        # if np.isnan(PV_.values[w_varshift:s_varshift]).any():
        #     print('w: nan')
        # else:
        #     print('w: No nan')
        # if np.isnan(PV_.values[s_varshift:qt_varshift]).any():
        #     print('s: nan')
        # else:
        #     print('s: No nan')
        # if np.isnan(PV_.values[w_varshift:s_varshift]).any():
        #     print('qt: nan')
        # else:
        #     print('qt: No nan')
        # # __



        self.Sur.initialize(self.Gr, self.Ref, self.DV, self.StatsIO, self.Pa)

        self.Fo.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Ra.initialize(self.Gr, self.StatsIO,self.Pa)
        self.Pr.initialize(namelist, self.Gr, self.Ref, self.DV, self.Pa)
        self.DV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Damping.initialize(self.Gr)
        self.Aux.initialize(namelist, self.Gr, self.PV, self.DV, self.StatsIO, self.Pa)
        self.CondStats.initialize(namelist, self.Gr, self.PV, self.DV, self.CondStatsIO, self.Pa)

        self.Pa.root_print('Initialization completed!')

        # # __
        # print('Sim.initialise: checking nans')
        # if np.isnan(PV_.values[u_varshift:v_varshift]).any():
        #     print('u: nan')
        # else:
        #     print('u: No nan')
        # if np.isnan(PV_.values[v_varshift:w_varshift]).any():
        #     print('v nan')
        # else:
        #     print('v: No nan')
        # if np.isnan(PV_.values[w_varshift:s_varshift]).any():
        #     print('w: nan')
        # else:
        #     print('w: No nan')
        # if np.isnan(PV_.values[s_varshift:qt_varshift]).any():
        #     print('s: nan')
        # else:
        #     print('s: No nan')
        # if np.isnan(PV_.values[w_varshift:s_varshift]).any():
        #     print('qt: nan')
        # else:
        #     print('qt: No nan')
        # self.PV.val_nan(self.Pa,'Nan checking in Simulation: initialisation')
        # # __

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

        #Do IO if not a restarted run
        if not self.Restart.is_restart_run:
            self.force_io()

        self.Pa.root_print('Run started')

        PV_.val_nan(self.Pa,'Nan checking in Simulation: time: '+str(self.TS.t))

        # # __
        cdef:
            Py_ssize_t u_varshift = PV_.get_varshift(self.Gr,'u')
            Py_ssize_t v_varshift = PV_.get_varshift(self.Gr,'v')
            Py_ssize_t w_varshift = PV_.get_varshift(self.Gr,'w')
            Py_ssize_t s_varshift = PV_.get_varshift(self.Gr,'s')
            Py_ssize_t qt_varshift = PV_.get_varshift(self.Gr,'qt')
        # # __

        while (self.TS.t < self.TS.t_max):
            time1 = time.time()
            self.Pa.root_print('time: '+str(self.TS.t))
            for self.TS.rk_step in xrange(self.TS.n_rk_steps):
                # # __
                if np.isnan(PV_.values).any():
                    if np.isnan(PV_.values[u_varshift:v_varshift]).any():
                        print('u: is nan')
                    else:
                        print('u (Sim): ', np.amax(PV_.values[u_varshift:v_varshift]))

                    if np.isnan(PV_.values[v_varshift:w_varshift]).any():
                        print('v: is nan')
                    else:
                        print('v (Sim): ', np.amax(PV_.values[v_varshift:w_varshift]))
                    if np.isnan(PV_.values[w_varshift:s_varshift]).any():
                        print('w: is nan')
                    else:
                        print('w (Sim): ', np.amax(PV_.values[w_varshift:s_varshift]))
                    if np.isnan(PV_.values[s_varshift:-1]).any():
                        print('s: is nan')
                    else:
                        print('s (Sim): ', np.amax(PV_.values[s_varshift:-1]))
                # # __
                self.Ke.update(self.Gr,PV_)
                self.Th.update(self.Gr,self.Ref,PV_,DV_)
                self.Micro.update(self.Gr, self.Ref, PV_, DV_, self.TS, self.Pa )
                self.SA.update(self.Gr,self.Ref,PV_, DV_,  self.Pa)
                self.MA.update(self.Gr,self.Ref,PV_,self.Pa)
                # # __
                if np.isnan(PV_.tendencies).any():
                    print('!!!! PV Tendencies nan')
                # else:
                #     print('No nan in PV tendencies')
                # # __
                self.Sur.update(self.Gr,self.Ref,self.PV, self.DV,self.Pa,self.TS)
                self.SGS.update(self.Gr,self.DV,self.PV, self.Ke,self.Pa)
                self.Damping.update(self.Gr,self.PV,self.Pa)
                self.SD.update(self.Gr,self.Ref,self.PV,self.DV)
                self.MD.update(self.Gr,self.Ref,self.PV,self.DV,self.Ke)

                self.Fo.update(self.Gr, self.Ref, self.PV, self.DV, self.Pa)
                self.Ra.update(self.Gr, self.Ref, self.PV, self.DV, self.Pa)
                if np.isnan(PV_.tendencies).any():
                    print('!!!! PV Tendencies nan (1)')
                # else:
                #     print('No nan in PV tendencies (1)')
                if np.isnan(PV_.values).any():
                    print('!!!! PV Values nan (1)')
                # else:
                #     print('No nan in PV values (1)')

                self.TS.update(self.Gr, self.PV, self.Pa)

                if np.isnan(PV_.values).any():
                    print('!!!! PV Values nan (2), rk step: ', self.TS.rk_step, self.TS.n_rk_steps)
                # else:
                #     print('No nan in PV values (2)')

                PV_.Update_all_bcs(self.Gr, self.Pa)

                if np.isnan(PV_.values).any():
                    print('!!!! PV Values nan (3), rk step: ', self.TS.rk_step)
                # else:
                #     print('No nan in PV values (3), rk step: ', self.TS.rk_step)

                self.Pr.update(self.Gr, self.Ref, self.DV, self.PV, self.Pa)
                # self.Pa.root_print('ok until here')

                if np.isnan(PV_.values).any():
                    print('!!!! PV Values nan (4), rk step: ', self.TS.rk_step)
                # else:
                #     print('No nan in PV values (4)')

                self.TS.adjust_timestep(self.Gr, self.PV, self.DV,self.Pa)
                self.io()
                #PV_.debug(self.Gr,self.Ref,self.StatsIO,self.Pa)
                # self.Pa.root_print('rk_step: '+str(self.TS.rk_step)+' (total steps: '+str(self.TS.n_rk_steps)+')')
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

                self.Micro.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa) # do Micro.stats_io prior to DV.stats_io to get sedimentation velocity only in output
                self.PV.stats_io(self.Gr, self.Ref, self.StatsIO, self.Pa)

                self.DV.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.Fo.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
                self.Th.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)

                self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke,self.StatsIO,self.Pa)
                self.SA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
                self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
                self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
                self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)
                self.Ke.stats_io(self.Gr,self.Ref,self.PV,self.StatsIO,self.Pa)
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


            if self.Restart.last_restart_time + self.Restart.frequency == self.TS.t:
                self.Pa.root_print('Dumping Restart Files!')
                self.Restart.last_restart_time = self.TS.t
                self.Restart.restart_data['last_stats_output'] = self.StatsIO.last_output_time
                self.Restart.restart_data['last_fields_output'] = self.FieldsIO.last_output_time
                self.Restart.restart_data['last_condstats_output'] = self.CondStatsIO.last_output_time
                self.Restart.restart_data['last_vis_time'] = self.VO.last_vis_time
                self.Gr.restart(self.Restart)
                self.Ref.restart(self.Gr, self.Restart)
                self.PV.restart(self.Gr, self.Restart)
                self.TS.restart(self.Restart)

                self.Restart.write(self.Pa)
                self.Pa.root_print('Finished Dumping Restart Files!')






        return

    def force_io(self):
        # output stats here

        self.Pa.root_print('Doing 3D FiledIO')
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
        self.Micro.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
        self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke ,self.StatsIO, self.Pa)
        self.SA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
        self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
        self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
        self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)
        self.Ke.stats_io(self.Gr, self.Ref, self.PV, self.StatsIO, self.Pa)
        self.Aux.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
        self.StatsIO.close_files(self.Pa)
        return

