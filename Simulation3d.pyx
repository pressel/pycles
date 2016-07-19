import time
import numpy as np
cimport numpy as np
# __
# import matplotlib as plt
import pylab as plt
import os
from mpi4py import MPI
try:
    import cPickle as pickle
except:
    import pickle as pickle # for Python 3 users
# __
from Initialization import InitializationFactory, AuxillaryVariables
from Thermodynamics import ThermodynamicsFactory
from Microphysics import MicrophysicsFactory
from Surface import SurfaceFactory
from Radiation import RadiationFactory
from SurfaceBudget import SurfaceBudgetFactory
from AuxiliaryStatistics import AuxiliaryStatistics
from ConditionalStatistics import ConditionalStatistics
# __
# from ConditionalStatistics import NanStatistics
# __
from Thermodynamics cimport LatentHeat
from Tracers import TracersFactory
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
# __
cimport StochasticNoise
# __

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
        # __
        self.SN = StochasticNoise.StochasticNoise(namelist)
        uuid = str(namelist['meta']['uuid'])
        self.outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + uuid[-5:]))
        self.count = 0
        # self.Nan = NanStatistics(self.Gr, self.PV, self.DV, self.CondStatsIO, self.Pa) # -> problem when calling
        # self.Nan.nan_checking('hoi', self.Gr, self.PV, self.DV, self.CondStatsIO, self.Pa)
        # __

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
        self.Tr.initialize(self.Gr, self.PV,self.StatsIO, self.Pa)
        self.PV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Ke.initialize(self.Gr, self.StatsIO, self.Pa)

        self.SA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.MA.initialize(self.Gr,self.PV, self.StatsIO, self.Pa)
        self.SD.initialize(self.Gr,self.PV,self.DV,self.StatsIO,self.Pa)
        self.MD.initialize(self.Gr,self.PV,self.DV,self.StatsIO, self.Pa)

        self.TS.initialize(namelist,self.PV,self.Pa)
        # __
        self.SN.initialize(self.Pa)
        # __

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
            self.Sur.init_from_restart(self.Restart)
            self.StatsIO.last_output_time = self.Restart.restart_data['last_stats_output']
            self.Pa.root_print('Restart output times')
            self.Pa.root_print(str(self.Restart.restart_data['last_stats_output']) + ', ' + str(self.Restart.restart_data['last_condstats_output']) + ', ' + str(self.Restart.restart_data['last_vis_time']))
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

        self.Sur.initialize(self.Gr, self.Ref,  self.StatsIO, self.Pa)


        self.Pr.initialize(namelist, self.Gr, self.Ref, self.DV, self.Pa)
        self.DV.initialize(self.Gr, self.StatsIO, self.Pa)
        self.Fo.initialize(self.Gr, self.Ref,self.StatsIO, self.Pa)
        self.Ra.initialize(self.Gr, self.StatsIO,self.Pa)
        self.Budg.initialize(self.Gr, self.StatsIO,self.Pa)
        self.Damping.initialize(self.Gr, self.Ref)
        self.Aux.initialize(namelist, self.Gr, self.PV, self.DV, self.StatsIO, self.Pa)
        self.CondStats.initialize(namelist, self.Gr, self.PV, self.DV, self.CondStatsIO, self.Pa)

        self.Pa.root_print('Initialization completed!')
        #__
        self.check_nans('Finished Initialization: ')
        #__
        return



    def run(self):
        self.Pa.root_print('Sim: start run')
        cdef PrognosticVariables.PrognosticVariables PV_ = self.PV
        cdef DiagnosticVariables.DiagnosticVariables DV_ = self.DV
        PV_.Update_all_bcs(self.Gr, self.Pa)
        cdef LatentHeat LH_ = self.LH
        cdef Grid.Grid GR_ = self.Gr
        cdef ParallelMPI.ParallelMPI PA_ = self.Pa
        cdef int rk_step
        # DO First Output
        self.Th.update(self.Gr, self.Ref, PV_, DV_)
        self.Ra.initialize_profiles(self.Gr, self.Ref, self.DV, self.StatsIO,self.Pa)

        #Do IO if not a restarted run
        if not self.Restart.is_restart_run:
            self.force_io()

        #_
        self.Pa.root_print('before loop 1')
        PV_.val_nan(self.Pa,'Nan checking in Simulation: time: '+str(self.TS.t))
        self.Pa.root_print('before loop 2')
        # self.debug_tend('before loop')
        self.Pa.root_print('before loop 3')
        #_

        while (self.TS.t < self.TS.t_max):
            time1 = time.time()
            self.Pa.root_print('time: '+str(self.TS.t))
            for self.TS.rk_step in xrange(self.TS.n_rk_steps):
                self.Ke.update(self.Gr,PV_)
                # __
                self.debug_tend('Ke')
                # self.Nan.nan_checking('hoi', self.Gr, self.PV, self.DV, self.CondStatsIO, self.Pa)
                # __

                self.Th.update(self.Gr,self.Ref,PV_,DV_)
                #_
                self.debug_tend('Th')   # only w-tendencies != 0 ?!!!
                #_
                self.Micro.update(self.Gr, self.Ref, PV_, DV_, self.TS, self.Pa )
                #_
                # self.debug_tend('Micro') # only w-tendencies != 0 ?!!!
                #_
                self.Tr.update(self.Gr, self.Ref, PV_, DV_, self.Pa)
                #_
                # self.debug_tend('Tr')
                #_
                self.SA.update(self.Gr,self.Ref,PV_, DV_,  self.Pa)
                #_
                self.debug_tend('SA')
                #_
                self.MA.update(self.Gr,self.Ref,PV_,self.Pa)
                #_
                self.debug_tend('MA')
                #_
                # __
                self.SN.update(self.Gr,self.Ref,PV_,self.Th,self.Pa)
                # __
                self.Sur.update(self.Gr,self.Ref,self.PV, self.DV,self.Pa,self.TS)
                #_
                self.debug_tend('Sur')
                #_
                self.SGS.update(self.Gr,self.DV,self.PV, self.Ke, self.Sur,self.Pa)
                #_
                self.debug_tend('SGS')
                #_
                self.Damping.update(self.Gr, self.Ref,self.PV, self.DV, self.Pa)
                #_
                self.debug_tend('Damping')
                #_

                self.SD.update(self.Gr,self.Ref,self.PV,self.DV)
                #_
                self.debug_tend('SD')
                # _
                self.MD.update(self.Gr,self.Ref,self.PV,self.DV,self.Ke)
                #_
                self.debug_tend('MD')
                #_

                self.Fo.update(self.Gr, self.Ref, self.PV, self.DV, self.Pa)
                #_
                self.debug_tend('Fo')
                #_
                self.Ra.update(self.Gr, self.Ref, self.PV, self.DV, self.Sur, self.TS, self.Pa)
                #_
                # self.debug_tend('Ra')
                #_
                self.Budg.update(self.Gr,self.Ra, self.Sur, self.TS, self.Pa)
                #_
                self.debug_tend('Budg')
                #_
                self.Tr.update_cleanup(self.Gr, self.Ref, PV_, DV_, self.Pa)
                self.TS.update(self.Gr, self.PV, self.Pa)
                #_
                self.debug_tend('TS update') # tendencies set to zero
                #_
                PV_.Update_all_bcs(self.Gr, self.Pa)
                self.Pr.update(self.Gr, self.Ref, self.DV, self.PV, self.Pa)
                self.TS.adjust_timestep(self.Gr, self.PV, self.DV,self.Pa)
                #_
                # self.debug_tend('TS adjust timestep')
                #_
                self.io()
                #PV_.debug(self.Gr,self.Ref,self.StatsIO,self.Pa)
                # self.Pa.root_print('rk_step: '+str(self.TS.rk_step)+' (total steps: '+str(self.TS.n_rk_steps)+')')
            time2 = time.time()
            self.Pa.root_print('T = ' + str(self.TS.t) + ' dt = ' + str(self.TS.dt) +
                               ' cfl_max = ' + str(self.TS.cfl_max) + ' walltime = ' + str(time2 - time1))

        self.Restart.cleanup()


        return

    def io(self):
        self.Pa.root_print('calling io')
        cdef:
            double fields_dt = 0.0
            double stats_dt = 0.0
            double condstats_dt = 0.0
            double restart_dt = 0.0
            double vis_dt = 0.0
            double min_dt = 0.0

        if self.TS.t > 0 and self.TS.rk_step == self.TS.n_rk_steps - 1:
            self.Pa.root_print('doing io: ' + str(self.TS.t) + ', ' + str(self.TS.rk_step))
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
            #if (1==1):
                self.Pa.root_print('Doing 3D FieldIO')
                self.FieldsIO.last_output_time = self.TS.t
                self.FieldsIO.update(self.Gr, self.PV, self.DV, self.TS, self.Pa)
                self.FieldsIO.dump_prognostic_variables(self.Gr, self.PV)
                self.FieldsIO.dump_diagnostic_variables(self.Gr, self.DV, self.Pa)
                self.Pa.root_print('Finished Doing 3D FieldIO')

            # If time to ouput stats do output
            self.Pa.root_print('StatsIO freq: ' + str(self.StatsIO.frequency) + ', ' + str(self.StatsIO.last_output_time) + ', ' + str(self.TS.t))
            if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t:
            #if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t or self.StatsIO.last_output_time + self.StatsIO.frequency + 10.0 == self.TS.t:
            #if (1==1):
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
                self.Tr.stats_io( self.Gr, self.StatsIO, self.Pa)
                self.Ra.stats_io(self.Gr, self.DV, self.StatsIO, self.Pa)
                self.Budg.stats_io(self.Sur, self.StatsIO, self.Pa)
                self.Aux.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
                self.StatsIO.close_files(self.Pa)
                self.Pa.root_print('Finished Doing StatsIO')


            # If time to ouput stats do output
            if self.CondStatsIO.last_output_time + self.CondStatsIO.frequency == self.TS.t:
            #if (1==1):
                self.Pa.root_print('Doing CondStatsIO')
                self.CondStatsIO.last_output_time = self.TS.t
                self.CondStatsIO.write_condstat_time(self.TS.t, self.Pa)

                self.CondStats.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.CondStatsIO, self.Pa)
                self.Pa.root_print('Finished Doing CondStatsIO')


            if self.VO.last_vis_time + self.VO.frequency == self.TS.t:
            #if (1==1):
                self.Pa.root_print('Dumping Visualisation File!')
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
                self.Sur.restart(self.Restart)
                self.Ref.restart(self.Gr, self.Restart)
                self.PV.restart(self.Gr, self.Restart)
                self.TS.restart(self.Restart)

                self.Restart.write(self.Pa)
                self.Pa.root_print('Finished Dumping Restart Files!')






        return

    def force_io(self):
        # output stats here
        self.Pa.root_print('Sim.force_io')

        self.Pa.root_print('Doing 3D FiledIO')
        self.FieldsIO.update(self.Gr, self.PV, self.DV, self.TS, self.Pa)
        self.FieldsIO.dump_prognostic_variables(self.Gr, self.PV)
        self.FieldsIO.dump_diagnostic_variables(self.Gr, self.DV, self.Pa)
        self.Pa.root_print('Finished Doing 3D FieldIO')

        self.StatsIO.open_files(self.Pa)
        self.StatsIO.write_simulation_time(self.TS.t, self.Pa)
        self.PV.stats_io(self.Gr, self.Ref, self.StatsIO, self.Pa)

        self.DV.stats_io(self.Gr, self.StatsIO, self.Pa)
        self.Pa.root_print('Sim.force_io: DV.stats_io finished')
        self.Fo.stats_io(
            self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Pa.root_print('Sim.force_io: Fo.stats_io finished')
        self.Th.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Pa.root_print('Sim.force_io: Th.stats_io finished')
        self.Micro.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
        self.Pa.root_print('Sim.force_io: Micro.stats_io finished')
        self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
        self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke ,self.StatsIO, self.Pa)
        self.SA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
        self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
        self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
        self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)

        self.Pa.root_print('Sim.force_io: MD.stats_io finished')

        self.Ke.stats_io(self.Gr, self.Ref, self.PV, self.StatsIO, self.Pa)

        self.Pa.root_print('Sim.force_io: Kinematics.stats_io finished')

        self.Tr.stats_io( self.Gr, self.StatsIO, self.Pa)
        self.Ra.stats_io(self.Gr, self.DV, self.StatsIO, self.Pa)
        self.Budg.stats_io(self.Sur, self.StatsIO, self.Pa)
        self.Pa.root_print('Sim.force_io: Budg.stats_io finished')
        self.Aux.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
        self.Pa.root_print('Sim.force_io: Aux.stats_io finished')
        self.StatsIO.close_files(self.Pa)

        self.Pa.root_print('Sim.force_io finished')

        return



    def check_nans(self,message):
        cdef PrognosticVariables.PrognosticVariables PV_ = self.PV

        cdef:
            Py_ssize_t u_varshift = PV_.get_varshift(self.Gr,'u')
            Py_ssize_t v_varshift = PV_.get_varshift(self.Gr,'v')
            Py_ssize_t w_varshift = PV_.get_varshift(self.Gr,'w')
            Py_ssize_t s_varshift = PV_.get_varshift(self.Gr,'s')
            Py_ssize_t qt_varshift = PV_.get_varshift(self.Gr,'qt')

        # self.Pa.root_print(u_varshift, v_varshift, w_varshift, s_varshift, qt_varshift)


        # # __
        nan = False
        if np.isnan(PV_.values[u_varshift:v_varshift]).any():
            self.Pa.root_print('u: nan')
        # else:
        #     self.Pa.root_print('u: No nan')
        if np.isnan(PV_.values[v_varshift:w_varshift]).any():
            self.Pa.root_print('v nan')
            nan = True
        # else:
        #     self.Pa.root_print('v: No nan')
        if np.isnan(PV_.values[w_varshift:s_varshift]).any():
            self.Pa.root_print('w: nan')
            nan = True
        # else:
        #     self.Pa.root_print('w: No nan')
        if np.isnan(PV_.values[s_varshift:qt_varshift]).any():
            self.Pa.root_print('s: nan')
            nan = True
        # else:
        #     self.Pa.root_print('s: No nan')
        if np.isnan(PV_.values[qt_varshift:-1]).any():
            self.Pa.root_print('qt: nan')
            nan = True
        # else:
        #     self.Pa.root_print('qt: No nan')

        if nan == True:
            a = message + ': Nans found'
        else:
            a = message + ': No nans found'
        self.Pa.root_print(a)
        return
        # __






    def debug_tend(self,message):
        # print('debug_tend Sim, rank: ', self.Pa.rank)
        cdef:
            PrognosticVariables.PrognosticVariables PV_ = self.PV
            DiagnosticVariables.DiagnosticVariables DV_ = self.DV
            Grid.Grid Gr_ = self.Gr

        cdef:
            Py_ssize_t u_varshift = PV_.get_varshift(self.Gr,'u')
            Py_ssize_t v_varshift = PV_.get_varshift(self.Gr,'v')
            Py_ssize_t w_varshift = PV_.get_varshift(self.Gr,'w')
            Py_ssize_t s_varshift = PV_.get_varshift(self.Gr,'s')

            Py_ssize_t istride = Gr_.dims.nlg[1] * Gr_.dims.nlg[2]
            Py_ssize_t jstride = Gr_.dims.nlg[2]
            Py_ssize_t imax = Gr_.dims.nlg[0]
            Py_ssize_t jmax = Gr_.dims.nlg[1]
            Py_ssize_t kmax = Gr_.dims.nlg[2]
            Py_ssize_t ijk_max = imax*istride + jmax*jstride + kmax

            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t imin = 0#Gr_.dims.gw
            Py_ssize_t jmin = 0#Gr_.dims.gw
            Py_ssize_t kmin = 0#Gr_.dims.gw
            # int [:] sk_arr = np.zeros(1, dtype=np.int)
            # int [:] qtk_arr = np.zeros(1, dtype=int)
        sk_arr = np.zeros(1,dtype=np.int)
        qtk_arr = np.zeros(1,dtype=np.int)

        u_max = np.nanmax(PV_.tendencies[u_varshift:v_varshift])
        uk_max = np.nanargmax(PV_.tendencies[u_varshift:v_varshift])
        u_min = np.nanmin(PV_.tendencies[u_varshift:v_varshift])
        uk_min = np.nanargmin(PV_.tendencies[u_varshift:v_varshift])
        v_max = np.nanmax(PV_.tendencies[v_varshift:w_varshift])
        vk_max = np.nanargmax(PV_.tendencies[v_varshift:w_varshift])
        v_min = np.nanmin(PV_.tendencies[v_varshift:w_varshift])
        vk_min = np.nanargmin(PV_.tendencies[v_varshift:w_varshift])
        w_max = np.nanmax(PV_.tendencies[w_varshift:s_varshift])
        wk_max = np.nanargmax(PV_.tendencies[w_varshift:s_varshift])
        w_min = np.nanmin(PV_.tendencies[w_varshift:s_varshift])
        wk_min = np.nanargmin(PV_.tendencies[w_varshift:s_varshift])

        u_nan = np.isnan(PV_.tendencies[u_varshift:v_varshift]).any()
        uk_nan = np.argmax(PV_.tendencies[u_varshift:v_varshift])
        v_nan = np.isnan(PV_.tendencies[v_varshift:w_varshift]).any()
        vk_nan = np.argmax(PV_.tendencies[v_varshift:w_varshift])
        w_nan = np.isnan(PV_.tendencies[w_varshift:s_varshift]).any()
        wk_nan = np.argmax(PV_.tendencies[w_varshift:s_varshift])

        w_max_val= np.nanmax(PV_.values[w_varshift:s_varshift])
        wk_max_val = np.nanargmax(PV_.values[w_varshift:s_varshift])
        w_min_val = np.nanmin(PV_.values[w_varshift:s_varshift])
        wk_min_val = np.nanargmin(PV_.tendencies[w_varshift:s_varshift])
        w_nan_val = np.isnan(PV_.values[w_varshift:s_varshift]).any()
        wk_nan_val = np.argmax(PV_.values[w_varshift:s_varshift])

        #if self.Pa.rank == 0:
        if 1 == 0:
            print(message, 'debugging (max, min, nan): ')
            print('shifts', u_varshift, v_varshift, w_varshift, s_varshift)
            print('u tend: ', u_max, uk_max, u_min, uk_min, u_nan, uk_nan)
            print('v tend: ', v_max, vk_max, v_min, vk_min, v_nan, vk_nan)
            print('w tend: ', w_max, wk_max, w_min, wk_min, w_nan, wk_nan)
            print('w val: ', w_max_val, wk_max_val, w_min_val, wk_min_val, w_nan_val, wk_nan_val)

        if 'qt' in PV_.name_index:
            qt_varshift = PV_.get_varshift(self.Gr,'qt')
            ql_varshift = DV_.get_varshift(self.Gr,'ql')

            s_max = np.nanmax(PV_.tendencies[s_varshift:qt_varshift])
            sk_max = np.nanargmax(PV_.tendencies[s_varshift:qt_varshift])
            s_min = np.nanmin(PV_.tendencies[s_varshift:qt_varshift])
            sk_min = np.nanargmin(PV_.tendencies[s_varshift:qt_varshift])
            qt_max = np.nanmax(PV_.tendencies[qt_varshift:-1])
            qtk_max = np.nanargmax(PV_.tendencies[qt_varshift:-1])
            qt_min = np.nanmin(PV_.tendencies[qt_varshift:-1])
            qtk_min = np.nanargmin(PV_.tendencies[qt_varshift:-1])

            s_nan = np.isnan(PV_.tendencies[s_varshift:qt_varshift]).any()
            sk_nan = np.argmax(PV_.tendencies[s_varshift:qt_varshift])
            qt_nan = np.isnan(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)]).any()
            qtk_nan = np.argmax(PV_.tendencies[qt_varshift:(qt_varshift + ijk_max)])

            s_max_val= np.nanmax(PV_.values[s_varshift:qt_varshift])
            sk_max_val = np.nanargmax(PV_.values[s_varshift:qt_varshift])
            s_min_val = np.nanmin(PV_.values[s_varshift:qt_varshift])
            sk_min_val = np.nanargmin(PV_.tendencies[s_varshift:qt_varshift])
            s_nan_val = np.isnan(PV_.values[s_varshift:qt_varshift]).any()
            sk_nan_val = np.argmax(PV_.values[s_varshift:qt_varshift])
            qt_max_val = np.nanmax(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            qtk_max_val = np.nanargmax(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            qt_min_val = np.nanmin(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            if qt_min_val < 0:
                self.Pa.root_print('qt val negative')
            qtk_min_val = np.nanargmin(PV_.values[qt_varshift:(qt_varshift + ijk_max)])
            qt_nan_val = np.isnan(PV_.values[qt_varshift:(qt_varshift + ijk_max)]).any()
            qtk_nan_val = np.argmax(PV_.values[qt_varshift:(qt_varshift + ijk_max)])

            ql_max_val = np.nanmax(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
            ql_min_val = np.nanmin(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
            qlk_max_val = np.nanargmax(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
            qlk_min_val = np.nanargmin(DV_.values[ql_varshift:(ql_varshift+ijk_max)])
            ql_nan_val = np.isnan(DV_.values[ql_varshift:(ql_varshift+ijk_max)]).any()
            qlk_nan_val = np.argmax(DV_.values[ql_varshift:(ql_varshift+ijk_max)])

            #if self.Pa.rank == 0:
            if 1 == 0:
                print('s tend: ', s_max, sk_max, s_min, sk_min, s_nan, sk_nan)
                print('s val: ', s_max_val, sk_max_val, s_min_val, sk_min_val, s_nan_val, sk_nan_val)
                print('qt tend: ', qt_max, qtk_max, qt_min, qtk_min, qt_nan, qtk_nan)
                print('qt val: ', qt_max_val, qtk_max_val, qt_min_val, qtk_min_val, qt_nan_val, qtk_nan_val)
                print('ql val: ', ql_max_val, qlk_max_val, ql_min_val, qlk_min_val, ql_nan_val, qlk_nan_val)
            #self.Pa.root_print('ql: ' + str(ql_max) + ', ' + str(ql_min))

        #for name in PV.name_index.keys():
            # with nogil:
            if 1 == 1:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            if np.isnan(PV_.values[s_varshift+ijk]):
                                sk_arr = np.append(sk_arr,ijk)
                            if np.isnan(PV_.values[qt_varshift+ijk]):
                                qtk_arr = np.append(qtk_arr,ijk)

            if np.size(sk_arr) > 1:
                # self.output_nan_array(sk_arr,'s',message, self.Pa)
                if self.Pa.rank == 0:
                    print('sk_arr size: ', sk_arr.shape)
                    print('sk_arr:', sk_arr)
                    # self.output_nan_array()
            if np.size(qtk_arr) > 1:
                # self.output_nan_array(qtk_arr,'qt',message, self.Pa)
                if self.Pa.rank == 0:
                    print('qtk_arr size: ', qtk_arr.shape)
                    print('qtk_arr: ', qtk_arr)





        else:
            s_max = np.nanmax(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            sk_max = np.nanargmax(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            s_min = np.nanmin(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            sk_min = np.nanargmin(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            s_nan = np.isnan(PV_.tendencies[s_varshift:(s_varshift + ijk_max)]).any()
            sk_nan = np.argmax(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])

            s_max_val= np.nanmax(PV_.values[s_varshift:(s_varshift + ijk_max)])
            sk_max_val = np.nanargmax(PV_.values[s_varshift:(s_varshift + ijk_max)])
            s_min_val = np.nanmin(PV_.values[s_varshift:(s_varshift + ijk_max)])
            sk_min_val = np.nanargmin(PV_.tendencies[s_varshift:(s_varshift + ijk_max)])
            s_nan_val = np.isnan(PV_.values[s_varshift:(s_varshift + ijk_max)]).any()
            sk_nan_val = np.argmax(PV_.values[s_varshift:(s_varshift + ijk_max)])

            #if self.Pa.rank == 0:
            if 1 == 0:
                print('s tend: ', s_max, sk_max, s_min, sk_min, s_nan, sk_nan)
                print('s val: ', s_max_val, sk_max_val, s_min_val, sk_min_val, s_nan_val, sk_nan_val)


            if 1 == 1:
                for i in range(imin, imax):
                    ishift = i * istride
                    for j in range(jmin, jmax):
                        jshift = j * jstride
                        for k in range(kmin, kmax):
                            ijk = ishift + jshift + k
                            if np.isnan(PV_.values[s_varshift+ijk]):
                                sk_arr = np.append(sk_arr,ijk)


            if np.size(sk_arr) > 1:
                # self.output_nan_array(sk_arr,'s',message, self.Pa)
                if self.Pa.rank == 0:
                    print('sk_arr size: ', sk_arr.shape)
                    print('sk_arr:', sk_arr)




        # cdef:
        #     Py_ssize_t i,j,k,ijk
        #     Py_ssize_t global_shift_i = Gr_.dims.indx_lo[0]
        #     Py_ssize_t global_shift_j = Gr_.dims.indx_lo[1]
        #     Py_ssize_t global_shift_k = Gr_.dims.indx_lo[2]
        #
        #     Py_ssize_t var_shift
        #     Py_ssize_t i2d, j2d, k2d
        #
        #     Py_ssize_t ishift, jshift
        #     Py_ssize_t gw = Gr_.dims.gw
        #
        #     Py_ssize_t imin = Gr_.dims.gw
        #     Py_ssize_t jmin = Gr_.dims.gw
        #     Py_ssize_t kmin = Gr_.dims.gw
        #     Py_ssize_t imax_ = Gr_.dims.nlg[0] - Gr_.dims.gw
        #     Py_ssize_t jmax_ = Gr_.dims.nlg[1] - Gr_.dims.gw
        #     Py_ssize_t kmax_ = Gr_.dims.nlg[2] - Gr_.dims.gw
        #     Py_ssize_t i_s, i_w
        #
        # comm = MPI.COMM_WORLD
        #
        # cdef:
        #     double [:,:] local_var
        #     double [:,:] reduced_var
        #     list pv_vars = ['s', 'w']
        #
        # if np.isnan(PV_.tendencies).any():
        #     if s_nan == True:
        #         print('s nan')
        #         a = np.unravel_index(sk_nan, (imax,jmax,kmax))
        #         i_s = a[0]
        #         print(i_s)
        #     # if qt_nan == True:
        #     #     a = np.unravel_index(qtk_nan, (imax,jmax,kmax))
        #     #     i_qt = a[0]
        #     if w_nan == True:
        #         print('w nan')
        #         a = np.unravel_index(wk_nan, (imax,jmax,kmax))
        #         i_w = a[0]
        #     # how to find indices of nan value?
        #     # i_s = np.argwhere(x==1)
        #     for var in pv_vars:
        #         local_var = np.zeros((Gr_.dims.n[1], Gr_.dims.n[2]), dtype=np.double, order='c')
        #         reduced_var = np.zeros((Gr_.dims.n[1], Gr_.dims.n[2]), dtype=np.double, order='c')
        #         var_shift = PV_.get_varshift(self.Gr, var)
        #
        #         if var == 'w':
        #             i = i_w
        #         elif var == 's':
        #             print('')
        #             i = i_s
        #         with nogil:
        #             if global_shift_i == 0:
        #                 # i = 0
        #                 ishift = i * istride
        #                 for j in xrange(jmin, jmax_):
        #                     jshift = j * jstride
        #                     for k in xrange(kmin, kmax_):
        #                         ijk = ishift + jshift + k
        #                         j2d = global_shift_j + j - gw
        #                         k2d = global_shift_k + k - gw
        #                         local_var[j2d, k2d] = PV_.values[var_shift + ijk]
        #         comm.Reduce(local_var, reduced_var, op=MPI.SUM)
        #         del local_var
        #
        #         plt.figure(1)
        #         plt.contourf(reduced_var.T)
        #         plt.title(var + ', ' + message)
        #         # plt.show()
        #         plt.savefig(self.outpath + '/' + var + '_' + message + '.png')
        #         plt.close()

        # PV_.values[s_varshift+1] = np.nan

        return


    def output_nan_array(self,arr,name,message,ParallelMPI.ParallelMPI Pa):

        # return
        ## self.Pa.root_print('!!! output nan array, rank: ' + str(self.Pa.rank))
        print(('!!! output nan array, rank: ' + str(Pa.rank)))
        print(self.outpath)
        # if 's' in self.PV.name_index:
        #     self.NC.write_condstat('sk_arr', 'nan_array', self.sk_arr[:,:], self.Pa)
        # if 'qt' in self.PV.name_index:
        #     self.NC.write_condstat('qtk_arr', 'nan_array', self.qt_arr[:,:], self.Pa)

        out_path = os.path.join(self.outpath, 'Nan')
        print('outpath', out_path)
        # print('time', self.TS.t, str(self.TS.t))
        if Pa.rank == 0:
            try:
                os.mkdir(out_path)
                print('doing out_path', self.outpath)
            except:
                print('NOT doing out_path')
                pass
            try:
                path = out_path + '/' + name + 'k_arr' + str(self.TS.t) + '_' + message[0:2]
                # path = out_path + '/' + name + 'k_arr' + str(np.int(self.TS.t)) + '_' + str(np.int(self.count))
                # path = out_path + '/sk_arr_' + str(np.int(self.TS.t))
                os.mkdir(path)
                print('doing path', path)
            except:
                print('NOT doing path')
                pass
        Pa.barrier()

        # path = self.outpath + 'Nan/sk_arr_' + str(np.int(self.TS.t))
        # path = out_path + '/sk_arr_' + str(np.int(self.TS.t))
        with open(path+ '/' + str(Pa.rank) + '.pkl', 'wb') as f:       # 'wb' = write binary file
            # pass
            print('dumping nan pickle: ', Pa.rank, path+ '/' + str(Pa.rank) + '.pkl')
            pickle.dump(arr, f,protocol=2)

        # self.count += 1
        print('finished dumping nan pickle')

        return
