
import numpy as np
from mpi4py import MPI
cimport numpy as np
import os
import cython
try:
    import cPickle as pickle
except:
    import pickle as pickle # for Python 3 users


cimport Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport PrognosticVariables
cimport ParallelMPI

cdef class VisualizationOutput:
    def __init__(self, dict namelist, ParallelMPI.ParallelMPI Pa):
        self.uuid = str(namelist['meta']['uuid'])

        try:
            outpath = str(os.path.join(str(namelist['output']['output_root'])
                                   + 'Output.' + str(namelist['meta']['simname']) + '.' + self.uuid[:]))
            self.vis_path = os.path.join(outpath, 'Visualization')
        except:
            self.vis_path = './Visualization.' + self.uuid[:]

        if Pa.rank == 0:
            try:
                os.mkdir(outpath)
            except:
                pass
            try:
                os.mkdir(self.vis_path)
            except:
                pass

        try:
            self.frequency = namelist['visualization']['frequency']
        except:
            self.frequency = 1e6

        return

    cpdef initialize(self):
        self.last_vis_time = 0.0

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef write(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS,
                PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                ParallelMPI.ParallelMPI Pa):

        cdef:
            double [:,:] local_lwp = np.zeros((Gr.dims.n[0], Gr.dims.n[1]), dtype=np.double, order='c')
            double [:,:] reduced_lwp = np.zeros((Gr.dims.n[0], Gr.dims.n[1]), dtype=np.double, order='c')
            Py_ssize_t i,j,k,ijk
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift, jshift

            Py_ssize_t global_shift_i = Gr.dims.indx_lo[0]
            Py_ssize_t global_shift_j = Gr.dims.indx_lo[1]
            Py_ssize_t global_shift_k = Gr.dims.indx_lo[2]

            Py_ssize_t var_shift
            Py_ssize_t i2d, j2d

            double dz = Gr.dims.dx[2]

            dict out_dict = {}

        comm = MPI.COMM_WORLD
        if 'ql' in DV.name_index:
            var_shift =  DV.get_varshift(Gr, 'ql')
            with nogil:
                for i in xrange(imin, imax):
                    ishift = i * istride
                    for j in xrange(jmin, jmax):
                        jshift = j * jstride
                        for k in xrange(kmin, kmax):
                            ijk = ishift + jshift + k
                            i2d = global_shift_i + i - Gr.dims.gw
                            j2d = global_shift_j + j - Gr.dims.gw

                            local_lwp[i2d, j2d] += (RS.rho0[k] * DV.values[var_shift + ijk] * dz)

            comm.Reduce(local_lwp, reduced_lwp, op=MPI.SUM)

            del local_lwp
            if Pa.rank == 0:
                out_dict['lwp'] = np.array(reduced_lwp,dtype=np.double)
            del reduced_lwp


        #Now get horizontal slices

        #Write output of Prognostic Variables and Diagnostic Variables
        cdef:
            double [:,:] local_var
            double [:,:] reduced_var
            list pv_vars = ['qt', 's']
            list dv_vars = ['ql']


        for var in pv_vars:
            local_var = np.zeros((Gr.dims.n[1], Gr.dims.n[2]), dtype=np.double, order='c')
            reduced_var = np.zeros((Gr.dims.n[1], Gr.dims.n[2]), dtype=np.double, order='c')
            if var in PV.name_index:
                var_shift = PV.get_varshift(Gr, var)

                with nogil:
                    if global_shift_i == 0:
                        i = 0
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            for k in xrange(kmin, kmax):
                                ijk = ishift + jshift + k
                                j2d = global_shift_j + j - Gr.dims.gw
                                k2d = global_shift_k + k - Gr.dims.gw
                                local_var[j2d, k2d] = PV.values[var_shift + ijk]

                comm.Reduce(local_var, reduced_var, op=MPI.SUM)
                del local_var
                if Pa.rank == 0:
                    out_dict[var] = np.array(reduced_var, dtype=np.double)
                del reduced_var


                #Now output a horizontal slice
                k = 4
                local_lwp = np.zeros((Gr.dims.n[0], Gr.dims.n[1]), dtype=np.double, order='c')
                reduced_lwp = np.zeros((Gr.dims.n[0], Gr.dims.n[1]), dtype=np.double, order='c')
                with nogil:
                    for i in xrange(imin, imax):
                        ishift = i * istride
                        for j in xrange(jmin, jmax):
                            jshift = j * jstride
                            ijk = ishift + jshift + k
                            i2d = global_shift_i + i - Gr.dims.gw
                            j2d = global_shift_j + j - Gr.dims.gw

                            local_lwp[i2d, j2d] += (PV.values[var_shift + ijk] )

                comm.Reduce(local_lwp, reduced_lwp, op=MPI.SUM)

                del local_lwp
                if Pa.rank == 0:
                    out_dict[var + '_h'] = np.array(reduced_lwp,dtype=np.double)
                del reduced_lwp


            else:
                Pa.root_print('Trouble Writing ' + var)


            for var in dv_vars:

                local_var = np.zeros((Gr.dims.n[1], Gr.dims.n[2]), dtype=np.double, order='c')
                reduced_var = np.zeros((Gr.dims.n[1], Gr.dims.n[2]), dtype=np.double, order='c')
                if var in DV.name_index:
                    var_shift = DV.get_varshift(Gr, var)

                    with nogil:
                        if global_shift_i == 0:
                            i = 1
                            ishift = i * istride
                            for j in xrange(jmin, jmax):
                                jshift = j * jstride
                                for k in xrange(kmin, kmax):
                                    ijk = ishift + jshift + k
                                    j2d = global_shift_j + j - Gr.dims.gw
                                    k2d = global_shift_k + k - Gr.dims.gw
                                    local_var[j2d, k2d] = DV.values[var_shift + ijk]

                    comm.Reduce(local_var, reduced_var, op=MPI.SUM)

                    del local_var
                    if Pa.rank == 0:
                        out_dict[var] = np.array(reduced_var, dtype=np.double)[:,:] - np.mean(reduced_var,axis=0)
                    del reduced_var

                    #Now output a horizontal slice
                    k = 4 
                    local_lwp = np.zeros((Gr.dims.n[0], Gr.dims.n[1]), dtype=np.double, order='c')
                    reduced_lwp = np.zeros((Gr.dims.n[0], Gr.dims.n[1]), dtype=np.double, order='c')
                    with nogil:
                        for i in xrange(imin, imax):
                            ishift = i * istride
                            for j in xrange(jmin, jmax):
                                jshift = j * jstride
                                ijk = ishift + jshift + k
                                i2d = global_shift_i + i - Gr.dims.gw
                                j2d = global_shift_j + j - Gr.dims.gw

                                local_lwp[i2d, j2d] += (DV.values[var_shift + ijk] )

                    comm.Reduce(local_lwp, reduced_lwp, op=MPI.SUM)

                    del local_lwp
                    if Pa.rank == 0:
                        out_dict[var + '_h'] = np.array(reduced_lwp,dtype=np.double)
                    del reduced_lwp

                else:
                    Pa.root_print('Trouble Writing ' + var)


        if Pa.rank == 0:
            with open(self.vis_path+ '/'  + str(10000000 + np.int(self.last_vis_time)) +  '.pkl', 'wb') as f:
                pickle.dump(out_dict, f, protocol=2)

        return
