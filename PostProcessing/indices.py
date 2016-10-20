# from 1D index to 3D indices

import numpy as np
import json as  simplejson
#----------------------------------------------------------------------

def main():
    ijk_ = 40852
    ijk_ = 40853
    ijk_ = 240937
    ijk_ = 13841

    case_name = 'Bomex'
    path = './../Output.' + case_name + '.e0f33/'
    print(path)

    var = 's'

    read_in_nml(path,case_name)
    find_indices(var,ijk_)
    return

#----------------------------------------------------------------------
def read_in_nml(path,case_name):

    global n, nl, nlg, npg, nprocx, nprocy, gw       # global variable has to be declared as global in same module where it is defined
    n = np.zeros(3, dtype=np.int)  # (e.g. n[0] = 'nx')      --> global number of pts per direction
    nl = np.zeros(3, dtype=np.int)  # --> local number of pts (per processor)
    nlg = np.zeros(3, dtype=np.int)  # --> local number of pts incl ghost pts
    try:
        # print(path + case_name + '.in')
        nml = simplejson.loads(open(path + case_name + '.in').read())
        print('found nml file')
        n[0] = nml['grid']['nx']
        n[1] = nml['grid']['ny']
        n[2] = nml['grid']['nz']
        gw = nml['grid']['gw']
        nprocx = nml['mpi']['nprocx']
        nprocy = nml['mpi']['nprocy']
    except IOError:
        print('no such file')
        dx = 25
        dy = 25
        dz = 25
        n[0] = 240
        n[1] = 240
        n[2] = 140
        gw = 4
        nprocx = 12
        nprocy = 4

    nl[0] = n[0]/nprocx
    nl[1] = n[1]/nprocy
    nl[2] = n[2]
    nlg[0] = n[0]/nprocx + 2*gw
    nlg[1] = n[1]/nprocy + 2*gw
    nlg[2] = n[2] + 2*gw
    npg = nlg[0] * nlg[1] * nlg[2]

    return
#----------------------------------------------------------------------

def find_indices(var,ijk_):
    print('loop through indices')
    #        double [:] blh = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

    imax = nlg[0]
    jmax = nlg[1]
    kmax = nlg[2]
    ijkmax = imax * nlg[1] * nlg[2] + jmax * nlg[2] + kmax
    print('grid: ', imax, jmax, kmax, ijkmax)

    var_shift = get_varshift(var)
    for i in xrange(imax):
        ishift =  i * nlg[1] * nlg[2]
        for j in xrange(jmax):
            jshift = j * nlg[2]
            for k in xrange(kmax):
                ijk = ishift + jshift + k
                # print(i,j,k)
                if var_shift + ijk == ijk_:
                    print('juhui: ', i, j, k)
                else:
                    pass
                    # print('ohoh')
                if ijk == ijkmax and ijk != ijk_:
                    print('no values found')
    return

#----------------------------------------------------------------------
def get_varshift(var):
    # cdef inline Py_ssize_t get_varshift(self, Grid.Grid Gr, str variable_name):
    #     return self.name_index[variable_name] * Gr.dims.npg
    if var == 'w':
        varshift = 2 * npg
    if var == 's':
        varshift = 3 * npg
    if var == 'buoyancy':
        varshift = 0 * npg

    return varshift
#----------------------------------------------------------------------

if __name__ == '__main__':
    main()


