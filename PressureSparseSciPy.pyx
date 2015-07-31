cimport ParallelMPI
cimport Grid
cimport ReferenceState

import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, gmres
import numpy as np
cimport numpy as np

cdef class PressureSparseSciPy:

    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS):

        cdef:
            double [:] data = np.zeros((Gr.dims.npd * 7,),dtype=np.double)
            long [:] row_ind = np.zeros((Gr.dims.npd * 7,),dtype=np.int)
            long [:] col_ind = np.zeros((Gr.dims.npd * 7,),dtype=np.int)




            long i,j,k,ijk,istride,jstride,count=0




        istride = Gr.dims.nl[1] * Gr.dims.nl[2]
        jstride = Gr.dims.nl[2]

        import time
        time1 = time.time()


        for i in xrange(Gr.dims.nl[0]):
            for j in xrange(Gr.dims.nl[1]):
                for k in xrange(Gr.dims.nl[2]):

                    if (i == 0 or j == 0 or k== 0 or i == Gr.dims.nl[0] - 1
                        or j == Gr.dims.nl[1] -1 or Gr.dims.nl[2] -1):
                        ijk = i * istride + j*jstride + k

                        if(i != Gr.dims.nl[0] - 1):
                            data[count] = 1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[0] * Gr.dims.dxi[0]
                            row_ind[count] = ijk
                            col_ind[count] = istride * (Gr.dims.nl[0] - 1) + j*jstride + k
                            count += 1
                        if(i != 0):
                            data[count] =1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[0]*Gr.dims.dxi[0]
                            row_ind[count] = ijk
                            col_ind[count] = j*jstride + k
                            count += 1
                        if(j!=Gr.dims.nl[1] - 1):
                            data[count] = 1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[1] * Gr.dims.dxi[1]
                            row_ind[count] = ijk
                            col_ind[count] = istride * i + (Gr.dims.nl[0] - 1)*jstride + k
                            count += 1
                        if(j != 0):
                            data[count] = 1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[1] * Gr.dims.dxi[1]
                            row_ind[count] = ijk
                            col_ind[count] = istride*i + k
                            count += 1
                        if(k!=Gr.dims.nl[2] - 1):
                            data[count] = 1.0/RS.alpha0[1 + Gr.dims.gw]  * Gr.dims.dxi[2] * Gr.dims.dxi[2]
                            row_ind[count] = ijk
                            col_ind[count] = 1 +  i * istride + j * jstride
                            count += 1
                        if(k!=0):
                            data[count] = 1.0/RS.alpha0[k - 2 + Gr.dims.gw]  * Gr.dims.dxi[2] * Gr.dims.dxi[2]
                            row_ind[count] = ijk #Gr.dims.nl[2] - 2 +  i * istride + j * jstride
                            col_ind[count] = Gr.dims.nl[2] - 2 +  i * istride + j * jstride
                            count += 1

        self.A = sp.csr_matrix((data[:count],(row_ind[:count],col_ind[:count])),shape=(Gr.dims.npd, Gr.dims.npd),dtype=np.double)

        #Now set boundary conditions in a second sparse matix that will be added to self.A then deleted
        data[:] = 0
        row_ind[:] = 0
        col_ind[:] = 0

        count = 0
        for i in xrange(Gr.dims.nl[0]):
            for j in xrange(Gr.dims.nl[1]):
                for k in xrange(Gr.dims.nl[2]):

                    if ~(i == 0 or j == 0 or k== 0 or i == Gr.dims.nl[0] - 1
                        or j == Gr.dims.nl[1] -1 or Gr.dims.nl[2] -1):
                            ijk = i * istride + j*jstride + k

                            data[count] = ((-1.0/RS.alpha0[k+Gr.dims.gw] -1.0/RS.alpha0[k-1+Gr.dims.gw])*Gr.dims.dxi[2] * Gr.dims.dxi[2]
                                        - 2.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[0] * Gr.dims.dxi[0] - 2.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[1] * Gr.dims.dxi[1])
                            row_ind[count] = ijk
                            col_ind[count] = ijk
                            count +=1

                            # #This is the i+1 point
                            data[count] = 1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[0] * Gr.dims.dxi[0]
                            row_ind[count] = ijk
                            col_ind[count] = istride * (i + 1) + jstride * j + k
                            count += 1

                            #
                            # #This is the i-1 point
                            data[count] = 1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[0] * Gr.dims.dxi[0]
                            row_ind[count] = ijk
                            col_ind[count] = istride * (i -1) + jstride * j + k
                            count += 1
                            #
                            #
                            # #This is the j+1 point
                            data[count] = 1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[1] * Gr.dims.dxi[1]
                            row_ind[count] = ijk
                            col_ind[count] = istride * (i) + jstride * (j+1) + k
                            count += 1
                            #
                            # #This is the j-1 point
                            data[count] = 1.0/RS.alpha0_half[k + Gr.dims.gw]  * Gr.dims.dxi[1] * Gr.dims.dxi[1]
                            row_ind[count] = ijk
                            col_ind[count] = istride * (i) + jstride * (j-1) + k
                            count += 1
                        #
                        # #This is the k+1 point
                            data[count] = 1.0/RS.alpha0[k  + Gr.dims.gw]  * Gr.dims.dxi[2] * Gr.dims.dxi[2]
                            row_ind[count] = ijk
                            col_ind[count] = istride * (i) + jstride * j  + k + 1
                            count += 1
                        #
                            data[count] = 1.0/RS.alpha0[k - 1 + Gr.dims.gw]  * Gr.dims.dxi[2] * Gr.dims.dxi[2]
                            row_ind[count] = ijk
                            col_ind[count] = istride * (i) + jstride * j + k - 1
                            count += 1





        bc_matrix = sp.csr_matrix((data[:count],(row_ind[:count],col_ind[:count])),shape=(Gr.dims.npd, Gr.dims.npd),dtype=np.double)

        #Now add in the boundary condition and delete un-needed memory
        self.A += bc_matrix
        del bc_matrix, data, row_ind, col_ind
        time2 = time.time()

        print(time2 - time1 )

        print(np.array(self.A[:,:]))

        import pylab as plt

        plt.figure(1)
        plt.spy(self.A)
        plt.show()

        pass


    cpdef solve(self,Grid.Grid Gr, ReferenceState.ReferenceState RS, double [:] divergence, ParallelMPI.ParallelMPI PM):

        print(np.shape(np.array(divergence)))
        print(self.A.shape)


        x,info = bicgstab(self.A,divergence)
        print(info)

        pass