import cython


@cython.boundscheck(False)  # Turn off numpy array index bounds checking
@cython.wraparound(False)  # Turn off numpy array wrap around indexing
@cython.cdivision(True)
cpdef to_3d(double[:] f_data, int nl_0, int nl_1, int nl_2, int indx_lo_0,
            int indx_lo_1, int indx_lo_2, double[:, :, :] f_data_3d):

    cdef:
        int istride = nl_2 * nl_1
        int jstride = nl_2
        int ishift
        int jshift

        int i, j, k

    with nogil:
        for i in range(nl_0):
            ishift = i * istride
            for j in range(nl_1):
                jshift = j * jstride
                for k in range(nl_2):
                    ijk = ishift + jshift + k
                    f_data_3d[
                        indx_lo_0 + i, indx_lo_1 + j, indx_lo_2 + k] = f_data[ijk]
