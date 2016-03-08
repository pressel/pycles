#pragma once        // preprocessor statement

#include <stdio.h>
#include <mpi.h>

// how to return a double array???
// double horizontal_mean(struct DimStruct *dims, double* restrict *values){
void horizontal_mean(struct DimStruct *dims, double* restrict values){
        /*
        Compute the horizontal mean of the array pointed to by values.
        values should have dimension of Gr.dims.nlg[0] * Gr.dims.nlg[1]
        * Gr.dims.nlg[1].

        :param Gr: Grid class
        :param values1: pointer to array of type double containing first value in product
        :return: memoryview type double with dimension Gr.dims.nlg[2]
        '''
        # Gr.dims.n[i] = namelist['grid']['ni'] (e.g. n[0] = 'nx')      --> total number of pts
        # Gr.dims.nl[i] = Gr.dims.n[i] // mpi_dims[i]                   --> local number of pts (per processor)
        # Gr.dims.nlg[i] = Gr.dims.nl[i] + 2*gw                         --> local number of pts incl ghost points
        # i = 0,1,2
        */

        printf("values[0] = %f\n", values[0]);
//        double *mean_local = (double *)malloc(sizeof(double) * dims->n[2]);       // Dynamically allocate array
//        double *mean_ = (double *)malloc(sizeof(double) * dims->n[2]);
//        double *mean = (double *)malloc(sizeof(double) * dims->n[2]);
        double *mean_local = (double *)malloc(sizeof(double) * dims->nlg[2]);       // Dynamically allocate array
        double *mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
        double *mean = (double *)malloc(sizeof(double) * dims->nlg[2]);
        //int i,j,k,ijk;
        int ijk;
        const ssize_t  imin = dims->gw;
        const ssize_t  jmin = dims->gw;
        const ssize_t  kmin = 0;
        const ssize_t  imax = dims->nlg[0] - dims->gw;
        const ssize_t  jmax = dims->nlg[1] - dims->gw;
        const ssize_t  kmax = dims->nlg[2];
        const ssize_t  istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t  jstride = dims->nlg[2];
        //int ishift, jshift;
        const double n_horizontal_i = 1.0/(dims->n[1]*dims->n[0]);

        for(ssize_t k=kmin; k<kmax; k++){
            mean_local[k] = 0;
        }

        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin; k<kmax; k++){
                    ijk = ishift + jshift + k;
                    mean_local[k] += values[ijk];
                }
            }
        }
//        printf("nx*ny = %f\n", 1/n_horizontal_i);
        printf("mean_local[0] = %f\n", mean_local[0]);
//        printf("mean_local[10] = %f\n", mean_local[0]);

        //#Here we call MPI_Allreduce on the sub_xy communicator as we only need communication among
        //#processes with the the same vertical rank
        // MPI_Reduce(&mean_local, &mean, dims->ng[2], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//        MPI_Allreduce(&mean_local, &mean_, dims->n[2], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for(ssize_t k=kmin; k<kmax; k++){
            MPI_Allreduce(&mean_local[k], &mean_[k], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            mean[k] = mean_[k] * n_horizontal_i;
        }
        printf("mean[0] = %f\n", mean[0]);
//        printf("mean[10] = %f\n", mean[0]);

        free(mean_local);
        free(mean_);

        return;
}


void horizontal_mean_return(struct DimStruct *dims, double* restrict values, double* restrict mean){
//        /*
//        Compute the horizontal mean of the array pointed to by values.
//        values should have dimension of Gr.dims.nlg[0] * Gr.dims.nlg[1]
//        * Gr.dims.nlg[1].
//
//        :param Gr: Grid class
//        :param values1: pointer to array of type double containing first value in product
//        :return: memoryview type double with dimension Gr.dims.nlg[2]
//        '''
//        # Gr.dims.n[i] = namelist['grid']['ni'] (e.g. n[0] = 'nx')      --> total number of pts
//        # Gr.dims.nl[i] = Gr.dims.n[i] // mpi_dims[i]                   --> local number of pts (per processor)
//        # Gr.dims.nlg[i] = Gr.dims.nl[i] + 2*gw                         --> local number of pts incl ghost points
//        # i = 0,1,2
//        */
//
        const ssize_t gw = dims->gw;
        printf("values[gw] = %f\n", values[gw]);
        printf("before: mean[gw] = %f\n", mean[gw]);

        double *mean_local = (double *)malloc(sizeof(double) * dims->nlg[2]);       // Dynamically allocate array
        double *mean_ = (double *)malloc(sizeof(double) * dims->nlg[2]);
        int ijk;
        const ssize_t  imin = dims->gw;
        const ssize_t  jmin = dims->gw;
        const ssize_t  kmin = 0;
        const ssize_t  imax = dims->nlg[0] - dims->gw;
        const ssize_t  jmax = dims->nlg[1] - dims->gw;
        const ssize_t  kmax = dims->nlg[2];
        const ssize_t  istride = dims->nlg[1] * dims->nlg[2];
        const ssize_t  jstride = dims->nlg[2];
        //int ishift, jshift;
        const double n_horizontal_i = 1.0/(dims->n[1]*dims->n[0]);

        for(ssize_t k=kmin; k<kmax; k++){
            mean_local[k] = 0;
        }

        for(ssize_t i=imin; i<imax; i++){
            const ssize_t ishift = i * istride;
            for(ssize_t j=jmin; j<jmax; j++){
                const ssize_t jshift = j * jstride;
                for(ssize_t k=kmin; k<kmax; k++){
                    ijk = ishift + jshift + k;
                    mean_local[k] += values[ijk];
                }
            }
        }
        printf("nx*ny = %f\n", 1/n_horizontal_i);
        printf("mean_local[gw] = %f\n", mean_local[gw]);
        printf("mean_local[10] = %f\n", mean_local[10]);

//        //#Here we call MPI_Allreduce on the sub_xy communicator as we only need communication among
//        //#processes with the the same vertical rank
//        // MPI_Reduce(&mean_local, &mean, dims->ng[2], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//        MPI_Allreduce(&mean_local, &mean_, dims->n[2], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for(ssize_t k=kmin; k<kmax; k++){
            MPI_Allreduce(&mean_local[k], &mean_[k], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            mean[k] = mean_[k] * n_horizontal_i;
        }
        printf("mean[gw] = %f\n", mean[gw]);
        printf("mean[10] = %f\n", mean[10]);

        return;
}
        /*
        Open MPI: https://www.open-mpi.org/doc/v1.8/man3/MPI_Allreduce.3.php
        http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/

        int MPI_Allreduce(void* send_data, void* recv_data, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm communicator)
        send_data = array of data that should be reduced
        recv_data = contains reduced data
        count = size of recvbuf (= dims->ng[2] ?)
        datatype = datatype of sendbuf
        op = MPI_SUM
        */

        /*
        // Example for Averaging using MPI_Allreduce:
        rand_nums = create_rand_nums(num_elements_per_proc);
        // Sum the numbers locally
        float local_sum = 0;
        int i;
        for (i = 0; i < num_elements_per_proc; i++) {
          local_sum += rand_nums[i];
        }

        // Reduce all of the local sums into the global sum in order to
        // calculate the mean
        float global_sum;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        float mean = global_sum / (num_elements_per_proc * world_size);
        */


        /*
        // Example for Averaging from: http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/

        float *rand_nums = NULL;
        rand_nums = create_rand_nums(num_elements_per_proc);

        // Sum the numbers locally
        float local_sum = 0;
        int i;
        for (i = 0; i < num_elements_per_proc; i++) {
          local_sum += rand_nums[i];
        }

        // Print the random numbers on each process
        printf("Local sum for process %d - %f, avg = %f\n",
               world_rank, local_sum, local_sum / num_elements_per_proc);

        // Reduce all of the local sums into the global sum
        float global_sum;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
                   MPI_COMM_WORLD);

        // Print the result
        if (world_rank == 0) {
          printf("Total sum = %f, avg = %f\n", global_sum,
                 global_sum / (world_size * num_elements_per_proc));
        }*/

        // in Cython:
        // mpi.MPI_Allreduce(&mean_local[0],&mean[0],Gr.dims.nlg[2],mpi.MPI_DOUBLE,mpi.MPI_SUM,self.cart_comm_sub_xy)


        //return mean;




