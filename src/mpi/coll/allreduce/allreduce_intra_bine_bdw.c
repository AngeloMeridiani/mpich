/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_bine.h"

int MPIR_Allreduce_intra_bine_bdw(const void *sendbuf,
                                  void *recvbuf,
                                  MPI_Aint count,
                                  MPI_Datatype datatype,
                                  MPI_Op op, MPIR_Comm* comm_ptr, int coll_attr) 
{
    int comm_size, rank, dest, steps, step, mpi_errno = MPI_SUCCESS;
    int adjsize, extra_ranks, is_power_of_two;
    int new_rank, loop_flag = 0;
    int phase;
    int *r_count = NULL, *s_count = NULL, *r_index = NULL, *s_index = NULL;
    int phase_scount, phase_rcount, num_phases, inbi, vdest;
    MPI_Aint w_size, segsize, segcount;
    MPI_Aint bine_allreduce_segsize = 0;
    int vrank;
    char *tmp_send = NULL, *tmp_recv = NULL;
    char *tmp_recv_phase = NULL, *tmp_send_phase = NULL;
    char *inbuf[2] = {NULL, NULL}, *inbuf_free[2] = {NULL, NULL};
    MPI_Aint lb = 0, extent, true_extent, inbuf_size;
    MPIR_Request* reqs[2] = {NULL, NULL};

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /* Special case for comm_size == 1 */
    if (comm_size == 1) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        }
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Determine nearest power of two less than or equal to comm_size
     * and return an error if comm_size is 0
     */
    steps = MPII_Bine_hibit(comm_size, (int)(sizeof(comm_size) * CHAR_BIT) - 1);
    MPIR_ERR_CHKANDJUMP(steps == -1, mpi_errno, MPI_ERR_ARG, "**arg");
    adjsize = 1 << steps; /* Largest power of two <= comm_size */

    /* Number of nodes that exceed the largest power of two less than or equal
     * to comm_size
     */
    extra_ranks = comm_size - adjsize;
    is_power_of_two = (comm_size & (comm_size - 1)) == 0;

    MPIR_Datatype_get_extent_macro(datatype, extent);
    MPIR_Type_get_true_extent_impl(datatype, &lb, &true_extent);

    segsize = bine_allreduce_segsize;
    if (segsize == 0) {
        segcount = count;
        segsize = segcount * extent;
    } else {
        segcount = segsize / extent; /* Number of elements in a segment */
    }

    /* Allocate temporary buffer for send/recv and reduce operations */
    inbuf_size = (segcount < (count >> 1))
                     ? true_extent + extent * segcount
                     : true_extent + extent * (count >> 1);
    MPIR_CHKLMEM_MALLOC(inbuf_free[0], inbuf_size);
    MPIR_CHKLMEM_MALLOC(inbuf_free[1], inbuf_size);
    inbuf[0] = inbuf_free[0] - lb;
    inbuf[1] = inbuf_free[1] - lb;

    /* First part of computation to get a 2^n number of nodes.
     * What happens is that first #extra_rank even nodes sends their
     * data to the successive node and do not partecipate in the general
     * collective call operation.
     * All the nodes that do not stop their computation will receive an alias
     * called new_node, used to calculate their correct destination wrt this
     * new "cut" topology.
     */
    new_rank = rank;
    loop_flag = 0;
    if (rank < (2 * extra_ranks)) {
        if (0 == (rank % 2)) {
            mpi_errno = MPIC_Send(sendbuf, count, datatype, (rank + 1), MPIR_ALLREDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            loop_flag = 1;
        } else {
            /* TODO: Pay attention to commuitativity of the operation */
            mpi_errno = MPIC_Recv(recvbuf, count, datatype, (rank - 1), MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
            MPIR_Reduce_local((char *)sendbuf, (char *)recvbuf, count, datatype, op);
            new_rank = rank >> 1;
        }
    } else {
        new_rank = rank - extra_ranks;
        /* Copy into receive_buffer content of send_buffer to not produce
         * side effects on send_buffer
         */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    /* Here the actual allreduce starts */
    MPIR_CHKLMEM_MALLOC(r_index, sizeof(*r_index) * steps);
    MPIR_CHKLMEM_MALLOC(s_index, sizeof(*s_index) * steps);
    MPIR_CHKLMEM_MALLOC(r_count, sizeof(*r_count) * steps);
    MPIR_CHKLMEM_MALLOC(s_count, sizeof(*s_count) * steps);

    /* Only the remaining ranks will do the following part */
    if (!loop_flag) {
        /* Reduce-Scatter phase */
        w_size = count;
        s_index[0] = r_index[0] = 0;
        vrank = MPII_Bine_remap_rank((uint32_t)adjsize, (uint32_t)new_rank);

        for (step = 0; step < steps; step++) {
            vdest = MPII_Bine_pi(new_rank, step, adjsize);

            dest = is_power_of_two         ? vdest
                   : (vdest < extra_ranks) ? (vdest << 1) + 1
                                           : vdest + extra_ranks;

            /* TODO: dest or vdest as param? */
            vdest = MPII_Bine_remap_rank((uint32_t)adjsize, (uint32_t)vdest);

            if (vrank < vdest) {
                r_count[step] = w_size / 2;
                s_count[step] = w_size - r_count[step];
                s_index[step] = r_index[step] + r_count[step];
            } else {
                s_count[step] = w_size / 2;
                r_count[step] = w_size - s_count[step];
                r_index[step] = s_index[step] + s_count[step];
            }

            num_phases = (r_count[step] > s_count[step])
                             ? (int)(r_count[step] / segcount)
                             : (int)(s_count[step] / segcount);

            phase_scount =
                (s_count[step] > segcount) ? segcount : s_count[step];
            phase_rcount =
                (r_count[step] > segcount) ? segcount : r_count[step];

            inbi = 0;
            mpi_errno = MPIC_Irecv(inbuf[inbi], phase_rcount, datatype, dest, MPIR_ALLREDUCE_TAG, comm_ptr,
                                   &reqs[inbi]);
            MPIR_ERR_CHECK(mpi_errno);

            tmp_send = (char *)recvbuf + s_index[step] * extent;
            mpi_errno = MPIC_Send(tmp_send, phase_scount, datatype, dest, MPIR_ALLREDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);

            tmp_recv = (char *)recvbuf + r_index[step] * extent;

            for (phase = 0; phase < num_phases - 1; phase++) {
                tmp_recv_phase =
                    tmp_recv + (MPI_Aint)(phase * phase_rcount * extent);
                tmp_send_phase =
                    tmp_send + (MPI_Aint)((phase + 1) * phase_scount * extent);
                inbi = inbi ^ 0x1;

                mpi_errno = MPIC_Irecv(inbuf[inbi], phase_rcount, datatype, dest, MPIR_ALLREDUCE_TAG, comm_ptr,
                                       &reqs[inbi]);
                MPIR_ERR_CHECK(mpi_errno);

                mpi_errno = MPIC_Wait(reqs[inbi ^ 0x1]);
                MPIR_ERR_CHECK(mpi_errno);
                MPIR_Request_free(reqs[inbi ^ 0x1]);

                mpi_errno = MPIR_Reduce_local(inbuf[inbi ^ 0x1], tmp_recv_phase,
                                              phase_rcount, datatype, op);
                MPIR_ERR_CHECK(mpi_errno);

                mpi_errno = MPIC_Send(tmp_send_phase, phase_scount, datatype, dest, MPIR_ALLREDUCE_TAG,
                                      comm_ptr, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }

            mpi_errno = MPIC_Wait(reqs[inbi]);
            MPIR_ERR_CHECK(mpi_errno);
            MPIR_Request_free(reqs[inbi]);

            if (num_phases != 0) {
                tmp_recv +=
                    (MPI_Aint)((num_phases - 1) * phase_rcount * extent);
            }
            mpi_errno = MPIR_Reduce_local(inbuf[inbi], tmp_recv, phase_rcount, datatype,
                                          op);
            MPIR_ERR_CHECK(mpi_errno);

            if (step + 1 < steps) {
                r_index[step + 1] = r_index[step];
                s_index[step + 1] = r_index[step];
                w_size = r_count[step];
            }
        }

        /* Allgather phase */
        for (step = steps - 1; step >= 0; step--) {
            vdest = MPII_Bine_pi(new_rank, step, adjsize);

            dest = is_power_of_two         ? vdest
                   : (vdest < extra_ranks) ? (vdest << 1) + 1
                                           : vdest + extra_ranks;

            tmp_send = (char *) recvbuf + r_index[step] * extent;
            tmp_recv = (char *) recvbuf + s_index[step] * extent;
            mpi_errno = MPIC_Sendrecv(tmp_send, r_count[step], datatype, dest, MPIR_ALLREDUCE_TAG,
                                      tmp_recv, s_count[step], datatype, dest, MPIR_ALLREDUCE_TAG, comm_ptr,
                                      MPI_STATUS_IGNORE, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    /* Final results is sent to nodes that are not included in general
     * computation (general computation loop requires 2^n nodes).
     */
    if (rank < (2 * extra_ranks)) {
        if (!loop_flag) {
            mpi_errno = MPIC_Send(recvbuf, count, datatype, (rank - 1), MPIR_ALLREDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            mpi_errno = MPIC_Recv(recvbuf, count, datatype, (rank + 1), MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    if (reqs[0] != NULL) {
        MPIR_Request_free(reqs[0]);
    }
    if (reqs[1] != NULL) {
        MPIR_Request_free(reqs[1]);
    }
    goto fn_exit;
}