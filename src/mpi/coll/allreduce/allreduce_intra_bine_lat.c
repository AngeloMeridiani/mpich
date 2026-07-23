/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_bine.h"

int MPIR_Allreduce_intra_bine_lat(const void *sendbuf,
                                  void *recvbuf,
                                  MPI_Aint count,
                                  MPI_Datatype datatype,
                                  MPI_Op op, MPIR_Comm* comm_ptr, int coll_attr) 
{
    int rank, comm_size, mpi_errno = MPI_SUCCESS;
    int ret, line; /* for error handling */
    int steps, adjsize, extra_ranks, is_power_of_two;
    int new_rank, loop_flag = 0;
    int s, vdest, dest;
    char *tmpsend = NULL, *tmprecv = NULL, *tmpbuf = NULL;
    MPI_Aint extent, true_extent, lb, span = 0;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /* Special case for comm_size == 1 */
    if (comm_size == 1) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        }
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Allocate and initialize temporary send buffer */
    MPIR_Datatype_get_extent_macro(datatype, extent);
    MPIR_Type_get_true_extent_impl(datatype, &lb, &true_extent);
    MPIR_CHKLMEM_MALLOC(tmpbuf, count * (MPL_MAX(extent, true_extent)));

    /* Copy content from send buffer to tmpbuf */
    if (MPI_IN_PLACE == sendbuf) {
        mpi_errno = MPIR_Localcopy(recvbuf, count, datatype, tmpbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, tmpbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    tmpsend = tmpbuf;
    tmprecv = (char *)recvbuf;

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
        if ((rank % 2) == 0) {
            mpi_errno = MPIC_Send(tmpsend, count, datatype, (rank + 1), MPIR_ALLREDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            loop_flag = 1;
        } else {
            mpi_errno = MPIC_Recv(tmprecv, count, datatype, (rank - 1), MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);

            mpi_errno = MPIR_Reduce_local((char *)tmprecv, (char *)tmpsend, count, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);
            new_rank = rank >> 1;
        }
    } else
        new_rank = rank - extra_ranks;

    /* Actual allreduce computation for general cases */
    for (s = 0; s < steps; s++) {
        if (loop_flag)
            break;
        vdest = MPII_Bine_pi(new_rank, s, adjsize);

        dest = is_power_of_two         ? vdest
               : (vdest < extra_ranks) ? (vdest << 1) + 1
                                       : vdest + extra_ranks;

        mpi_errno = MPIC_Sendrecv(tmpsend, count, datatype, dest, MPIR_ALLREDUCE_TAG, tmprecv, count,
                                  datatype, dest, MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

        mpi_errno = MPIR_Reduce_local((char *)tmprecv, (char *)tmpsend, count, datatype, op);
        MPIR_ERR_CHECK(mpi_errno);

    }

    /* Final results is sent to nodes that are not included in general
     * computation (general computation loop requires 2^n nodes).
     */
    if (rank < (2 * extra_ranks)) {
        if (!loop_flag) {
            mpi_errno = MPIC_Send(tmpsend, count, datatype, (rank - 1), MPIR_ALLREDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            mpi_errno = MPIC_Recv(recvbuf, count, datatype, (rank + 1), MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
            tmpsend = (char *)recvbuf;
        }
    }

    if (tmpsend != recvbuf) {
        mpi_errno = MPIR_Localcopy(tmpsend, count, datatype, recvbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}