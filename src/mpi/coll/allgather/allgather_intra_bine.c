/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_bine.h"

static inline int MPIR_Allgather_intra_bine_permute(const void *sendbuf, MPI_Aint sendcount,
                                                    MPI_Datatype sendtype, void *recvbuf,
                                                    MPI_Aint recvcount, MPI_Datatype recvtype,
                                                    MPIR_Comm *comm_ptr, int coll_attr)
{

    int rank, comm_size, steps, mpi_errno = MPI_SUCCESS, remote, data_exchange;
    int *permutation = NULL;
    MPI_Aint rext, rsize;
    char *tmprecv = NULL;
    void *tmp_buf = NULL;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /*
     * Current implementation only handles power-of-two number of processes.
     */
    steps = MPII_Bine_log2(comm_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);

    MPIR_CHKLMEM_MALLOC(tmp_buf, comm_size * rsize * recvcount);

    if (MPI_IN_PLACE != sendbuf) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   tmp_buf, recvcount * rsize, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        mpi_errno = MPIR_Localcopy((char *) recvbuf + rank * recvcount * rext,
                                   recvcount, recvtype, tmp_buf, recvcount * rsize,
                                   MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    }

    MPIR_CHKLMEM_MALLOC(permutation, comm_size * sizeof(int));

    memset(permutation, -1, comm_size * sizeof(int));
    *(permutation + rank) = 0;

    data_exchange = 1;
    for (int step = steps - 1; step >= 0; step--) {
        remote = MPII_Bine_pi(rank, step, comm_size);

        MPII_Bine_get_permutation(rank, step, steps, comm_size, permutation, data_exchange);

        tmprecv = (char *) tmp_buf + (MPI_Aint) data_exchange *(MPI_Aint) recvcount *rsize;

        mpi_errno =
            MPIC_Sendrecv(tmp_buf, data_exchange * recvcount * rsize, MPIR_BYTE_INTERNAL, remote,
                          MPIR_ALLGATHER_TAG, tmprecv, data_exchange * recvcount * rsize,
                          MPIR_BYTE_INTERNAL, remote, MPIR_ALLGATHER_TAG, comm_ptr,
                          MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        data_exchange <<= 1;
    }

    mpi_errno =
        MPII_Bine_reorder_blocks(tmp_buf, recvcount * rsize, MPIR_BYTE_INTERNAL, permutation,
                                 comm_size);
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = MPIR_Localcopy(tmp_buf, comm_size * recvcount * rsize, MPIR_BYTE_INTERNAL,
                               recvbuf, comm_size * recvcount, recvtype);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static inline int MPIR_Allgather_intra_bine_block_by_block(const void *sendbuf, MPI_Aint sendcount,
                                                           MPI_Datatype sendtype, void *recvbuf,
                                                           MPI_Aint recvcount,
                                                           MPI_Datatype recvtype,
                                                           MPIR_Comm *comm_ptr, int coll_attr)
{
    int rank, comm_size, steps, mpi_errno = MPI_SUCCESS, remote;
    int *s_bitmap = NULL, *r_bitmap = NULL;
    MPI_Aint rext, rsize;
    char *tmpsend = NULL, *tmprecv = NULL;
    void *tmp_buf = NULL;
    MPIR_Request **requests = NULL;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /*
     * Current implementation only handles power-of-two number of processes.
     */
    steps = MPII_Bine_log2(comm_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);

    MPIR_CHKLMEM_MALLOC(tmp_buf, recvcount * comm_size * rsize);

    if (MPI_IN_PLACE != sendbuf) {
        tmpsend = (char *) sendbuf;
        tmprecv = (char *) tmp_buf + (MPI_Aint) rank *(MPI_Aint) recvcount *rsize;
        mpi_errno = MPIR_Localcopy(tmpsend, sendcount, sendtype,
                                   tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        tmpsend = (char *) recvbuf + (MPI_Aint) rank *(MPI_Aint) recvcount *rext;
        tmprecv = (char *) tmp_buf + (MPI_Aint) rank *(MPI_Aint) recvcount *rsize;
        mpi_errno = MPIR_Localcopy(tmpsend, recvcount, recvtype,
                                   tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    }

    MPIR_CHKLMEM_MALLOC(s_bitmap, comm_size * sizeof(int));
    MPIR_CHKLMEM_MALLOC(r_bitmap, comm_size * sizeof(int));
    MPIR_CHKLMEM_MALLOC(requests, comm_size * sizeof(MPIR_Request *));

    for (int step = steps - 1; step >= 0; step--) {
        int num_reqs = 0;
        remote = MPII_Bine_pi(rank, step, comm_size);

        memset(s_bitmap, 0, comm_size * sizeof(int));
        memset(r_bitmap, 0, comm_size * sizeof(int));
        MPII_Bine_get_indexes(rank, step, steps, comm_size, r_bitmap);
        MPII_Bine_get_indexes(remote, step, steps, comm_size, s_bitmap);

        for (int block = 0; block < comm_size; block++) {
            if (s_bitmap[block] != 0) {
                tmpsend = (char *) tmp_buf + (MPI_Aint) block *(MPI_Aint) recvcount *rsize;
                mpi_errno = MPIC_Isend(tmpsend, recvcount * rsize, MPIR_BYTE_INTERNAL, remote,
                                       MPIR_ALLGATHER_TAG, comm_ptr,
                                       requests + num_reqs, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
                num_reqs++;
            }
            if (r_bitmap[block] != 0) {
                tmprecv = (char *) tmp_buf + (MPI_Aint) block *(MPI_Aint) recvcount *rsize;
                mpi_errno = MPIC_Irecv(tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL, remote,
                                       MPIR_ALLGATHER_TAG, comm_ptr, requests + num_reqs);
                MPIR_ERR_CHECK(mpi_errno);
                num_reqs++;
            }
        }
        mpi_errno = MPIC_Waitall(num_reqs, requests, MPI_STATUSES_IGNORE);
        MPIR_ERR_CHECK(mpi_errno);
    }

    mpi_errno = MPIR_Localcopy(tmp_buf, comm_size * recvcount * rsize, MPIR_BYTE_INTERNAL,
                               recvbuf, comm_size * recvcount, recvtype);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static inline int MPIR_Allgather_intra_bine_send_remap(const void *sendbuf, MPI_Aint sendcount,
                                                       MPI_Datatype sendtype, void *recvbuf,
                                                       MPI_Aint recvcount, MPI_Datatype recvtype,
                                                       MPIR_Comm *comm_ptr, int coll_attr)
{
    int rank, comm_size, steps, mpi_errno = MPI_SUCCESS;
    int vrank, remote, vremote, send_block_location, distance;
    MPI_Aint rext, rsize;
    MPI_Aint step_scount;
    char *tmpsend = NULL, *tmprecv = NULL;
    void *tmp_buf = NULL;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /*
     * Current implementation only handles power-of-two number of processes.
     */
    steps = MPII_Bine_log2(comm_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);

    MPIR_CHKLMEM_MALLOC(tmp_buf, recvcount * comm_size * rsize);

    /* Initialization step:
     * - if I gather the result for another rank, I send my buffer to that rank
     *   and I receive the data from the rank at the inverse permutation
     * - if I gather the result for myself, I copy the data from the send buffer
     */
    vrank = (int) MPII_Bine_remap_rank(comm_size, rank);
    tmprecv = (char *) tmp_buf + (MPI_Aint) vrank * (MPI_Aint) recvcount * rsize;
    if (vrank != rank) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno =
                MPIC_Sendrecv(sendbuf, sendcount, sendtype,
                              MPII_Bine_get_sender_rec(comm_size, rank), MPIR_ALLGATHER_TAG,
                              tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL, vrank,
                              MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        } else {
            tmpsend = (char *) recvbuf + (MPI_Aint) rank *(MPI_Aint) recvcount *rext;
            mpi_errno =
                MPIC_Sendrecv(tmpsend, recvcount, recvtype,
                              MPII_Bine_get_sender_rec(comm_size, rank), MPIR_ALLGATHER_TAG,
                              tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL, vrank,
                              MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        }
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                       tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL);
        } else {
            tmpsend = (char *) recvbuf + (MPI_Aint) vrank *(MPI_Aint) recvcount *rext;
            mpi_errno = MPIR_Localcopy(tmpsend, recvcount, recvtype,
                                       tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL);
        }
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Communication step:
     * At every step i, rank r:
     * - exchanges message with rank remote = (r ^ 2^i).
     */
    distance = 0x1;
    send_block_location = vrank;
    for (int step = steps - 1; step >= 0; step--) {
        step_scount = recvcount * distance;
        remote = MPII_Bine_pi(rank, step, comm_size);
        vremote = (int) MPII_Bine_remap_rank(comm_size, remote);

        if (vrank < vremote) {
            tmpsend = (char *) tmp_buf + (MPI_Aint) send_block_location *
                (MPI_Aint) recvcount *rsize;
            tmprecv = (char *) tmp_buf + (MPI_Aint) (send_block_location + distance) *
                (MPI_Aint) recvcount *rsize;
        } else {
            tmpsend = (char *) tmp_buf +
                (MPI_Aint) send_block_location *(MPI_Aint) recvcount *rsize;
            tmprecv = (char *) tmp_buf + (MPI_Aint) (send_block_location - distance) *
                (MPI_Aint) recvcount *rsize;
            send_block_location -= distance;
        }

        /* Sendreceive */
        mpi_errno =
            MPIC_Sendrecv(tmpsend, step_scount * rsize, MPIR_BYTE_INTERNAL, remote,
                          MPIR_ALLGATHER_TAG, tmprecv, step_scount * rsize, MPIR_BYTE_INTERNAL,
                          remote, MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

        distance <<= 1;
    }

    mpi_errno = MPIR_Localcopy(tmp_buf, comm_size * recvcount * rsize, MPIR_BYTE_INTERNAL,
                               recvbuf, comm_size * recvcount, recvtype);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static inline int MPIR_Allgather_intra_bine_two_blocks(const void *sendbuf, MPI_Aint sendcount,
                                                     MPI_Datatype sendtype, void *recvbuf,
                                                     MPI_Aint recvcount, MPI_Datatype recvtype,
                                                     MPIR_Comm *comm_ptr, int coll_attr)
{

    int rank, comm_size, steps, mpi_errno = MPI_SUCCESS, remote;
    int mask, my_first, recv_index, send_index;
    int send_count, recv_count, extra_send, extra_recv, extra_tag;
    MPI_Aint rext, rsize;
    MPIR_Request *req;
    char *tmpsend = NULL, *tmprecv = NULL;
    void *tmp_buf = NULL;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /*
     * Current implementation only handles power-of-two number of processes.
     */
    steps = MPII_Bine_log2(comm_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);

    MPIR_CHKLMEM_MALLOC(tmp_buf, comm_size * rsize * recvcount);

    /* Initialization step:
     * - if send buffer is not MPI_IN_PLACE, copy send buffer to block  of
     * receive buffer
     */

    if (MPI_IN_PLACE != sendbuf) {
        tmpsend = (char *) sendbuf;
        tmprecv = (char *) tmp_buf + (MPI_Aint) rank *(MPI_Aint) recvcount *rsize;
        mpi_errno = MPIR_Localcopy(tmpsend, sendcount, sendtype,
                                   tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        tmpsend = (char *) recvbuf + (MPI_Aint) rank *(MPI_Aint) recvcount *rext;
        tmprecv = (char *) tmp_buf + (MPI_Aint) rank *(MPI_Aint) recvcount *rsize;
        mpi_errno = MPIR_Localcopy(tmpsend, recvcount, recvtype,
                                   tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Communication step.
     *  At every step i, rank r:
     *  - communication peer is calculated by pi(rank, step, comm_size)
     *  - if the step is even, even ranks send the next `mask` blocks and
     *  odd ranks send the previous `mask` blocks.
     *  - if the step is odd, even ranks send the previous `mask` blocks and
     *  odd ranks send the next `mask` blocks.
     */
    mask = 0x1;
    my_first = rank;
    for (int step = 0; step < steps; step++) {
        req = NULL;
        remote = MPII_Bine_pi(rank, step, comm_size);
        send_index = my_first;

        /* Calculate the send and receive indexes by alternating send/recv
         * direction.
         */
        if ((step & 1) == (rank & 1)) {
            recv_index = (send_index + mask + comm_size) % comm_size;
        } else {
            recv_index = (send_index - mask + comm_size) % comm_size;
            my_first = recv_index;
        }

        /* Control if the previously calculated indexes imply out of bound
         * send/recv. If so, split the communication with an extra send/recv.
         */
        extra_recv = (recv_index + mask > comm_size) ? ((recv_index + mask) - comm_size) : 0;
        recv_count = mask - extra_recv;

        extra_send = (send_index + mask > comm_size) ? ((send_index + mask) - comm_size) : 0;
        send_count = mask - extra_send;

        /* warparound communication */
        if (extra_recv != 0) {
            tmprecv = (char *) tmp_buf;
            mpi_errno =
                MPIC_Irecv(tmprecv, (MPI_Aint) extra_recv * recvcount * rsize, MPIR_BYTE_INTERNAL,
                           remote, MPIR_ALLGATHER_TAG, comm_ptr, &req);
            MPIR_ERR_CHECK(mpi_errno);
        }
        if (extra_send != 0) {
            tmpsend = (char *) tmp_buf;
            mpi_errno =
                MPIC_Send(tmpsend, (MPI_Aint) extra_send * recvcount * rsize, MPIR_BYTE_INTERNAL,
                          remote, MPIR_ALLGATHER_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
        }
        /* Simple case: no wrap-around */
        tmpsend = (char *) tmp_buf + (MPI_Aint) send_index *(MPI_Aint) recvcount *rsize;
        tmprecv = (char *) tmp_buf + (MPI_Aint) recv_index *(MPI_Aint) recvcount *rsize;

        mpi_errno =
            MPIC_Sendrecv(tmpsend, (MPI_Aint) send_count * recvcount * rsize, MPIR_BYTE_INTERNAL,
                          remote, MPIR_ALLGATHER_TAG, tmprecv,
                          (MPI_Aint) recv_count * recvcount * rsize, MPIR_BYTE_INTERNAL, remote,
                          MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

        if (extra_recv != 0) {
            mpi_errno = MPIC_Wait(req);
            MPIR_ERR_CHECK(mpi_errno);
            MPIR_Request_free(req);
        }
        mask <<= 1;
    }

    mpi_errno = MPIR_Localcopy(tmp_buf, comm_size * recvcount * rsize, MPIR_BYTE_INTERNAL,
                               recvbuf, comm_size * recvcount, recvtype);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Allgather_intra_bine(const void *sendbuf, MPI_Aint sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              MPI_Aint recvcount, MPI_Datatype recvtype,
                              MPIR_Comm *comm_ptr, int bine_type, int coll_attr)
{

    int mpi_errno = MPI_SUCCESS;

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        goto fn_exit;

    /* Here we use the CVAR MPIR_CVAR_ALLGATHER_BINE_TYPE to select the
     * correct algorithm. If an invalid value is given, then the algorithm
     * MPIR_Allgather_intra_bine_permute is used by default.
     */
    switch (bine_type) {
        case MPIR_BINE_TYPE_PERMUTE:
            mpi_errno = MPIR_Allgather_intra_bine_permute(sendbuf, sendcount, sendtype,
                                                          recvbuf, recvcount, recvtype,
                                                          comm_ptr, coll_attr);
            break;
        case MPIR_BINE_TYPE_SEND_REMAP:
            mpi_errno = MPIR_Allgather_intra_bine_send_remap(sendbuf, sendcount, sendtype,
                                                             recvbuf, recvcount, recvtype,
                                                             comm_ptr, coll_attr);
            break;
        case MPIR_BINE_TYPE_BLOCK_BY_BLOCK:
            mpi_errno = MPIR_Allgather_intra_bine_block_by_block(sendbuf, sendcount, sendtype,
                                                                 recvbuf, recvcount, recvtype,
                                                                 comm_ptr, coll_attr);
            break;
        case MPIR_BINE_TYPE_TWO_BLOCKS:
            mpi_errno = MPIR_Allgather_intra_bine_two_blocks(sendbuf, sendcount, sendtype,
                                                           recvbuf, recvcount, recvtype,
                                                           comm_ptr, coll_attr);
            break;
        default:
            mpi_errno = MPIR_Allgather_intra_bine_permute(sendbuf, sendcount, sendtype,
                                                          recvbuf, recvcount, recvtype,
                                                          comm_ptr, coll_attr);
            break;

    }
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
