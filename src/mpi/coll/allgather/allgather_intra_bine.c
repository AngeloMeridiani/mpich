#include "mpiimpl.h"
#include "mpir_bine.h"

static inline int MPIR_Allgather_intra_bine_permutation(const void *sendbuf, MPI_Aint sendcount,
                                                        MPI_Datatype sendtype, void *recvbuf,
                                                        MPI_Aint recvcount, MPI_Datatype recvtype,
                                                        MPIR_Comm *comm_ptr, int coll_attr)
{

    int rank, comm_size, steps, mpi_errno = MPI_SUCCESS, remote, data_exchange;
    int *permutation = NULL;
    MPI_Aint rext, rsize;
    char *tmprecv = NULL;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    steps = MPII_Bine_log2(comm_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");
    MPIR_ERR_CHKANDJUMP((steps < 1), mpi_errno, MPI_ERR_ARG, "**arg");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);

    if (MPI_IN_PLACE != sendbuf) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   recvbuf, recvcount * rsize, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        if (rank != 0) {
            mpi_errno = MPIR_Localcopy((char *) recvbuf + rank * recvcount * rext,
                                       recvcount, recvtype, recvbuf, recvcount * rsize,
                                       MPIR_BYTE_INTERNAL);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    MPIR_CHKLMEM_MALLOC(permutation, comm_size * sizeof(int));

    memset(permutation, -1, comm_size * sizeof(int));
    *(permutation + rank) = 0;

    data_exchange = 1;
    for (int step = steps - 1; step >= 0; step--) {
        remote = MPII_Bine_pi(rank, step, comm_size);

        MPII_Bine_get_permutation(rank, step, steps, comm_size, permutation, data_exchange);

        tmprecv = (char *) recvbuf + (MPI_Aint) data_exchange *(MPI_Aint) recvcount *rext;

        mpi_errno = MPIC_Sendrecv(recvbuf, data_exchange * recvcount, recvtype, remote,
                                  MPIR_ALLGATHER_TAG, tmprecv, data_exchange * recvcount, recvtype,
                                  remote, MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        data_exchange <<= 1;
    }

    mpi_errno = MPII_Bine_reorder_blocks_gpu(recvbuf, recvcount, recvtype, permutation, comm_size);
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
    int node_size, node_rank, node_offset, local_rank;
    int num_reqs;
    int *s_bitmap = NULL, *r_bitmap = NULL;
    MPI_Aint rext, rsize;
    char *tmpsend = NULL, *tmprecv = NULL;
    MPIR_Request **requests = NULL;
    int task_on_node;

    mpi_errno = MPII_Bine_pico_task_on_node(&task_on_node);
    MPIR_ERR_CHECK(mpi_errno);

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    MPII_Bine_pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank,
                                    task_on_node, comm_size, rank);

    steps = MPL_log2(node_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");
    MPIR_ERR_CHKANDJUMP((steps < 1 && node_size > 1), mpi_errno, MPI_ERR_ARG, "**arg");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);

    if (MPI_IN_PLACE != sendbuf) {
        tmpsend = (char *) sendbuf;
        tmprecv = (char *) recvbuf + (MPI_Aint) rank *(MPI_Aint) recvcount *rext;
        mpi_errno = MPIR_Localcopy(tmpsend, sendcount, sendtype,
                                   tmprecv, recvcount * rsize, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    }

    MPIR_CHKLMEM_MALLOC(s_bitmap, node_size * sizeof(int));
    MPIR_CHKLMEM_MALLOC(r_bitmap, node_size * sizeof(int));
    MPIR_CHKLMEM_MALLOC(requests, comm_size * 2 * sizeof(MPIR_Request *));

    for (int step = steps - 1; step >= 0; step--) {
        num_reqs = 0;
        remote = MPII_Bine_pi(node_rank, step, node_size);

        memset(s_bitmap, 0, node_size * sizeof(int));
        memset(r_bitmap, 0, node_size * sizeof(int));
        MPII_Bine_get_indexes(node_rank, step, steps, node_size, r_bitmap);
        MPII_Bine_get_indexes(remote, step, steps, node_size, s_bitmap);

        remote = remote * task_on_node + local_rank;

        for (int block = 0; block < node_size; block++) {
            if (s_bitmap[block] != 0) {
                tmpsend = (char *) recvbuf +
                    (MPI_Aint) (block * task_on_node + local_rank) * (MPI_Aint) recvcount *rext;
                mpi_errno = MPIC_Isend(tmpsend, recvcount, recvtype, remote,
                                       MPIR_ALLGATHER_TAG, comm_ptr,
                                       requests + num_reqs, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
                num_reqs++;
            }
            if (r_bitmap[block] != 0) {
                tmprecv = (char *) recvbuf +
                    (MPI_Aint) (block * task_on_node + local_rank) * (MPI_Aint) recvcount *rext;
                mpi_errno = MPIC_Irecv(tmprecv, recvcount, recvtype, remote,
                                       MPIR_ALLGATHER_TAG, comm_ptr, requests + num_reqs);
                MPIR_ERR_CHECK(mpi_errno);
                num_reqs++;
            }
        }
        mpi_errno = MPIC_Waitall(num_reqs, requests, MPI_STATUSES_IGNORE);
        MPIR_ERR_CHECK(mpi_errno);
    }

    // local exchange
    num_reqs = 0;
    for (int i = 0; i < task_on_node; i++) {
        if (i == local_rank)
            continue;

        for (int j = 0; j < node_size; j++) {
            tmpsend =
                (char *) recvbuf + (MPI_Aint) (j * task_on_node +
                                               local_rank) * (MPI_Aint) recvcount *rext;
            tmprecv = (char *) recvbuf + (MPI_Aint) (j * task_on_node + i) * recvcount * rext;

            mpi_errno = MPIC_Isend(tmpsend, recvcount, recvtype,
                                   node_offset + i, MPIR_ALLGATHER_TAG,
                                   comm_ptr, &requests[num_reqs], coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            num_reqs++;

            mpi_errno = MPIC_Irecv(tmprecv, recvcount, recvtype, node_offset + i,
                                   MPIR_ALLGATHER_TAG, comm_ptr, &requests[num_reqs]);
            MPIR_ERR_CHECK(mpi_errno);
            num_reqs++;
        }
    }
    mpi_errno = MPIC_Waitall(num_reqs, requests, MPI_STATUSES_IGNORE);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static inline int MPIR_Allgather_intra_bine_send(const void *sendbuf, MPI_Aint sendcount,
                                                 MPI_Datatype sendtype, void *recvbuf,
                                                 MPI_Aint recvcount, MPI_Datatype recvtype,
                                                 MPIR_Comm *comm_ptr, int coll_attr)
{

    int rank, comm_size, steps, mpi_errno = MPI_SUCCESS;
    int vrank, remote, vremote, send_block_location, distance;
    int node_size, node_rank, node_offset, local_rank;
    MPI_Aint rext, rsize;
    char *tmpsend = NULL, *tmprecv = NULL;
    void *perm_buff = NULL, *global_temp = NULL;
    MPIR_Request **requests = NULL;
    int task_on_node;

    mpi_errno = MPII_Bine_pico_task_on_node(&task_on_node);
    MPIR_ERR_CHECK(mpi_errno);


    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    MPII_Bine_pico_get_group_config(&node_size, &node_rank, &node_offset, &local_rank,
                                    task_on_node, comm_size, rank);

    MPIR_CHKLMEM_MALLOC(requests, task_on_node * 2 * sizeof(MPIR_Request *));

    /*
     * Current implementation only handles power-of-two number of processes.
     */
    steps = MPII_Bine_log2(node_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");
    MPIR_ERR_CHKANDJUMP((steps < 1 && node_size > 1), mpi_errno, MPI_ERR_ARG, "**arg");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);


#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaMalloc((void **) &perm_buff, comm_size * recvcount * rext));
#else
    MPIR_CHKLMEM_MALLOC(perm_buff, comm_size * recvcount * rext);
#endif


    /* Initialization step:
     * - if I gather the result for another rank, I send my buffer to that rank
     *   and I receive the data from the rank at the inverse permutation
     * - if I gather the result for myself, I copy the data from the send buffer
     */
    vrank = (int) MPII_Bine_remap_rank(node_size, node_rank);
    int node_to_rank = vrank * task_on_node + local_rank;

    /*if (MPI_IN_PLACE != sendbuf) {
     * tmpsend = (char *)sendbuf;
     * tmprecv = (char *)perm_buff + (MPI_Aint)(local_rank * node_size + vrank) * (MPI_Aint)recvcount * rext;
     * mpi_errno = MPIR_Localcopy(tmpsend, sendcount, sendtype,
     * tmprecv, recvcount*rsize, MPIR_BYTE_INTERNAL);
     * MPIR_ERR_CHECK(mpi_errno);
     * } */

    if (vrank != node_rank) {
        tmprecv = (char *) perm_buff + (MPI_Aint) (local_rank * node_size + vrank) *
            (MPI_Aint) recvcount *rext;
        if (MPI_IN_PLACE != sendbuf) {
            tmpsend = (char *) sendbuf;
        } else {
            tmpsend =
                (char *) perm_buff + (MPI_Aint) (local_rank * node_size +
                                                 vrank) * (MPI_Aint) recvcount *rext;
        }
        mpi_errno = MPIC_Sendrecv(tmpsend, sendcount, sendtype,
                                  MPII_Bine_get_sender_rec(node_size,
                                                           node_rank) * task_on_node + local_rank,
                                  MPIR_ALLGATHER_TAG, tmprecv, recvcount, recvtype, node_to_rank,
                                  MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        tmpsend = (char *) sendbuf;
        tmprecv =
            (char *) perm_buff + (MPI_Aint) (local_rank * node_size +
                                             vrank) * (MPI_Aint) recvcount *rext;
        mpi_errno =
            MPIR_Localcopy(tmpsend, sendcount, sendtype, tmprecv, recvcount * rsize,
                           MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Communication step:
     * At every step i, rank r:
     * - exchanges message with rank remote = (r ^ 2^i).
     */
    distance = 0x1;
    send_block_location = vrank;
    global_temp = (char *) perm_buff + (MPI_Aint) local_rank *
        (MPI_Aint) node_size *(MPI_Aint) recvcount *rext;
    for (int step = steps - 1; step >= 0; step--) {
        size_t step_scount = recvcount * distance;
        remote = MPII_Bine_pi(node_rank, step, node_size);
        vremote = (int) MPII_Bine_remap_rank(node_size, remote);
        node_to_rank = remote * task_on_node + local_rank;

        if (vrank < vremote) {
            tmpsend = (char *) global_temp +
                (MPI_Aint) send_block_location *(MPI_Aint) recvcount *rext;
            tmprecv = (char *) global_temp +
                (MPI_Aint) (send_block_location + distance) * (MPI_Aint) recvcount *rext;
        } else {
            tmpsend = (char *) global_temp +
                (MPI_Aint) send_block_location *(MPI_Aint) recvcount *rext;
            tmprecv = (char *) global_temp +
                (MPI_Aint) (send_block_location - distance) * (MPI_Aint) recvcount *rext;
            send_block_location -= distance;
        }

        /* Sendreceive */
        mpi_errno = MPIC_Sendrecv(tmpsend, step_scount, recvtype, node_to_rank, MPIR_ALLGATHER_TAG,
                                  tmprecv, step_scount, recvtype, node_to_rank, MPIR_ALLGATHER_TAG,
                                  comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        distance <<= 1;
    }

    // local exchange
    int num_reqs = 0;
    tmpsend = global_temp;
    for (int i = 0; i < task_on_node; i++) {
        if (i == local_rank)
            continue;

        tmprecv = (char *) perm_buff + (MPI_Aint) i *(MPI_Aint) node_size *recvcount * rext;

        mpi_errno =
            MPIC_Isend(tmpsend, (MPI_Aint) node_size * recvcount, recvtype, node_offset + i,
                       MPIR_ALLGATHER_TAG, comm_ptr, &requests[num_reqs], coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        num_reqs++;

        mpi_errno =
            MPIC_Irecv(tmprecv, (MPI_Aint) node_size * recvcount, recvtype, node_offset + i,
                       MPIR_ALLGATHER_TAG, comm_ptr, &requests[num_reqs]);
        MPIR_ERR_CHECK(mpi_errno);
        num_reqs++;
    }
    mpi_errno = MPIC_Waitall(num_reqs, requests, MPI_STATUSES_IGNORE);
    MPIR_ERR_CHECK(mpi_errno);

#ifdef PICO_MPI_CUDA_AWARE
    reorder_kernel_wrapper(perm_buff, recvbuf, recvcount, comm_size, task_on_node, recvtype);
    BINE_CUDA_CHECK(cudaDeviceSynchronize());
#else
    for (int i = 0; i < comm_size; i++) {
        int elem_local_rank = i / node_size;
        int elem_node_rank = i % node_size;
        mpi_errno = MPIR_Localcopy(perm_buff + i * recvcount * rext, recvcount, recvtype,
                                   recvbuf +
                                   ((elem_node_rank * task_on_node +
                                     elem_local_rank) * recvcount) * rext, recvcount, recvtype);
        MPIR_ERR_CHECK(mpi_errno);
    }
#endif

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
#ifdef PICO_MPI_CUDA_AWARE
    BINE_CUDA_CHECK(cudaFree(perm_buff));
#endif
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static inline int MPIR_Allgather_intra_bine_2_blocks(const void *sendbuf, MPI_Aint sendcount,
                                                     MPI_Datatype sendtype, void *recvbuf,
                                                     MPI_Aint recvcount, MPI_Datatype recvtype,
                                                     MPIR_Comm *comm_ptr, int coll_attr)
{

    int rank, comm_size, steps, mpi_errno = MPI_SUCCESS, remote;
    int mask, my_first, recv_index, send_index;
    int send_count, recv_count, extra_send, extra_recv, extra_tag;
    MPI_Aint rext, rsize;
    char *tmpsend = NULL, *tmprecv = NULL;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /*
     * Current implementation only handles power-of-two number of processes.
     */
    steps = MPL_log2(comm_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");
    MPIR_ERR_CHKANDJUMP(steps < 1, mpi_errno, MPI_ERR_ARG, "**arg");

    MPIR_Datatype_get_extent_macro(recvtype, rext);
    MPIR_Datatype_get_size_macro(recvtype, rsize);

    /* Initialization step:
     * - if send buffer is not MPI_IN_PLACE, copy send buffer to block  of
     * receive buffer
     */

    if (MPI_IN_PLACE != sendbuf) {
        tmpsend = (char *) sendbuf;
        tmprecv = (char *) recvbuf + (MPI_Aint) rank *(MPI_Aint) recvcount *rext;
        mpi_errno = MPIR_Localcopy(tmpsend, sendcount, sendtype,
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
    extra_tag = 1;
    for (int step = 0; step < steps; step++) {
        MPIR_Request *req;
        remote = MPII_Bine_pi(rank, step, comm_size);
        send_index = my_first;

        // Calculate the send and receive indexes by alternating send/recv
        // direction.
        if ((step & 1) == (rank & 1)) {
            recv_index = (send_index + mask + comm_size) % comm_size;
        } else {
            recv_index = (send_index - mask + comm_size) % comm_size;
            my_first = recv_index;
        }

        // Control if the previously calculated indexes imply out of bound
        // send/recv. If so, split the communication with an extra send/recv.
        extra_recv = (recv_index + mask > comm_size) ? ((recv_index + mask) - comm_size) : 0;
        recv_count = mask - extra_recv;

        extra_send = (send_index + mask > comm_size) ? ((send_index + mask) - comm_size) : 0;
        send_count = mask - extra_send;

        // warparound communication
        if (extra_recv != 0) {
            tmprecv = (char *) recvbuf;
            mpi_errno = MPIC_Irecv(tmprecv, (MPI_Aint) extra_recv * recvcount, recvtype, remote,
                                   extra_tag, comm_ptr, &req);
            MPIR_ERR_CHECK(mpi_errno);
        }
        if (extra_send != 0) {
            tmpsend = (char *) recvbuf;
            mpi_errno = MPIC_Send(tmpsend, (MPI_Aint) extra_send * recvcount, recvtype, remote,
                                  extra_tag, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
        }
        // Simple case: no wrap-around
        tmpsend = (char *) recvbuf + (MPI_Aint) send_index *(MPI_Aint) recvcount *rext;
        tmprecv = (char *) recvbuf + (MPI_Aint) recv_index *(MPI_Aint) recvcount *rext;

        mpi_errno = MPIC_Sendrecv(tmpsend, send_count * recvcount, recvtype, remote, 0,
                                  tmprecv, recv_count * recvcount, recvtype, remote, 0,
                                  comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

        if (extra_recv != 0) {
            mpi_errno = MPIC_Wait(req);
            MPIR_ERR_CHECK(mpi_errno);
        }

        mask <<= 1;
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Allgather_intra_bine(const void *sendbuf, MPI_Aint sendcount,
                              MPI_Datatype sendtype, void *recvbuf,
                              MPI_Aint recvcount, MPI_Datatype recvtype,
                              MPIR_Comm *comm_ptr, int coll_attr)
{

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        goto fn_exit;

    int rank, comm_size, mpi_errno = MPI_SUCCESS;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /*MPIR_Allgather_intra_bine_permutation(sendbuf, sendcount, sendtype,
     * recvbuf, recvcount, recvtype,
     * comm_ptr, coll_attr); */

    mpi_errno = MPIR_Allgather_intra_bine_block_by_block(sendbuf, sendcount, sendtype,
                                                         recvbuf, recvcount, recvtype,
                                                         comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    /*mpi_errno = MPIR_Allgather_intra_bine_send(sendbuf, sendcount, sendtype,
     * recvbuf, recvcount, recvtype,
     * comm_ptr, coll_attr);
     * MPIR_ERR_CHECK(mpi_errno); */

    /*mpi_errno = MPIR_Allgather_intra_bine_2_blocks(sendbuf, sendcount, sendtype,
     * recvbuf, recvcount, recvtype,
     * comm_ptr, coll_attr);
     * MPIR_ERR_CHECK(mpi_errno); */


  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    if (rank == 0)
        printf("Allgather bine eseguito con successo \n");
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
