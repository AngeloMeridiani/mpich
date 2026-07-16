/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_bine.h"

int MPIR_Reduce_intra_bine_bdw(const void *sendbuf,
                               void *recvbuf,
                               MPI_Aint count,
                               MPI_Datatype datatype,
                               MPI_Op op, int root, MPIR_Comm * comm_ptr, int coll_attr)
{

    int comm_size, rank, vrank, dtsize, mpi_errno = MPI_SUCCESS, steps, step;
    int count_per_rank, rem, mask = 0x1, inverse_mask;
    int block_first_mask, remapped_rank, receiving_mask;
    int *rindex = NULL, *sindex = NULL, *rcount = NULL, *scount = NULL;
    char *resbuf = NULL, *tmpbuf = NULL;
    MPI_Aint true_lb, true_extent, extent;
    MPI_Aint buf_size;
    MPIR_CHKLMEM_DECL();

    if (count == 0) {
        goto fn_exit;
    } 

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    if (comm_size == 1) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
        }
        goto fn_exit;
    }

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

    steps = MPL_log2(comm_size);
    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    vrank = MPII_Bine_mod_pof2(rank - root, comm_size);
    /* TODO: It would be more efficient if bytes are divided instead 
       of count to balance work */
    count_per_rank = count / comm_size;
    rem = count % comm_size;

    buf_size = count * (MPL_MAX(extent, true_extent));
    MPIR_CHKLMEM_MALLOC(tmpbuf, buf_size);
    tmpbuf = (void *) ((char *) tmpbuf - true_lb);

    if (rank == root) {
        resbuf = recvbuf;
    } else {
        MPIR_CHKLMEM_MALLOC(resbuf, buf_size);
        resbuf = (void *) ((char *) resbuf - true_lb);
    }

    if ((rank != root) || (sendbuf != MPI_IN_PLACE)) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, resbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    mask = 0x1;
    inverse_mask = 0x1 << (int)(MPL_log2(comm_size) - 1);
    block_first_mask = ~(inverse_mask - 1);
    remapped_rank = MPII_Bine_remap_rank(comm_size, vrank);

    /***** Reduce_scatter *****/
    MPIR_CHKLMEM_MALLOC(rindex, sizeof(int) * steps);
    MPIR_CHKLMEM_MALLOC(sindex, sizeof(int) * steps);
    MPIR_CHKLMEM_MALLOC(rcount, sizeof(int) * steps);
    MPIR_CHKLMEM_MALLOC(scount, sizeof(int) * steps);
    step = 0;
    while (mask < comm_size) {
        int partner, abs_partner;
        int nbtb = MPII_Bine_negabinary_to_binary((mask << 1) - 1);
        if (vrank % 2 == 0) {
            partner = MPII_Bine_mod_pof2(vrank + nbtb, comm_size);
        } else {
            partner = MPII_Bine_mod_pof2(vrank - nbtb, comm_size);
        }
        abs_partner = MPII_Bine_mod_pof2(partner + root, comm_size);

        /* Compute send block boundaries inline */
        int send_block_first = MPII_Bine_remap_rank(comm_size, partner) & block_first_mask;
        int send_block_last = send_block_first + inverse_mask - 1;
        sindex[step] = count_per_rank * send_block_first +
                       (send_block_first < rem ? send_block_first : rem);
        scount[step] =
            count_per_rank * (send_block_last - send_block_first + 1) +
            (MPL_MIN(send_block_last, rem) - MPL_MIN(send_block_first, rem)) +
            (send_block_last < rem ? 1 : 0);

        /* Compute recv block boundaries inline */
        int recv_block_first = remapped_rank & block_first_mask;
        int recv_block_last = recv_block_first + inverse_mask - 1;
        rindex[step] = count_per_rank * recv_block_first +
                       (recv_block_first < rem ? recv_block_first : rem);
        rcount[step] =
            count_per_rank * (recv_block_last - recv_block_first + 1) +
            (MPL_MIN(recv_block_last, rem) - MPL_MIN(recv_block_first, rem)) +
            (recv_block_last < rem ? 1 : 0);

        mpi_errno = MPIC_Sendrecv(resbuf + sindex[step] * extent, scount[step], datatype,
                                  abs_partner, MPIR_REDUCE_TAG, tmpbuf + rindex[step] * extent,
                                  rcount[step], datatype, abs_partner, MPIR_REDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        mpi_errno = MPIR_Reduce_local(tmpbuf + rindex[step] * extent,
                                      resbuf + rindex[step] * extent, rcount[step], datatype,
                                      op);
        MPIR_ERR_CHECK(mpi_errno);

        mask <<= 1;
        inverse_mask >>= 1;
        block_first_mask >>= 1;
        step++;
    }

    /***** Gather *****/
    mask >>= 1;
    inverse_mask = 0x1;
    block_first_mask = ~0x0;
    /* I send in the step corresponding to the position (starting from right)
     * of the first 1 in my remapped rank -- this indicates the step when the
     * data reaches me in a scatter
     */
    receiving_mask = 0; /* Root never sends in gather */
    if (rank != root) {
        receiving_mask = 0x1 << (MPII_Bine_ffs(remapped_rank) - 1); /* ffs starts counting from 1, thus -1 */
    }
    step = steps - 1;
    while (mask > 0) {
        int partner, abs_partner;
        int nbtb = MPII_Bine_negabinary_to_binary((mask << 1) - 1);
        if (vrank % 2 == 0) {
            partner = MPII_Bine_mod_pof2(vrank + nbtb, comm_size);
        } else {
            partner = MPII_Bine_mod_pof2(vrank - nbtb, comm_size);
        }
        abs_partner = MPII_Bine_mod_pof2(partner + root, comm_size);

        /* Only the one with 0 in the i-th bit starting from the left (i is the
         * step) survives
         */
        if (inverse_mask & receiving_mask) {
            mpi_errno = MPIC_Send(resbuf + rindex[step] * extent, rcount[step], datatype,
                                  abs_partner, MPIR_REDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            break;
        } else {
            /* Something similar for the block to recv.
             * I receive my partner's block, but aligned to the power of two
             */
            mpi_errno = MPIC_Recv(resbuf + sindex[step] * extent, scount[step], datatype,
                                  abs_partner, MPIR_REDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }

        mask >>= 1;
        inverse_mask <<= 1;
        block_first_mask <<= 1;
        step--;
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
