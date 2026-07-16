/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_bine.h"

int MPIR_Gather_intra_bine(const void *sendbuf, MPI_Aint sendcount,
                           MPI_Datatype sendtype, void *recvbuf,
                           MPI_Aint recvcount, MPI_Datatype recvtype, int root,
                           MPIR_Comm *comm_ptr, int coll_attr)
{

    int comm_size, rank, stsize, mpi_errno = MPI_SUCCESS;
    int vrank, extension_direction, mask, partner, mask_lsbs, lsbs, equal_lsbs;
    int nbytes = 0, sendtype_size, recvtype_size;
    MPI_Aint min_block_resident, max_block_resident;
    MPI_Aint recv_start, recv_end;
    MPI_Aint extent = 0;
    MPI_Aint tmp_buf_size;
    void *tmp_buf = NULL;
    MPIR_CHKLMEM_DECL();

    if (comm_size == 1) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                       recvbuf, recvcount, recvtype);
            MPIR_ERR_CHECK(mpi_errno);
        }
        goto fn_exit;
    }

    if ((rank == root && recvcount == 0) || (rank != root && sendcount == 0)) {
        goto fn_exit;
    }

    MPIR_Assert(sendcount == recvcount && sendtype == recvtype);

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    if (rank != root) {
        MPIR_Datatype_get_extent_macro(recvtype, extent);
    }

    if (rank == root) {
        MPIR_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvtype_size * recvcount;
    } else {
        MPIR_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendtype_size * sendcount;
    }
    tmp_buf_size = comm_size * nbytes;

    if (rank == root) {
        if (sendbuf !=  MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                       ((char *) recvbuf + extent * recvcount * rank),
                                       recvcount, recvtype);
            MPIR_ERR_CHECK(mpi_errno);
        }
    } else {
        /* copy from sendbuf into tmp_buf */
        mpi_errno =
            MPIR_Localcopy(sendbuf, sendcount, sendtype, tmp_buf, nbytes, MPIR_BYTE_INTERNAL);
        MPIR_ERR_CHECK(mpi_errno); 
    }

    /* I have the blocks in range [min_block_resident, max_block_resident] */
    min_block_resident = rank;
    max_block_resident = rank;
    /* MPII_Bine_mod computes math modulo rather than reminder */
    vrank = MPII_Bine_mod(rank - root, comm_size);
    extension_direction = 1; /* Down */
    if (rank % 2) {
        extension_direction = -1; /* Up */
    }
    mask = 0x1;
    while (mask < comm_size) {
        partner = MPII_Bine_binary_to_negabinary(vrank) ^ ((mask << 1) - 1);
        partner = MPII_Bine_mod(MPII_Bine_negabinary_to_binary(partner) + root,
                                comm_size);
        /* Mask with step + 2 LSBs set to 1 */
        mask_lsbs = (mask << 2) - 1;
        /* Extract k LSBs */
        lsbs = MPII_Bine_binary_to_negabinary(vrank) & mask_lsbs;
        equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

        if (!equal_lsbs || ((mask << 1) >= comm_size && (rank != root))) {
            if (max_block_resident >= min_block_resident) {
                /* Single send */
                mpi_errno = MPIC_Send((char *)recvbuf + min_block_resident * recvcount * stsize,
                                      recvcount * (max_block_resident - min_block_resident + 1),
                                      sendtype, partner, 0, comm_ptr, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            } else {
                /* Wrapped send */
                mpi_errno = MPIC_Send((char *)recvbuf + min_block_resident * recvcount * stsize,
                                      recvcount * ((comm_size - 1) - min_block_resident + 1),
                                      sendtype, partner, 0, comm_ptr, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);

                mpi_errno = MPIC_Send((char *)recvbuf, recvcount * (max_block_resident + 1),
                                      sendtype, partner, 0, comm_ptr, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }
            break;
        } else {
            /* Determine if I extend the data I have up or down */
            /* Receive [recv_start, recv_end] */
            if (extension_direction == 1) {
                recv_start = MPII_Bine_mod(max_block_resident + 1, comm_size);
                recv_end = MPII_Bine_mod(max_block_resident + mask, comm_size);
                max_block_resident = recv_end;
            } else {
                recv_end = MPII_Bine_mod(min_block_resident - 1, comm_size);
                recv_start =
                    MPII_Bine_mod(min_block_resident - mask, comm_size);
                min_block_resident = recv_start;
            }
            if (recv_end >= recv_start) {
                /* Single recv */
                mpi_errno =
                    MPIC_Recv((char *)recvbuf + recv_start * recvcount * stsize,
                              recvcount * (recv_end - recv_start + 1), sendtype,
                              partner, 0, comm_ptr, MPI_STATUS_IGNORE);
                MPIR_ERR_CHECK(mpi_errno);
            } else {
                /* Wrapped recv */
                mpi_errno = MPIC_Recv((char *)recvbuf + recv_start * recvcount * stsize,
                                      recvcount * ((comm_size - 1) - recv_start + 1), sendtype,
                                      partner, 0, comm_ptr, MPI_STATUS_IGNORE);
                MPIR_ERR_CHECK(mpi_errno);
                mpi_errno = MPIC_Recv((char *)recvbuf,
                                      recvcount * (recv_end + 1), sendtype,
                                      partner, 0, comm_ptr, MPI_STATUS_IGNORE);
                MPIR_ERR_CHECK(mpi_errno);
            }

            extension_direction *= -1;
        }
        mask <<= 1;
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}