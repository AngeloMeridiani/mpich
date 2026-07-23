/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_bine.h"

int MPIR_Reduce_intra_bine_lat(const void *sendbuf,
                               void *recvbuf,
                               MPI_Aint count,
                               MPI_Datatype datatype,
                               MPI_Op op, int root, MPIR_Comm * comm_ptr, int coll_attr)
{

    int comm_size, rank, dtsize, vrank, mask, mpi_errno = MPI_SUCCESS;
    int partner, btnb_vrank, mask_lsbs, lsbs, equal_lsbs;
    MPI_Aint true_lb, true_extent, extent;
    const void *send_ptr = NULL;
    char *tmpbuf = NULL;
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

    MPIR_Assert(MPIR_Op_is_commutative(op));

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    /* If I'm not the root, then my recvbuf may not be valid, therefore
     * I have to allocate a temporary one */
    if (rank != root) {
        MPIR_CHKLMEM_MALLOC(recvbuf, count * (MPL_MAX(extent, true_extent)));
        recvbuf = (void *) ((char *) recvbuf - true_lb);
    }

    if ((rank != root) || (sendbuf != MPI_IN_PLACE)) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* mod computes math modulo rather than reminder */
    vrank = MPII_Bine_mod_pof2(rank - root, comm_size);
    mask = 0x1;
    btnb_vrank = MPII_Bine_binary_to_negabinary(vrank);
    while (mask < comm_size) {
        partner = btnb_vrank ^ ((mask << 1) - 1);
        partner = MPII_Bine_mod_pof2(MPII_Bine_negabinary_to_binary(partner) + root, comm_size);
        mask_lsbs = (mask << 2) - 1;   /* Mask with step + 2 LSBs set to 1 */
        lsbs = btnb_vrank & mask_lsbs; /* Extract k LSBs */
        equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

        if (!equal_lsbs || ((mask << 1) >= comm_size && (rank != root))) {
            /* TODO: use a pointer to choose whether to send recvbuf or sendbuf beforehand 
             * (avoids using MPIR_Localcopy()) */
            mpi_errno = MPIC_Send(recvbuf, count, datatype, partner, MPIR_REDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            break;
        } else {

            if (tmpbuf == NULL) {
                MPIR_CHKLMEM_MALLOC(tmpbuf, count * (MPL_MAX(extent, true_extent)));
                /* adjust for potential negative lower bound in datatype */
                tmpbuf = (void *) ((char *) tmpbuf - true_lb);
            }

            mpi_errno = MPIC_Recv(tmpbuf, count, datatype, partner, MPIR_REDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);

            mpi_errno = MPIR_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
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