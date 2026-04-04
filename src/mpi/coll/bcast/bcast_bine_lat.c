#include "mpiimpl.h"
#include "mpir_bine.h"

/* Algorithm: Bine tree bcast
 * For short messages, we use a Bine tree algorithm implemented with distance
 * halving Bine trees for broadcasting data from the root to the leaves.
 */

int MPIR_Bcast_bine_lat_i(void *buf, MPI_Aint count,
                          MPI_Datatype datatype, int root,
                          MPIR_Comm *comm_ptr, int coll_attr)
{
    int comm_size, rank, mpi_errno = MPI_SUCCESS, btnb_vrank;
    int vrank, mask, recvd, req_count = 0, steps;
    int partner, mask_lsbs, lsbs, equal_lsbs;
    MPIR_Request **requests;
    int is_contig;
    MPI_Aint nbytes = 0;
    MPI_Aint type_size;
    void *tmp_buf = NULL;

    MPIR_CHKLMEM_DECL();

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    MPIR_Datatype_get_size_macro(datatype, type_size);
    if (HANDLE_IS_BUILTIN(datatype))
        is_contig = 1;
    else {
        MPIR_Datatype_is_contig(datatype, &is_contig);
    }

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;

    if (!is_contig) {
        MPIR_CHKLMEM_MALLOC(tmp_buf, nbytes);
        if (rank == root) {
            mpi_errno = MPIR_Localcopy(buf, count, datatype,
                                       tmp_buf, nbytes, MPIR_BYTE_INTERNAL);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    vrank = MPII_Bine_mod(rank - root, comm_size); /* MPII_Bine_mod computes math modulo rather than reminder */
    steps = MPL_log2(comm_size);
    mask = 0x1 << (int)(steps - 1);
    recvd = (root == rank);
    btnb_vrank = binary_to_negabinary(vrank);

    MPIR_CHKLMEM_MALLOC(requests, steps * sizeof(MPIR_Request *));

    while (mask > 0) {
        partner = btnb_vrank ^ ((mask << 1) - 1);
        partner = MPII_Bine_mod(MPII_Bine_negabinary_to_binary(partner) + root, comm_size);
        mask_lsbs = (mask << 1) - 1;   /* Mask with num_steps - step + 1 LSBs set to 1 */
        lsbs = btnb_vrank & mask_lsbs; /* Extract k LSBs */
        equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

        if (recvd) {
            if (!is_contig) {
                mpi_errno = MPIC_Isend(tmp_buf, nbytes, MPIR_BYTE_INTERNAL, partner,
                                       MPIR_BCAST_TAG, comm_ptr, &requests[req_count++], coll_attr);
            }
            else {
                mpi_errno = MPIC_Isend(buf, count, datatype, partner,
                                       MPIR_BCAST_TAG, comm_ptr, &requests[req_count++], coll_attr);
            }
            MPIR_ERR_CHECK(mpi_errno);
        }
        else if (equal_lsbs) {
            if (!is_contig) {
                mpi_errno = MPIC_Recv(tmp_buf, nbytes, MPIR_BYTE_INTERNAL,
                                      partner, MPIR_BCAST_TAG, comm_ptr, MPI_STATUS_IGNORE);
            }
            else {
                mpi_errno = MPIC_Recv(buf, count, datatype,
                                      partner, MPIR_BCAST_TAG, comm_ptr, MPI_STATUS_IGNORE);
            }
            MPIR_ERR_CHECK(mpi_errno);
            recvd = 1;
        }
        mask >>= 1;
    }

    mpi_errno = MPIC_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    MPIR_ERR_CHECK(mpi_errno);

    if (!is_contig) {
        if (rank != root) {
            mpi_errno = MPIR_Localcopy(tmp_buf, nbytes, MPIR_BYTE_INTERNAL,
                                       buf, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
