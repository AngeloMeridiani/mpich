#include "mpiimpl.h"
#include "mpir_bine.h"

int MPIR_Scatter_bine(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                      void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                      int root, MPIR_Comm* comm_ptr, int coll_attr) 
{
    MPIR_Assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    MPIR_assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype

    int comm_size, rank, stsize, mpi_errno = MPI_SUCCESS;
    int vrank, halving_direction, mask, recvd = 0, is_leaf = 0;
    int sbuf_offset, vrank_nb;
    int partner, mask_lsbs, lsbs, equal_lsbs;
    size_t min_resident_block, max_resident_block;
    size_t top_start, top_end, bottom_start, bottom_end;
    size_t send_start, send_end, recv_start, recv_end;
    size_t num_blocks;
    int nbytes;
    int is_contig;
    char *tmp_buf = NULL, *sbuf = NULL, *rbuf = NULL;

    MPIR_CHKLMEM_DECL();
    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);
    MPIR_Datatype_get_size_macro(sendtype, stsize);
    if (HANDLE_IS_BUILTIN(sendtype))
        is_contig = 1;
    else
        MPIR_Datatype_is_contig(sendtype, &is_contig);
    
    nbytes = stsize * sendcount;
    if (nbytes == 0)
        goto fn_exit;

    vrank = MPII_Bine_mod(rank - root, comm_size); // MPII_Bine_mod computes math modulo rather than reminder
    halving_direction = 1;      // Down -- send bottom half
    if (rank % 2) {
        halving_direction = -1; // Up -- send top half
    }
    // The gather started with these directions. Thus this will
    // be the direction they ended up with if we have an odd number
    // of steps. If not, invert.
    if (MPL_log2(comm_size) % 2 == 0) {
        halving_direction *= -1;
    }

    // I need to do the opposite of what I did in the gather.
    // Thus, I need to know where min_resident_block and max_resident_block
    // ended up after the last step.
    // Even ranks added 2^0, 2^2, 2^4, ... to max_resident_block
    //   and subtracted 2^1, 2^3, 2^5, ... from min_resident_block
    // Odd ranks subtracted 2^0, 2^2, 2^4, ... from min_resident_block
    //      and added 2^1, 2^3, 2^5, ... to max_resident_block
    if (rank % 2 == 0) {
        max_resident_block =
            MPII_Bine_mod((rank + 0x55555555) & ((0x1 << (int)MPL_log2(comm_size)) - 1), comm_size);
        min_resident_block =
            MPII_Bine_mod((rank - 0xAAAAAAAA) & ((0x1 << (int)MPL_log2(comm_size)) - 1), comm_size);
    } else {
        min_resident_block =
            MPII_Bine_mod((rank - 0x55555555) & ((0x1 << (int)MPL_log2(comm_size)) - 1), comm_size);
        max_resident_block =
            MPII_Bine_mod((rank + 0xAAAAAAAA) & ((0x1 << (int)MPL_log2(comm_size)) - 1), comm_size);
    }

    mask = 0x1 << (int)(MPL_log2(comm_size) - 1);
    sbuf_offset = rank;
    if (root == rank) {
        recvd = 1;
        sbuf = (char *)sendbuf;
    }

    vrank_nb = MPII_Bine_binary_to_negabinary(vrank);
    while (mask > 0) {
        partner = vrank_nb ^ ((mask << 1) - 1);
        partner = MPII_Bine_mod(MPII_Bine_negabinary_to_binary(partner) + root, comm_size);
        mask_lsbs = (mask << 1) - 1; // Mask with num_steps - step + 1 LSBs set to 1
        lsbs = vrank_nb & mask_lsbs; // Extract k LSBs
        equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

        top_start = min_resident_block;
        top_end = MPII_Bine_mod(min_resident_block + mask - 1, comm_size);
        bottom_start = MPII_Bine_mod(top_end + 1, comm_size);
        bottom_end = max_resident_block;
        if (halving_direction == 1) {
            // Send bottom half [..., size - 1]
            send_start = bottom_start;
            send_end = bottom_end;
            recv_start = top_start;
            recv_end = top_end;
            max_resident_block = MPII_Bine_mod(max_resident_block - mask, comm_size);
        } else {
            // Send top half [0, ...]
            send_start = top_start;
            send_end = top_end;
            recv_start = bottom_start;
            recv_end = bottom_end;
            min_resident_block = MPII_Bine_mod(min_resident_block + mask, comm_size);
        }

        if (recvd) {
            if (send_end >= send_start) {
                mpi_errno = MPIC_Send((char *)sbuf + send_start * sendcount * stsize,
                               sendcount * (send_end - send_start + 1), sendtype,
                               partner, 0, comm_ptr, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            } else {
                mpi_errno = MPIC_Send((char *)sbuf + send_start * sendcount * stsize,
                               sendcount * ((comm_size - 1) - send_start + 1), sendtype,
                               partner, 0, comm_ptr, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
                mpi_errno = MPIC_Send((char *)sbuf, sendcount * (send_end + 1), sendtype,
                               partner, 0, comm_ptr, coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }
        } else if (equal_lsbs) {
            // Setup the buffers to be used from now on
            // How large should the tmpbuf be?
            // It must be large enough to hold a number of blocks
            // equal to the number of children in the tree rooted in me.
            num_blocks = MPII_Bine_mod((recv_end - recv_start + 1), comm_size);
            if (recv_start == recv_end) {
                // I am a leaf and this is the last step, I do not need a tmpbuf
                rbuf = (char *)recvbuf;
                is_leaf = 1;
            } else {
                MPIR_CHKLMEM_MALLOC(tmp_buf, nbytes);
                tmp_buf = (char *)malloc(recvcount * num_blocks * stsize);
                if (tmp_buf == NULL) {
                    MPIR_ERR_CHECK(mpi_errno);
                }
                sbuf = (char *)tmp_buf;
                rbuf = (char *)tmp_buf;

                // Adjust min and max resident blocks
                min_resident_block = 0;
                max_resident_block = num_blocks - 1;

                sbuf_offset = MPII_Bine_mod(rank - recv_start, comm_size);
            }
            if (recv_end >= recv_start) {
                mpi_errno = MPIC_Recv((char *)rbuf, recvcount * num_blocks, sendtype,
                               partner, 0, comm_ptr,MPI_STATUS_IGNORE);
                MPIR_ERR_CHECK(mpi_errno);
            } else {
                mpi_errno = MPIC_Recv((char *)rbuf,
                               recvcount * ((comm_size - 1) - recv_start + 1), sendtype,
                               partner, 0, comm_ptr, MPI_STATUS_IGNORE);
                MPIR_ERR_CHECK(mpi_errno);
                mpi_errno = MPIC_Recv((char *)rbuf +
                                   (recvcount * ((comm_size - 1) - recv_start + 1)) *
                                       stsize,
                               recvcount * (recv_end + 1), sendtype, partner, 0, comm_ptr,
                               MPI_STATUS_IGNORE);
                MPIR_ERR_CHECK(mpi_errno);
            }
            recvd = 1;
        }
        mask >>= 1;
        halving_direction *= -1;
    }

    if (!is_leaf) {
        mpi_errno = MPIR_Localcopy((char *)sbuf + sbuf_offset * recvcount * stsize, recvcount * stsize, sendtype, 
                                   (char *)recvbuf, recvcount, recvtype);
        /*memcpy((char *)recvbuf, (char *)sbuf + sbuf_offset * recvcount * stsize,
               recvcount * stsize);*/
        MPIR_ERR_CHECK(mpi_errno);
    }

fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
