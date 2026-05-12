/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_bine.h"

int MPIR_Bcast_intra_bine_bdw_remap(void *buf, MPI_Aint count, MPI_Datatype datatype, int root,
                                    MPIR_Comm *comm_ptr, int coll_attr);
int MPIR_Bcast_intra_bine_lat_i(void *buf, MPI_Aint count, MPI_Datatype datatype, int root,
                                MPIR_Comm *comm_ptr, int coll_attr);

int MPIR_Bcast_intra_bine(void *buf, MPI_Aint count, MPI_Datatype datatype, int root,
                          MPIR_Comm *comm_ptr, int coll_attr)
{
    int comm_size, mpi_errno = MPI_SUCCESS;
    MPI_Aint nbytes = 0;
    MPI_Aint type_size;

    MPIR_CHKLMEM_DECL();

    comm_size = comm_ptr->local_size;

    MPIR_Datatype_get_size_macro(datatype, type_size);
    nbytes = type_size * count;

    if (nbytes == 0) {
        goto fn_exit;
    }

    MPIR_ERR_CHKANDJUMP(!MPL_is_pof2(comm_size), mpi_errno, MPI_ERR_COMM, "**comm");

    if ((nbytes < MPIR_CVAR_BCAST_SHORT_MSG_SIZE) ||
        (comm_size < MPIR_CVAR_BCAST_MIN_PROCS)) {
        mpi_errno = MPIR_Bcast_intra_bine_lat_i(buf, count, datatype, root, comm_ptr, coll_attr);
    } else {
        mpi_errno = MPIR_Bcast_intra_bine_bdw_remap(buf, count, datatype, root, comm_ptr, coll_attr);
    }
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
