from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total_ranks = comm.Get_size()

if (rank == 0):
    start = MPI.Wtime()

N = 100000000
massiv = np.ones(N)

if (rank == 0):
    totals = np.empty(N)
else:
    totals = None

comm.Reduce(
    [massiv, MPI.DOUBLE],
    [totals, MPI.DOUBLE],
    op = MPI.SUM,
    root = 0
)

if (rank == 0):
    print(f'Sum: {totals[:10]}')
    print(f'Time: {MPI.Wtime() - start} sec')