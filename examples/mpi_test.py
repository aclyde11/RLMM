from mpi4py import MPI
import numpy as np


def func(i):
    return i * i
def single_threaded(n):
    sum = 0
    for i in range(n):
        sum += func(i)
    return sum

def master():
    splits = np.array_split(list(range(n)), world_size - 1)
    for i in range(1, world_size):
        print("Sending work to ", i)
        comm.send(splits[i - 1], dest=i)   #
    sum = 0
    for j in range(1, world_size):
        sum += comm.recv(source=j)
    print("Total sum ", sum)
def slave():
    work = comm.recv(source=0)
    print(f'Got work from master, rank {rank}, {work}')
    sum = 0
    for value in work:
        sum += func(value)
    comm.send(sum, dest=0)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    n = 100
    if rank == 0: # master
        master()
    else:
        slave()
    comm.Barrier()