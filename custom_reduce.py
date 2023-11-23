from mpi4py import MPI
import numpy as np

def findNext(last):
    next = [last[i] for i in range(1, len(last), 2)] # каждый четный уже отправил, поэтому убираем их из списка
    
    last_rank = total_ranks - 1
    
    if (len(next) == 1): # если остался один, то есть два варианта
        if (last_rank not in next): # последнего нет в списке - значит он в очереди и его нужно добавить
            next.append(last_rank)
        else: # иначе возвращаем список, в котором один только последний ранк - сигнал о конце коммуникаций
            return next
    
    if (len(next) % 2 != 0): # если список содержит нечетное количество, то два варианта
        if (last_rank in next): # последнему не хватает пары, отставляем его в очередь на данную итерацию
            next.pop(-1)
        else: # последнего нет в списке и кому-то не хватает пары, поэтому забираем последнего из очереди
            next.append(last_rank)
    
    return next

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total_ranks = comm.Get_size()

if (rank == total_ranks - 1):
    start = MPI.Wtime()

active = [i for i in range(total_ranks)] # сначала активны все
if (total_ranks % 2 != 0): # кроме, может быть, последнего, когда начальное количество процессов нечетное
    active.pop(-1) # последний сразу попадает в очередь, у него нет пары

N = 100000000
massiv = np.ones(N)

while (len(active) >= 2):
    # пускаем в коммуникации только тех, кто активен на данной итерации + последний ранк (он может быть в очереди)
    if (rank in active or rank == (total_ranks - 1)):
        
        if (rank not in active): # это условие срабатывает только тогда, когда последний ранк в очереди
            active = findNext(active) # обновляем у него список активных на след. итерации
            continue
        
        r_ind = active.index(rank) # узнаем индекс ранка в списке активных процессов
        
        if (r_ind % 2 == 0): # четные отправляют нечетным
            comm.Send(massiv, dest=active[r_ind + 1], tag=13) # отправляем соседу справа
        else: # нечетные принимают от четных
            in_data = np.empty(N)
            comm.Recv(in_data, source=active[r_ind - 1], tag=13) # принимаем от соседа слева
            
            massiv += in_data # складываем
    else:
        break # если ранк выпал из коммуникации, отправляем его умирать
    active = findNext(active) # обновляем список активных процессов после итерации

if (rank == total_ranks - 1): # до конца доживает только последний процесс, проводим финальное суммирование
    print(f'Sum: {massiv[:10]}')
    print(f'Time: {MPI.Wtime() - start} sec')