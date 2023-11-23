Алгоритм custom_reduce.py работает следующим образом:

-На входе у каждого процесса массив из N единиц

-Алгоритм складывает соответсвутющие элементы всех массивов

-На выходе один массив из чисел M, где M - количество запущенных процессов

## Подробности

1. Все процессы нумеруются согласно их рангу.

2. Если в начале их число нечетное, то последний отставляется в "очередь".

3. Теперь, когда число активных процессов четное, все процессы с четным номером отправляют свой вектор нечетному соседу справа. Нечетный процесс принимает вектор и поэлементно складывает два вектора - свой и полученный.

4. На этом итерация заканчивается и вычисляется список активных процессов на следующей итерации с помощью функции findNext. Если процесс не нашел себя в списке активных, то он выходит из цикла коммуникаций.
   
5. Функция findNext - описание:
   1. Принимает в себя предыдущий список и убирает из него все четные процессы - они уже выполнили свою работу по отправке и могут выходить из цикла
   2. Если процессов нечетное количество И последнего нет в списке => значит он в очереди и его можно добавить в список активных на данной итерации
   3. Если процессов осталось нечетное количество И последний уже в списке => у последнего нет пары и он отставляется в очередь
   4. Если процессов четное количество то все ок можно идти на следующую итерацию
   5. Если процессов осталось 1 штука и последний в очереди, то его нужно добавить, чтобы образовалась пара
   6. Если же остался 1 процесс и он последний то просто возвращается список из одного (последнего) процесса - это значит, что все коммуникации окончены

6. Как только в списке активных остался только последний процесс - итерации завершаются и для него, идет вывод результата 

ПРИМЕР

        rank:       0       1       2       3       4
        ITER 0:     |       |       |       |       |
                    |-send->|       |-send->|       | WAIT
        ITER 1:     x       |       x       |       |
                            |------send---->|       | WAIT        
        ITER 2:             x               |       |
                                            |-send->|                      
        ITER 3:                             x       | OUTPUT RESULT


В файле сравнение.xlsx приводится сравнение алгоритма, который описан выше, со стандартным методом MPI Reduce.
