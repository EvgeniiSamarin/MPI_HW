#include <iostream>
#include <mpi.h>

int rank, size;
const int array_size = 10000;

void fillArray(int* array) {
    for (int i = 0; i < array_size; i++) {
        array[i] = rand();
    }
}

void first_task() {
    printf("Hello, world! size = %d, rank = %d \n", size, rank);
}

void second_task() {
    int array[array_size];
    if (rank == 0) {
        fillArray(array);
    }
    int differnce = array_size / size;
    int start = rank * differnce;
    int end = differnce * (rank + 1);
    if (rank == size - 1) {
        end = size;
    }
    MPI_Bcast(&array, array_size, MPI_INT, 0, MPI_COMM_WORLD);
    int max = array[start];
    for (int i = start + 1; i < end; i++) {
        if (array[i] > max) max = array[i];
    }
    int abs_max;
    MPI_Reduce(&max, &abs_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    printf("local_max = %d, rank = %d \n", max, rank);
    if (rank == 0) {
        printf("max = %d \n", abs_max);
    }
}

void third_task() {
    int radius = 100;
    int square_side = radius * 2;
    int point_in_circle_one_rank = 0;
    int iteration_for_one_rank = 1e7;
    for (int i = 0; i < iteration_for_one_rank; i++) {
        int x = rand() % square_side - square_side / 2;
        int y = rand() % square_side - square_side / 2;
        if ((x * x + y * y) <= radius * radius)
            point_in_circle_one_rank++;
    }
    int all_points_in_circle = 0;
    MPI_Reduce(&point_in_circle_one_rank, &all_points_in_circle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double pi = 4.0 * all_points_in_circle / ((double)iteration_for_one_rank * size);
        printf("pi = %.6f \n", pi);
    }
}

void fourth_task() {
    int offset = array_size / size;
    int buffer_size = offset;
    int* buffer = new int[buffer_size];
    int* scounts = new int[size];
    int* displs = new int[size];
    int array[array_size];
    for (int i = 0; i < size - 1; i++) {
        scounts[i] = offset;
        displs[i] = i * offset;
    }
    scounts[size - 1] = offset + array_size % size;
    displs[size - 1] = (size - 1) * array_size / size;
    if (rank == 0) {
        for (int i = 0; i < array_size; i++) {
            array[i] = rand() % 100;
        }
    }
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            buffer = new int[scounts[i]];
            break;
        }
    }
    MPI_Scatterv(array, scounts, displs, MPI_INT, buffer, buffer_size, MPI_INT, 0, MPI_COMM_WORLD);
    int local_count = 0;
    int local_sum = 0;
    for (int i = 0; i < buffer_size; i++) {
        if (buffer[i] > 0) {
            local_sum += buffer[i];
            local_count++;
        }
    }
    int time_buffer[2] = { local_sum, local_count };
    const int new_buf_size = 2 * size;
    int* new_buffer = new int[new_buf_size];
    MPI_Gather(&time_buffer[0], 2, MPI_INT, &new_buffer[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int sum = 0;
        int count = 0;
        for (int i = 0; i < new_buf_size; i++) {
            if (i % 2 == 0) {
                sum += new_buffer[i];
            }
            else {
                count += new_buffer[i];
            }
        }
        printf("average = %d \n", sum / count);
    }
}

void fifth_task() {
    const int N = 10;
    int vect_a[N], vect_b[N], proc_mult_sum = 0, mult_sum = 0;
    srand(time(0));

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            vect_a[i] = rand() % 10;
            vect_b[i] = rand() % 10;

            printf("[%d] ", vect_a[i]);
            printf("[%d] \n", vect_b[i]);
        }
    }

    int *len = new int[size];
    int *ind = new int[size];

    int rest = N;
    int k = rest / size;
    len[0] = k;
    ind[0] = 0;

    for (int i = 1; i < size; i++) {
        rest -= k;
        k = rest / (size - i);
        len[i] = k;
        ind[i] = ind[i - 1] + len[i - 1];
    }

    int proc_num = len[rank];
    int *vect_a_proc = new int[proc_num];
    int *vect_b_proc = new int[proc_num];

    MPI_Scatterv(vect_a, len, ind, MPI_INT, vect_a_proc, proc_num,
                 MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vect_b, len, ind, MPI_INT, vect_b_proc, proc_num,
                 MPI_INT, 0, MPI_COMM_WORLD);

    proc_mult_sum = 0;
    for (int i = 0; i < proc_num; i++) {
        proc_mult_sum += vect_a_proc[i] * vect_b_proc[i];
    }

    MPI_Reduce(&proc_mult_sum, &mult_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\nscalar multiplication: %d", mult_sum);
    }
}

void sixth_task() {
    const int N = 10;
    int rank, size, proc_n, proc_min_n;
    int a[N * N];
    int *proc_a, *proc_min_a;

    int maxmin;

    int *localmins = new int[N];

    srand(time(0));

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            int localmin = 1000;
            for (int j = 0; j < N; j++) {

                int f = rand()%1000;
                a[i * N + j] = f;

                if (f < localmin)
                    localmin = f;

                printf("\t%d\t", a[i * N + j]);

            }
            printf(" local min: %d \n", localmin);
            localmins[i] = 1000;
        }

    }
    int *len = new int[size];
    int *ind = new int[size];

    int *len_min = new int[size];
    int *ind_min = new int[size];

    int rest = N;
    int k = rest / size;
    len[0] = k * N;
    ind[0] = 0;

    len_min[0] = k;
    ind_min[0] = 0;

    for (int i = 1; i < size; i++) {
        rest -= k;
        k = rest / (size - i);
        len[i] = k * N;
        ind[i] = ind[i - 1] + len[i - 1];

        len_min[i] = k;
        ind_min[i] = ind_min[i - 1] + len_min[i - 1];
    }

    proc_n = len[rank];
    proc_a = new int[proc_n];

    proc_min_n = len_min[rank];
    proc_min_a = new int[proc_min_n];

    MPI_Scatterv(a, len, ind, MPI_INT, proc_a, proc_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(localmins, len_min, ind_min, MPI_INT, proc_min_a, proc_min_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    for (int i = 0; i < proc_n / N; i++) {

        for (int j = 0; j < N; j++) {
            if (proc_min_a[i] > proc_a[i * N + j]) {
                proc_min_a[i] = proc_a[i * N + j];
            }
        }

    }

    for (int i = 0; i < proc_n / N; i++) {
        if (proc_min_a[0] < proc_min_a[i]) {
            proc_min_a[0] = proc_min_a[i];
        }
    }

    MPI_Reduce(proc_min_a, &maxmin, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MAXMIN: %d \n", maxmin);
    }
}

void seventh_task() {
    int x[5][5];
    int y[5];
    int ProcRank, ProcNum, N = 5;

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    int elements_per_proc = N;
    int* subarr1 = new int[elements_per_proc];
    int* result = new int[N];

    if (ProcRank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                x[i][j] = rand() % 100;
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("\t%d\t", x[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        for (int i = 0; i < N; i++) {
            y[i] = rand() % 100;
            printf("\t%d\t", y[i]);
        }
        printf("\n\n");
    }
    int xi[N * N];
    for(int i = 0; i < N; i++) {
        for(int j = 0;j < N; j++){
            xi[i * N + j] = x[j][i];
        }
    }
    MPI_Scatter(xi, elements_per_proc, MPI_INT,
                subarr1, elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < elements_per_proc; i++)
        subarr1[i] *= y[ProcRank];


    MPI_Reduce(subarr1, result, elements_per_proc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ProcRank == 0)
        for (int i = 0; i < N; i++) {
            printf(" %d ", result[i]);
        }
}


void eigth_task() {
    int x[10];
    int ProcRank, ProcNum, N = 10;

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    srand(time(0));

    int elements_per_proc = N / ProcNum;
    int *subarr1 = new int[elements_per_proc];
    int newArr[elements_per_proc];

    if (ProcRank == 0) {
        for (int i = 0; i < N; i++) {
            x[i] = rand() % 100;
            printf(" %d ", x[i]);
        }
        printf("\n");
        for (int i = 0; i < N; i += elements_per_proc) {
            MPI_Send(x + i, elements_per_proc, MPI_INT, i / elements_per_proc, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Recv(subarr1, elements_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < elements_per_proc; i++) {
        printf(" %d ", subarr1[i]);
    }
    printf(" : from process %d\n", ProcRank);
    MPI_Send(subarr1, elements_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD);


    if (ProcRank == 0) {
        for (int i = 0; i < ProcNum; ++i) {
            MPI_Recv(newArr, elements_per_proc, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < elements_per_proc; ++j) {
                printf(" %d ", newArr[j]);
            }
        }
    }
}

void ninth_task() {
    int ProcRank, ProcNum, N = 17;
    int* x = new int[N];
    int* result = new int[N];

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        for (int i = 0; i < N; i++) {
            x[i] = rand() % 100;
            printf(" %d ",x[i]);
        }
        printf("\n");
    }

    int* len = new int[ProcNum];
    int* ind = new int[ProcNum];
    int* revind = new int[ProcNum];

    int rest = N;
    int k = rest / ProcNum;
    len[0] = k;
    ind[0] = 0;
    revind[0] = N - k;

    for (int i = 1; i < ProcNum; i++) {
        rest -= k;
        k = rest / (ProcNum - i);
        len[i] = k;
        ind[i] = ind[i - 1] + len[i - 1];
        revind[i] = revind[i - 1] - len[i];
    }
    int ProcLen = len[ProcRank];
    int* subarr = new int[ProcLen];

    MPI_Scatterv(x, len, ind, MPI_INT, subarr, ProcLen, MPI_INT, 0, MPI_COMM_WORLD);

    int* revers = new int[ProcLen];
    for(int i = 0; i < ProcLen; i++) {
        revers[i] = subarr[ProcLen - i - 1];
    }

    MPI_Gatherv(revers, ProcLen, MPI_INT, result, len, revind, MPI_INT, 0, MPI_COMM_WORLD);

    if(ProcRank == 0) {
        printf("\n");
        for(int i = 0; i < N; i++)
            printf(" %d ", result[i]);
    }
}

void tenth_task() {
    int x[2000000];
    int ProcRank, ProcNum, N = 2000000;
    double startSend, endSend;

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if(ProcRank == 0) {
        for (int i = 0; i < N; i++) {
            x[i] = rand() % 100;
        }

        startSend = MPI_Wtime();
        MPI_Send(x, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(x, N, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_Barrier(MPI_COMM_WORLD);
        endSend = MPI_Wtime();
        printf("Send time %f\n", endSend-startSend);

        startSend = MPI_Wtime();
        MPI_Ssend(x, N, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(x, N, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        endSend = MPI_Wtime();
        printf("Ssend time %f\n", endSend-startSend);

        startSend = MPI_Wtime();
        MPI_Rsend(x, N, MPI_INT, 1, 2, MPI_COMM_WORLD);
        MPI_Recv(x, N, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        endSend = MPI_Wtime();
        printf("rsend time %f\n", endSend-startSend);

    } else {
        MPI_Recv(x, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(x, N, MPI_INT, 0, 0, MPI_COMM_WORLD);

        MPI_Recv(x, N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Ssend(x, N, MPI_INT, 0, 1, MPI_COMM_WORLD);

        MPI_Recv(x, N, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Rsend(x, N, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

}

void eleventh_task() {
    MPI_Status st;
    int value;
    if (rank == 0) {
        value = 1;
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, size - 1, size - 1, MPI_COMM_WORLD, &st);
        printf("value = %d, rank = %d \n", value, rank);
    }
    else {
        int sender = rank - 1;
        int reciver = rank + 1;
        if (rank == size - 1) reciver = 0;
        MPI_Recv(&value, 1, MPI_INT, sender, sender, MPI_COMM_WORLD, &st);
        printf("value = %d, rank = %d \n", value, rank);
        value += rank;
        MPI_Send(&value, 1, MPI_INT, reciver, rank, MPI_COMM_WORLD);
    }
}

/*
* После полного круга из задачи 11, создаем новый коммуникатор (произвольным образом, с меньшим количеством процессов) и повторяем процедуру.
*/
void twelfth_task() {
    eleventh_task();
    MPI_Finalize();
    MPI_Init(NULL, NULL);
    MPI_Group group;
    MPI_Group new_group;
    MPI_Comm comm;
    MPI_Comm_group(MPI_COMM_WORLD, &group);
    const int ranks[1] = { 3 };
    MPI_Group_excl(group, 1, ranks, &new_group);

    MPI_Comm_create(MPI_COMM_WORLD, new_group, &comm);
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Status st;
    int value;
    if (rank == 0) {
        value = 1;
        MPI_Send(&value, 1, MPI_INT, 1, 0, comm);
        MPI_Recv(&value, 1, MPI_INT, size - 1, size - 1, comm, &st);
        printf("value = %d, rank = %d \n", value, rank);
    }
    else {
        int sender = rank - 1;
        int reciver = rank + 1;
        if (rank == size - 1) reciver = 0;
        MPI_Recv(&value, 1, MPI_INT, sender, sender, MPI_COMM_WORLD, &st);
        printf("value = %d, rank = %d \n", value, rank);
        value += rank;
        MPI_Send(&value, 1, MPI_INT, reciver, rank, comm);
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
   // MPI_Comm_size(MPI_COMM_WORLD, &rank);
   // MPI_Comm_rank(MPI_COMM_WORLD, &size);
    ninth_task();
    MPI_Finalize();
    return EXIT_SUCCESS;
}