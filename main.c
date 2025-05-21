#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void read_matrix(const char *filename, float *mat, int N) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < N * N; i++) {
        fscanf(f, "%f,", &mat[i]);
    }
    fclose(f);
}

void write_matrix(const char *filename, float *mat, int N) {
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%f,", mat[i * N + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char* argv[]) {
    int rank, size, N;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    N = atoi(argv[1]);
    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "Matrix size must be divisible by number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    int rows_per_process = N / size;

    float *A = NULL, *B = NULL, *C = NULL;
    float *local_A = malloc(rows_per_process * N * sizeof(float));
    float *local_C = malloc(rows_per_process * N * sizeof(float));
    B = malloc(N * N * sizeof(float));

    if (rank == 0) {
        A = malloc(N * N * sizeof(float));
        C = malloc(N * N * sizeof(float));
        read_matrix("matrix_a.csv", A, N);
        read_matrix("matrix_b.csv", B, N);
    }

    double start = MPI_Wtime();

    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows_per_process * N, MPI_FLOAT,
                local_A, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_process * N, MPI_FLOAT,
               C, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_matrix("matrix_result.csv", C, N);
        double end = MPI_Wtime();
        printf("Execution Time: %f seconds\n", end - start);
        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
