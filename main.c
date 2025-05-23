#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * read_matrix
 * -----------
 * Reads an NxN matrix from a CSV file into a 1D float array.
 *
 * Parameters:
 *   filename - path to the CSV file
 *   mat      - pointer to the float array to fill
 *   N        - size of the matrix (NxN)
 *
 * Side Effects:
 *   Exits the program using MPI_Abort if the file can't be opened.
 */
void read_matrix(const char *filename, float *mat, int N) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < N * N; i++) {
        fscanf(f, "%f", &mat[i]);  // Handles without commas
        fgetc(f); // Skip comma or newline
    }
    fclose(f);
}

/**
 * write_matrix
 * ------------
 * Writes an NxN matrix from a 1D float array to a CSV file.
 *
 * Parameters:
 *   filename - path to the output CSV file
 *   mat      - pointer to the float array containing the matrix
 *   N        - size of the matrix (NxN)
 */
void write_matrix(const char *filename, float *mat, int N) {
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%f", mat[i * N + j]);
            if (j < N - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/**
 * main
 * ----
 * Entry point of the MPI matrix multiplication program.
 * Distributes the matrix rows among processes, performs multiplication,
 * and gathers the results.
 *
 * Usage: ./matrix_mpi <matrix_size>
 *
 * Parameters:
 *   argc - argument count
 *   argv - argument vector
 *
 * Returns:
 *   0 on success, non-zero on error (e.g., invalid matrix size).
 *
 * Side Effects:
 *   Reads/writes files, prints execution time, allocates/frees memory.
 */
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

    if (!B || !local_A || !local_C) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = malloc(N * N * sizeof(float));
        C = malloc(N * N * sizeof(float));
        if (!A || !C) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        read_matrix("data/matrix_a.csv", A, N);
        read_matrix("data/matrix_b.csv", B, N);
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
        write_matrix("data/matrix_result.csv", C, N);
        double end = MPI_Wtime();
        printf("Execution Time: %f seconds\n", end - start);
        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
