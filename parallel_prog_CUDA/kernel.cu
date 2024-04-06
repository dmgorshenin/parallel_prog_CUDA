#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
using namespace std;
using namespace std::chrono;

__global__ void multiplyMatricesKernel(int* matrix1, int* matrix2, int* result_matrix, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int value = 0;
        for (int k = 0; k < size; ++k) {
            value += matrix1[row * size + k] * matrix2[k * size + col];
        }
        result_matrix[row * size + col] = value;
    }
}

class MultiplyMatrix {
private:
    int size;
    int* matrix1;
    int* matrix2;
    int* result_matrix;
    string filename_matrix1 = "matrix1.txt";
    string filename_matrix2 = "matrix2.txt";
    string filename_matrix_res = "result_matrix.txt";

public:
    MultiplyMatrix(int N) : size(N) {
        matrix1 = new int[N * N];
        matrix2 = new int[N * N];
        result_matrix = new int[N * N];
    }

    ~MultiplyMatrix() {
        delete[] matrix1;
        delete[] matrix2;
        delete[] result_matrix;
    }

    void generateRandomMatrix() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dis(0, 99);

        for (int i = 0; i < size * size; ++i) {
            matrix1[i] = dis(gen);
            matrix2[i] = dis(gen);
        }
    }

    void generate_and_save_matrices() {
        generateRandomMatrix();

        writeMatrixToFile(filename_matrix1, matrix1);
        writeMatrixToFile(filename_matrix2, matrix2);
    }

    void multiplyMatrices() {
        int* d_matrix1, * d_matrix2, * d_result_matrix;
        int rank = 128;
        cudaMalloc(&d_matrix1, size * size * sizeof(int));
        cudaMalloc(&d_matrix2, size * size * sizeof(int));
        cudaMalloc(&d_result_matrix, size * size * sizeof(int));

        cudaMemcpy(d_matrix1, matrix1, size * size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix2, matrix2, size * size * sizeof(int), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(rank, rank);
        dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        multiplyMatricesKernel<<<numBlocks, threadsPerBlock>>>(d_matrix1, d_matrix2, d_result_matrix, size);

        cudaMemcpy(result_matrix, d_result_matrix, size * size * sizeof(int), cudaMemcpyDeviceToHost);

        //writeMatrixToFile(filename_matrix_res, result_matrix);

        cudaFree(d_matrix1);
        cudaFree(d_matrix2);
        cudaFree(d_result_matrix);
    }

    void writeMatrixToFile(const string& filename, int* matrix) {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Unable to open file: " << filename << endl;
            exit(1);
        }

        file << size << endl;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                file << matrix[i * size + j] << " ";
            }
            file << endl;
        }

        file.close();
    }
};

void writeTimeToFile(long long computation_time, const string& filename) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    file << computation_time << endl;

    file.close();
}

void writeTaskSizeToFile(int size, long long task_size, const string& filename) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    file << size << endl << task_size << endl;

    file.close();
}

int main() {
    setlocale(LC_ALL, "ru");
    int N = 500;
    string file_stat = "stats_cuda128.txt";

    while (N <= 2300) {
        long long task_size = static_cast<long long>(N) * static_cast<long long>(N) * static_cast<long long>(N);
        writeTaskSizeToFile(N, task_size, file_stat);

        cout << "Размер матриц " << N << "x" << N << endl << "Объем задачи: " << task_size << endl;

        for (size_t i = 0; i < 10; ++i) {
            MultiplyMatrix matrix(N);

            //matrix.generate_and_save_matrices();
            matrix.generateRandomMatrix();

            auto start_compute = high_resolution_clock::now();
            matrix.multiplyMatrices();
            auto stop_compute = high_resolution_clock::now();

            cout << "Матрицы перемножены." << endl;

            auto duration_computation = duration_cast<milliseconds>(stop_compute - start_compute);

            cout << "Время умножения матриц: " << duration_computation.count() << " мс" << endl;
            writeTimeToFile(duration_computation.count(), file_stat);
        }

        N += 100;
    }

    return 0;
}
