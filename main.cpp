#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>

void printHelloWorld() {
#pragma omp parallel num_threads(4) default(none) shared(std::cout)
    {
        std::cout << "Hello World!" << std::endl;
    }
}

void printThreadInfo(int k) {
#pragma omp parallel num_threads(k) default(none) shared(std::cout, k)
    {
        int thread_num = omp_get_thread_num();
        int num_threads = omp_get_num_threads();


        if (thread_num % 2 == 0) { // Проверяем, является ли номер нити четным
#pragma omp critical // Для синхронизации доступа к std::cout
            {
                std::cout << "I am " << thread_num << " thread from " << num_threads << " threads!" << std::endl;
            }
        }
    }
}

void printThreadRank(int k) {
#pragma omp parallel num_threads(k) default(none) shared(std::cout, k)
    {
        int rank = omp_get_thread_num(); // rank - приватная переменная для каждого потока

#pragma omp critical // Для синхронизации доступа к std::cout
        {
            std::cout << "I am " << rank << " thread." << std::endl;
        }
    }
}

void computeSum(int N, int k) {
    int totalSum = 0;
#pragma omp parallel num_threads(k) default(none) shared(std::cout, k, N) reduction(+:totalSum)
    {
        int thread_num = omp_get_thread_num();
        int sum = 0;
        int chunk_size = (N + k - 1) / k; // Число элементов, которое каждая нить должна обработать

        int start = thread_num * chunk_size + 1; // Стартовый индекс для каждой нити
        int end = std::min((thread_num + 1) * chunk_size, N); // Конечный индекс для каждой нити

        for (int i = start; i <= end; ++i) {
            sum += i;
        }

#pragma omp critical
        {
            std::cout << "[" << thread_num << "]: Sum = " << sum << std::endl;
        }

        totalSum += sum;
    }

    std::cout << "Sum = " << totalSum << std::endl;
}

void computeSumFor(int N, int k) {
    int totalSum = 0;
    std::vector<int> partialSums(k, 0);

#pragma omp parallel num_threads(k)  default(none) shared(std::cout, k, N, partialSums)
    {
        int thread_num = omp_get_thread_num();

#pragma omp for
        for (int i = 1; i <= N; ++i) {
            partialSums[thread_num] += i;
        }
    }

    // Вывод частичных сумм и вычисление общей суммы
    for (int i = 0; i < k; ++i) {
        std::cout << "[" << i << "]: Sum = " << partialSums[i] << std::endl;
        totalSum += partialSums[i];
    }

    std::cout << "Sum = " << totalSum << std::endl;
}



void computeSumWithSchedule(int k, int N) {
    int totalSum = 0;
    std::vector<int> partialSums(k, 0);

#pragma omp parallel num_threads(k) default(none) shared(std::cout, k, N, partialSums)
    {
        int thread_num = omp_get_thread_num();

#pragma omp for schedule(dynamic)
//#pragma omp for schedule(static)
        for (int i = 1; i <= N; ++i) {
#pragma omp critical
            {
                std::cout << "[" << thread_num << "]: calculation of the iteration number " << i << std::endl;
            }
            partialSums[thread_num] += i;
        }
    }

    for (int i = 0; i < k; ++i) {
        std::cout << "[" << i << "]: Sum = " << partialSums[i] << std::endl;
        totalSum += partialSums[i];
    }
    std::cout << "Sum = " << totalSum << std::endl;
}

double calculatePi(int numSteps) {
    double step = 1.0 / double(numSteps);
    double sum = 0.0;

#pragma omp parallel  for default(none)  reduction(+:sum) shared(step, numSteps)
    for (int i = 0; i < numSteps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    return step * sum;
}


long long parallel_sum(int N, int num_threads) {
    // Инициализация вектора для хранения частичных сумм
    std::vector<long long> partial_sums(num_threads, 0);

    // Распараллеливаем цикл
#pragma omp parallel num_threads(num_threads) default(none) shared( num_threads, N, partial_sums)
    {
        // Получаем номер потока
        int thread_num = omp_get_thread_num();

        // Разбиваем интервал [1, N] на num_threads частей
        int start = (N / num_threads) * thread_num + 1;
        int end = thread_num == num_threads - 1 ? N : start + (N / num_threads) - 1;

        // Вычисляем частичную сумму
        for (int i = start; i <= end; ++i) {
            partial_sums[thread_num] += i;
        }
    }

    // Суммируем частичные суммы
    long long total_sum = 0;
    for (const auto &partial_sum: partial_sums) {
        total_sum += partial_sum;
    }

    return total_sum;
}

void multiplyMatrices(double **A, double **B, double **C, int n) {
#pragma omp parallel for default(none) shared(n, A, B, C)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void performParallelSections(int k) {
    omp_set_num_threads(k); // Установка количества потоков

#pragma omp parallel default(none) shared(std::cout )
    {
        int thread_id = omp_get_thread_num(); // Получение ID текущей нити
        std::cout << "[" << thread_id << "]: parallel region" << std::endl;

#pragma omp sections
        {
#pragma omp section
            {
                std::cout << "[" << thread_id << "]: came in section 1" << std::endl;
            }

#pragma omp section
            {
                std::cout << "[" << thread_id << "]: came in section 2" << std::endl;
            }

#pragma omp section
            {
                std::cout << "[" << thread_id << "]: came in section 3" << std::endl;
            }
        }
    }
}

int main() {
    int choice;
    int N, k;
    while (true) {
        std::cout << "Enter a number from 2 to 10: ";
        std::cin >> choice;


        switch (choice) {
            case 2:
                printHelloWorld();
                break;
            case 3:
                std::cout << "Enter the number of threads: ";
                std::cin >> k;

                printThreadInfo(k);
                break;
            case 4:
                std::cout << "Enter the number of threads: ";
                std::cin >> k;

                printThreadRank(k);
                break;
            case 5:
                std::cout << "Enter the number of threads k: ";
                std::cin >> k;
                std::cout << "Enter the number N: ";
                std::cin >> N;

                computeSum(N, k);
                break;

            case 6:
                std::cout << "Enter the number of threads k: ";
                std::cin >> k;
                std::cout << "Enter the number N: ";
                std::cin >> N;

                computeSumFor(N, k);
                break;

            case 7:
                std::cout << "Enter the number of threads k: ";
                std::cin >> k;
                std::cout << "Enter the number N: ";
                std::cin >> N;

                computeSumWithSchedule(k, N);
                break;

            case 8:
                std::cout << "Enter the number N: ";
                std::cin >> N;

                {
                    double pi = calculatePi(N);
                    std::cout << "Pi: " << pi << std::endl;
                }
                break;

            case 9:
                std::cout << "Enter the number N: ";
                std::cin >> N;
                {

                    double **A = new double *[N];
                    double **B = new double *[N];
                    double **C = new double *[N];

                    for (int i = 0; i < N; i++) {
                        A[i] = new double[N];
                        B[i] = new double[N];
                        C[i] = new double[N];
                    }

                    std::cout << "Enter elements of matrix A:" << std::endl;
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            std::cin >> A[i][j];
                        }
                    }

                    std::cout << "Enter elements of matrix B:" << std::endl;
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            std::cin >> B[i][j];
                        }
                    }

                    multiplyMatrices(A, B, C, N);

                    std::cout << "Resultant matrix C:" << std::endl;
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            std::cout << C[i][j] << " ";
                        }
                        std::cout << std::endl;
                    }

                    // Освобождение памяти
                    for (int i = 0; i < N; i++) {
                        delete[] A[i];
                        delete[] B[i];
                        delete[] C[i];
                    }
                    delete[] A;
                    delete[] B;
                    delete[] C;
                }
                break;

            case 10:
                std::cout << "Enter the number of threads (k): ";
                std::cin >> k;

                performParallelSections(k);
                break;
            case 0:
                std::cout << "Enter the number N: ";
                std::cin >> N;

                std::cout << "Enter the number of threads k: ";
                std::cin >> k;

                {
                    auto start_time = std::chrono::high_resolution_clock::now();
                    long long total_sum = parallel_sum(N, k);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time).count();

                    std::cout << "Total sum from 1 to " << N << " is " << total_sum << std::endl;
                    std::cout << "Time taken: " << duration << " microseconds" << std::endl;
                }

                break;
            default:
                std::cout << "The number isn't 2-10 ";
                break;
        }
    }
    return 0;
}
