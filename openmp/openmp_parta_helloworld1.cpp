#include <iostream>
#include <omp.h>

int main() {
    constexpr int kNumThreads = 10;

    // Fix the number of threads
    #pragma omp parallel num_threads(kNumThreads)
    {
        // Get thread number
        int tid = omp_get_thread_num();

        // Print hello world with thread number
        #pragma omp critical
        {
            std::cout << "Thread " << tid << ": Hello world" << std::endl;
        }
    }

    return 0;
}