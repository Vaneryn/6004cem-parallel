#include <iostream>
#include <mutex>
#include <omp.h>

int main() {
    std::mutex mtx;

    // Number of threads is already set by altering the environment variable (OMP_NUM_THREADS) using a Linux command
    #pragma omp parallel
    {
        // Lock mutex to ensure mutually exclusive access to the output stream
        std::lock_guard<std::mutex> lock(mtx);

        // Get thread number
        int tid = omp_get_thread_num();
        
        // Print hello world with thread number
        std::cout << "Thread " << tid << ": Hello world" << std::endl;
    }
}