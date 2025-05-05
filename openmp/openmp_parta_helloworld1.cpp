#include <iostream>
#include <mutex>
#include <omp.h>

int main() {
    std::mutex mtx;
    constexpr int kNumThreads = 10;

    // Fix the number of threads
    #pragma omp parallel num_threads(kNumThreads)
    {
        // Lock mutex to ensure mutually exclusive access to the output stream
        mtx.lock();

        // Get thread number
        int tid = omp_get_thread_num();
        // Print hello world with thread number
        std::cout << "Thread " << tid << ": Hello world" << std::endl;

        // Unlock mutex
        mtx.unlock();
    }
}