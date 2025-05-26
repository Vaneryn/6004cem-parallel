#include <iostream>
#include <omp.h>

int main() {
    // Number of threads is already set by altering the environment variable (OMP_NUM_THREADS) using a Linux command
    #pragma omp parallel
    {
        // Get thread number
        int tid = omp_get_thread_num();
        
        // Print hello world with thread number
        #pragma omp critical
        {
            std::cout << "Thread " << tid << ": Hello world" << std::endl;
        }
    }
}