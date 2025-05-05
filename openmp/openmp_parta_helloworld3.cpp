#include <iostream>
#include <mutex>
#include <omp.h>

int main() {
    std::mutex mtx;
    int numThreads;
    bool hasError = true;

    // Prompt user to enter number of threads to be created
    while (hasError) {
        std::cout << "Enter number of threads: ";
        std::cin >> numThreads;

        // Check for input failure or non-positive integers
        if (std::cin.fail() || numThreads <= 0) {
            // Invalid input
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "* * * Invalid input. Please enter a positive integer * * *\n";
        } else {
            // Valid input
            hasError = false;
        }
    }

    std::cout << std::endl;

    // Set the number of threads based on user input
    #pragma omp parallel num_threads(numThreads)
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