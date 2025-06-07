#include <iostream>
#include <limits>
#include <omp.h>

int main() {
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