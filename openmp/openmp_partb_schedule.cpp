#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Initializes a vector with a fixed sized and value.
 * 
 * Fills the given vector with `size` number of elements, each initialized to `value`.
 * 
 * @param vect The vector to initialize.
 * @param size The number of elements to assign.
 * @param value The value to assign to every element.
 */
void initVector(std::vector<int>& vect, int size, int value) {
    vect.assign(size, value);
}

/**
 * @brief Prints the contents of a vector.
 * 
 * Used for debugging or visual inspection of a vector's contents.
 * 
 * @param vect The vector to print.
 */
void printVector(std::vector<int>& vect) {
    for (int num : vect) {
        std::cout << " " << num;
    }
}

/**
 * @brief Prints a formatted table header with fixed column widths.
 * 
 * @param headers A vector of column header title strings.
 * @param widths A vector of column widths corresponding to each header.
 * @param lineLength The total length of the line separator.
 */
void printTableHeader(const std::vector<std::string>& headers, const std::vector<int>& widths, int lineLength) {
    for (size_t i = 0; i < headers.size(); ++i)
        std::cout << std::left << std::setw(widths[i]) << headers[i];
    std::cout << "\n" << std::string(lineLength, '-') << "\n";
}

/**
 * @brief Prints a single row in a formatted table.
 * 
 * @param values A vector of strings representing the values in the row.
 * @param width2A vector of column widths corresponding to each value.
 */
void printTableRow(const std::vector<std::string>& values, const std::vector<int>& widths) {
    for (size_t i = 0; i < values.size(); ++i)
        std::cout << std::left << std::setw(widths[i]) << values[i];
    std::cout << "\n";
}

/**
 * @brief Runs a parallel vector addition using a specific scheduling method.
 * 
 * Supports both static and dynamic shceduling, with optional chunk size and thread count.
 * Results are printed in a formatted table showing thread ID, iteration, and result.
 * 
 * @param vect1 First input vector.
 * @param vect2 Second input vector.
 * @param vect3 Output vector to store results.
 * @param scheduleType The scheduling method to use ("static" or "dynamic").
 * @param kColWidths Column widths for formatted table output display.
 * @param numThreads Optional number of threads (0 -> use all available processors).
 * @param chunkSize Optional chunk size for scheduling (0 -> use default chunk size).
 */
void runSchedule(const std::vector<int>& vect1, const std::vector<int>& vect2, std::vector<int>& vect3,
    const std::string& scheduleType, const std::vector<int>& kColWidths, int numThreads = 0, int chunkSize = 0) {
    // Set default number of threads
    if (numThreads == 0)
        numThreads = omp_get_num_procs();
        
    const int kSize = vect1.size();

    if (scheduleType == "static") {
        if (chunkSize > 0) {
            #pragma omp parallel for schedule(static, chunkSize) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                int tid = omp_get_thread_num();
                vect3[i] = vect1[i] + vect2[i];
                #pragma omp critical
                printTableRow({std::to_string(tid), std::to_string(i), std::to_string(vect3[i])}, kColWidths);
            }
        } else {
            #pragma omp parallel for schedule(static) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                int tid = omp_get_thread_num();
                vect3[i] = vect1[i] + vect2[i];
                #pragma omp critical
                printTableRow({std::to_string(tid), std::to_string(i), std::to_string(vect3[i])}, kColWidths);
            }
        }
    } else if (scheduleType == "dynamic") {
        if (chunkSize > 0) {
            #pragma omp parallel for schedule(dynamic, chunkSize) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                int tid = omp_get_thread_num();
                vect3[i] = vect1[i] + vect2[i];
                #pragma omp critical
                printTableRow({std::to_string(tid), std::to_string(i), std::to_string(vect3[i])}, kColWidths);
            }
        } else {
            #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                int tid = omp_get_thread_num();
                vect3[i] = vect1[i] + vect2[i];
                #pragma omp critical
                printTableRow({std::to_string(tid), std::to_string(i), std::to_string(vect3[i])}, kColWidths);
            }
        }
    }
}

/**
 * @brief Measures execution time of parallel vector addition using a specific scheduling method.
 * 
 * Runs the vector addition using the specified OpenMP scheduling method and returns the time taken.
 * Output is not printed, only computation is measured for accurate performance measurement.
 * 
 * @param vect1 First input vector.
 * @param vect2 Second input vector.
 * @param vect3 Output vector to store results.
 * @param scheduleType The scheduling method to use ("static" or "dynamic").
 * @param isBalanced Determines whether to run the test with balanced or imbalanced workload per iteration.
 * @param numThreads Optional number of threads (0 -> use all available processors).
 * @param chunkSize Optional chunk size for scheduling (currently not used).
 * @return Elapsed time in seconds.
 */
double measureSchedule(const std::vector<int>& vect1, const std::vector<int>& vect2, std::vector<int>& vect3,
    const std::string& scheduleType, bool isBalanced, int numThreads = 0, int chunkSize = 0) {
    // Set default number of threads
    if (numThreads == 0)
        numThreads = omp_get_num_procs();

    const int kSize = vect1.size();

    double startTime;
    double endTime;

    if (scheduleType == "static") {
        // Static scheduling
        if (isBalanced) {
            // Balanced workload per iteration
            startTime = omp_get_wtime();
            #pragma omp parallel for schedule(static) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                vect3[i] = vect1[i] + vect2[i];
            }
            endTime = omp_get_wtime();
        } else {
            // Imbalanced workload per iteration
            startTime = omp_get_wtime();
            #pragma omp parallel for schedule(static) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                vect3[i] = vect1[i] + vect2[i];

                // Simulate imbalanced workload
                if (i % 100 == 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
            endTime = omp_get_wtime();
        }
    } else if (scheduleType == "dynamic") {
        // Dynamic scheduling
        if (isBalanced) {
            // Balanced workload per iteration
            startTime = omp_get_wtime();
            #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                vect3[i] = vect1[i] + vect2[i];
            }
            endTime = omp_get_wtime();
        } else {
            // Imbalanced workload per iteration
            startTime = omp_get_wtime();
            #pragma omp parallel for schedule(dynamic) num_threads(numThreads)
            for (int i = 0; i < kSize; ++i) {
                vect3[i] = vect1[i] + vect2[i];

                // Simulate imbalanced workload
                if (i % 100 == 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
            endTime = omp_get_wtime();
        }
    }

    return (endTime - startTime);
}

int main() {
    /**
     * OUTLINE: Split program into 2 sections
     * @section 1: Scheduling Behaviour
     * Compare distribution of iterations amongst threads with a very low iteration count (e.g. 12) to visually inspect how chunks are assigned.
     * Static:
     * - run with default chunk size (each thread gets one chunk; iterations are divided into as evenly sized chunks as possible)
     * - run with specified chunk size (block-cyclic behavior/round roubin if size is 1)
     * Dynamic:
     * - run with default chunk size (1; threads request single iterations to process as they become available)
     * - run with specified chunk size (threads request fixed-size chunks to process as they become available)
     * @section 2: Performance Comparison
     * Compare execution time with increasing iterations
     * - run with balanced workload per iteration
     * - run with imbalanced workload per iteration
     */

    // Console UI elements
    constexpr int kLineLength = 50;
    const std::string kSingleLine(kLineLength, '-');
    const std::string kDoubleLine(kLineLength, '=');
    const std::vector<std::string> kScheduleHeaders = {"TID", "Iteration", "Result"};
    const std::vector<std::string> kPerformanceHeaders = {"Size", "Static Total (s)", "Dynamic Total (s)", "Static Average (s)", "Dynamic Average (s)"};
    const std::vector<int> kScheduleColWidths = {10, 15, 10};
    const std::vector<int> kPerformanceColWidths = {10, 20, 20, 22, 22};

    // Parallelization configs
    constexpr int kNumThreads = 4;
    constexpr int kStaticChunkSize = 2;
    constexpr int kDynamicChunkSize = 2;

    // Vector properties
    constexpr int kSize = 12;
    constexpr int kValue1 = 10;
    constexpr int kValue2 = 20;
    constexpr int kValue3 = 0;

    // Vectors
    std::vector<int> vect1;
    std::vector<int> vect2;
    std::vector<int> vect3;

    // Timing variables
    double startTime;
    double endTime;
    double staticTime;
    double dynamicTime;

    // Initialize vectors
    initVector(vect1, kSize, kValue1);
    initVector(vect2, kSize, kValue2);
    initVector(vect3, kSize, kValue3);

    /**
     * @section Scheduling Behaviour
     */
    std::cout << std::endl << kDoubleLine << "\nSCHEDULING BEHAVIOUR\n" << kDoubleLine << std::endl;

    // Display configurations
    std::cout << "Configuration\n" << kSingleLine << std::endl
        << "Number of threads: " << kNumThreads << std::endl
        << "Vector size: " << kSize << std::endl
        << "Vector1 value: " << kValue1 << std::endl
        << "Vector2 value: " << kValue2 << std::endl
        << "Vector3 value: " << kValue3 << std::endl;

    // Static Scheduling
    // Default chunk size
    std::cout << "\n[1.1] Static Scheduling - Default Chunk Size\n" << kSingleLine << std::endl;
    printTableHeader(kScheduleHeaders, kScheduleColWidths, 50);
    runSchedule(vect1, vect2, vect3, "static", kScheduleColWidths, kNumThreads);
    // Specified chunk size
    std::cout << "\n[1.2] Static Scheduling - Specified Chunk Size (" << kStaticChunkSize << ")\n" << kSingleLine << std::endl;
    printTableHeader(kScheduleHeaders, kScheduleColWidths, 50);
    runSchedule(vect1, vect2, vect3, "static", kScheduleColWidths, kNumThreads, kStaticChunkSize);

    // Dynamic schedule
    // Default chunk size
    std::cout << "\n[2.1] Dynamic Scheduling - Default Chunk Size\n" << kSingleLine << std::endl;
    printTableHeader(kScheduleHeaders, kScheduleColWidths, 50);
    runSchedule(vect1, vect2, vect3, "dynamic", kScheduleColWidths, kNumThreads);
    // Specified chunk size
    std::cout << "\n[2.2] Dynamic Scheduling - Specified Chunk Size (" << kDynamicChunkSize << ")\n" << kSingleLine << std::endl;
    printTableHeader(kScheduleHeaders, kScheduleColWidths, 50);
    runSchedule(vect1, vect2, vect3, "dynamic", kScheduleColWidths, kNumThreads, kDynamicChunkSize);

    /**
     * @section Performance Comparison
     */
    constexpr int kMaxSize = 1000000;
    constexpr int kStartSize = 10;
    constexpr int kSizeMultiplication = 10;
    constexpr int kTestCount = 100;

    std::cout << std::endl << kDoubleLine << "\nPERFORMANCE COMPARISON\n" << kDoubleLine << std::endl;

    // Display configurations
    std::cout << "Configuration\n" << kSingleLine << std::endl
        << "Number of threads: " << kNumThreads << std::endl
        << "Vector1 value: " << kValue1 << std::endl
        << "Vector2 value: " << kValue2 << std::endl
        << "Vector3 value: " << kValue3 << std::endl;

    // Balanced Workload per Iteration
    std::cout << "\n[1] Performance Over Increasing Iterations (Balanced)\n" << kSingleLine << kSingleLine << std::endl;
    printTableHeader(kPerformanceHeaders, kPerformanceColWidths, 100);

    // Conduct comparison over increasing vector sizes
    for (int i = kStartSize; i <= kMaxSize; i *= kSizeMultiplication) {
        const int kCurrentSize = i;
        double staticTime = 0;
        double dynamicTime = 0;

        // Initialize vectors
        initVector(vect1, kCurrentSize, kValue1);
        initVector(vect2, kCurrentSize, kValue2);
        initVector(vect3, kCurrentSize, kValue3);

        // Repeat test with current size for consistent average results
        for (int j = 0; j < kTestCount; j++) {
            // Measure execution time for static and dynamic scheduling
            staticTime += measureSchedule(vect1, vect2, vect3, "static", true, kNumThreads);
            dynamicTime += measureSchedule(vect1, vect2, vect3, "dynamic", true, kNumThreads);
        }

        printTableRow({std::to_string(i), std::to_string(staticTime), std::to_string(dynamicTime),
            std::to_string(staticTime / kTestCount), std::to_string(dynamicTime / kTestCount)}, kPerformanceColWidths);
    }

    // Imbalanced Workload per Iteration
    std::cout << "\n[2] Performance Over Increasing Iterations (Imbalanced)\n" << kSingleLine << kSingleLine << std::endl;
    printTableHeader(kPerformanceHeaders, kPerformanceColWidths, 100);

    // Conduct comparison over increasing vector sizes
    for (int i = kStartSize; i <= kMaxSize; i *= kSizeMultiplication) {
        const int kCurrentSize = i;
        double staticTime = 0;
        double dynamicTime = 0;

        // Initialize vectors
        initVector(vect1, kCurrentSize, kValue1);
        initVector(vect2, kCurrentSize, kValue2);
        initVector(vect3, kCurrentSize, kValue3);

        // Repeat test with current size for consistent average results
        for (int j = 0; j < kTestCount; j++) {
            // Measure execution time for static and dynamic scheduling
            staticTime += measureSchedule(vect1, vect2, vect3, "static", false, kNumThreads);
            dynamicTime += measureSchedule(vect1, vect2, vect3, "dynamic", false, kNumThreads);
        }

        printTableRow({std::to_string(i), std::to_string(staticTime), std::to_string(dynamicTime),
            std::to_string(staticTime / kTestCount), std::to_string(dynamicTime / kTestCount)}, kPerformanceColWidths);
    }

    return 0;
}