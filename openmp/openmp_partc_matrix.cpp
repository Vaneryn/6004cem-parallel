#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<int>>;

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
 * @brief Prints a grouped table header with main groups and sub-headers.
 * 
 * Creates a two-tier header system with grouped columns for better organization.
 * The first row shows the main group headers, and the second row shows sub-headers.
 * 
 * @param groupHeaders Vector of main group header strings.
 * @param groupWidths Vector of total widths for each group.
 * @param subHeaders Vector of sub-header strings.
 * @param subWidths Vector of widths for each sub-header.
 * @param lineLength The total length of the line separator.
 */
void printGroupedTableHeader(const std::vector<std::string>& groupHeaders, 
                            const std::vector<int>& groupWidths,
                            const std::vector<std::string>& subHeaders, 
                            const std::vector<int>& subWidths, 
                            int lineLength) {
    // Print main group headers
    for (size_t i = 0; i < groupHeaders.size(); ++i)
        std::cout << std::left << std::setw(groupWidths[i]) << groupHeaders[i];
    std::cout << "\n";

    // Print sub-headers
    for (size_t i = 0; i < subHeaders.size(); ++i)
        std::cout << std::left << std::setw(subWidths[i]) << subHeaders[i];
        std::cout << "\n" << std::string(lineLength, '-') << "\n";
    }

/**
 * @brief Prints a single row in a formatted table.
 * 
 * @param values A vector of strings representing the values in the row.
 * @param widths vector of column widths corresponding to each value.
 */
void printTableRow(const std::vector<std::string>& values, const std::vector<int>& widths) {
    for (size_t i = 0; i < values.size(); ++i)
        std::cout << std::left << std::setw(widths[i]) << values[i];
    std::cout << "\n";
}

/**
 * @brief Prints the contents of a vector.
 * 
 * Used for debugging or visual inspection of a vector's contents.
 * 
 * @param vect The vector to print.
 */
void printVector(const std::vector<int>& vect) {
    for (int num : vect) {
        std::cout << " " << num;
    }
}

/**
 * @brief Initializes a matrix with random integer values.
 * 
 * Fills the provided matrix with random integers using the given random number
 * generator and distribution. The matrix is filled row by row.
 * 
 * @param rng Reference to a Mersenne Twister random number generator.
 * @param dist Reference to a uniform integer distribution for generating values.
 * @param matrix Reference to the matrix to be initialized.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
void initMatrix(std::mt19937& rng, std::uniform_int_distribution<int>& dist,
    Matrix& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(rng);
        }
    }
}

/**
 * @brief Performs matrix multiplication using OpenMP with outer loop parallelization.
 * 
 * Multiplies two square matrices using the standard matrix multiplication algorithm
 * with OpenMP parallelization applied to the outermost loop (i-loop). This approach
 * distributes rows of the result matrix across different threads.
 * 
 * @param matrix1 First input matrix (left operand).
 * @param matrix2 Second input matrix (right operand).
 * @param resultMatrix Reference to the output matrix where results are stored.
 * @param size Dimension of the square matrices (size x size).
 * @param numThreads Number of OpenMP threads to use for parallelization.
 * @return Execution time in seconds.
 */
double multiplyOuterParallel(const Matrix& matrix1, const Matrix& matrix2, Matrix& resultMatrix, int size, int numThreads) {
    double startTime;
    double endTime;

    startTime = omp_get_wtime();

    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    endTime = omp_get_wtime();

    return (endTime - startTime);
}

/**
 * @brief Performs matrix multiplication using OpenMP with inner loop parallelization.
 * 
 * Multiplies two square matrices using the standard matrix multiplication algorithm
 * with OpenMP parallelization applied to the middle loop (j-loop). This approach
 * distributes columns of each row across different threads, with the outer loop
 * remaining sequential.
 * 
 * @param matrix1 First input matrix (left operand).
 * @param matrix2 Second input matrix (right operand).
 * @param resultMatrix Reference to the output matrix where results are stored.
 * @param size Dimension of the square matrices (size x size).
 * @param numThreads Number of OpenMP threads to use for parallelization.
 * @return Execution time in seconds.
 */
double multiplyInnerParallel(const Matrix& matrix1, const Matrix& matrix2, Matrix& resultMatrix, int size, int numThreads) {
    double startTime;
    double endTime;

    startTime = omp_get_wtime();

    for (int i = 0; i < size; ++i) {
        #pragma omp parallel for num_threads(numThreads)
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    endTime = omp_get_wtime();

    return (endTime - startTime);
}

/**
 * @brief Performs matrix multiplication using OpenMP with collapsed loop parallelization.
 * 
 * Multiplies two square matrices using the standard matrix multiplication algorithm
 * with OpenMP parallelization applied using the collapse(2) clause. This approach
 * combines the outer two loops (i and j loops) into a single parallel iteration
 * space, potentially providing better load balancing across threads.
 * 
 * @param matrix1 First input matrix (left operand).
 * @param matrix2 Second input matrix (right operand).
 * @param resultMatrix Reference to the output matrix where results are stored.
 * @param size Dimension of the square matrices (size x size).
 * @param numThreads Number of OpenMP threads to use for parallelization.
 * @return Execution time in seconds.
 */
double multiplyCollapseParallel(const Matrix& matrix1, const Matrix& matrix2, Matrix& resultMatrix, int size, int numThreads) {
    double startTime;
    double endTime;

    startTime = omp_get_wtime();

    #pragma omp parallel for collapse(2) num_threads(numThreads)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    endTime = omp_get_wtime();

    return (endTime - startTime);
}

int main() {
    // Console UI elements
    constexpr int kLineLength = 80;
    const std::string kSingleLine(kLineLength, '-');
    const std::string kDoubleLine(kLineLength, '=');
    const std::vector<std::string> kGroupHeaders = {"", "Total Time (s)", "Average Time (s)"};
    const std::vector<int> kGroupWidths = {12, 36, 36};
    const std::vector<std::string> kSubHeaders = {"NumThreads", "Outer", "Inner", "Collapse", "Outer", "Inner", "Collapse"};
    const std::vector<int> kSubWidths = {12, 12, 12, 12, 12, 12, 12};

    // RNG
    std::mt19937 rng(42);   // Mersenne Twister engine with fixed seed
    std::uniform_int_distribution<int> dist(1, 100);    // Random int from 1-100

    // Program configurations
    const std::vector<int> kMatrixSizeOptions = {50, 500};
    const std::vector<int> kNumThreadsOptions = {1, 4, 8, 16};
    constexpr int kTestRuns = 10;

    // Display configurations
    std::cout << "Configuration\n" << kSingleLine;
    std::cout << "\nMatrix Size Options:";
    printVector(kMatrixSizeOptions);
    std::cout << "\nNumThreads Options:";
    printVector(kNumThreadsOptions);
    std::cout << "\nTest Runs per NumThreads: " << kTestRuns << std::endl;

    // Experiment with all required matrix sizes
    for (int i = 0; i < kMatrixSizeOptions.size(); ++i) {
        const int kMatrixSize = kMatrixSizeOptions[i];

        // Allocate matrices
        Matrix matrix1(kMatrixSize, std::vector<int>(kMatrixSize));
        Matrix matrix2(kMatrixSize, std::vector<int>(kMatrixSize));

        // Initialize matrices
        initMatrix(rng, dist, matrix1, kMatrixSize, kMatrixSize);
        initMatrix(rng, dist, matrix2, kMatrixSize, kMatrixSize);

        // Experiment with all required number of threads
        std::cout << "\n[" << i + 1 << "] " << kMatrixSize << "x" << kMatrixSize << " Matrix Multiplication\n" << kSingleLine << std::endl;
        printGroupedTableHeader(kGroupHeaders, kGroupWidths, kSubHeaders, kSubWidths, kLineLength);

        for (const int kNumThreads : kNumThreadsOptions) {
            double outerTotalTime = 0;
            double innerTotalTime = 0;
            double collapseTotalTime = 0;

            // Repeat test to get average result
            for (int j = 0; j < kTestRuns; ++j) {
                Matrix outerResult(kMatrixSize, std::vector<int>(kMatrixSize));
                Matrix innerResult(kMatrixSize, std::vector<int>(kMatrixSize));
                Matrix collapseResult(kMatrixSize, std::vector<int>(kMatrixSize));

                outerTotalTime += multiplyOuterParallel(matrix1, matrix2, outerResult, kMatrixSize, kNumThreads);
                innerTotalTime += multiplyInnerParallel(matrix1, matrix2, innerResult, kMatrixSize, kNumThreads);
                collapseTotalTime += multiplyCollapseParallel(matrix1, matrix2, collapseResult, kMatrixSize, kNumThreads);
            }

            printTableRow({std::to_string(kNumThreads),
                std::to_string(outerTotalTime), std::to_string(innerTotalTime), std::to_string(collapseTotalTime),
                std::to_string((outerTotalTime / kTestRuns)), std::to_string((innerTotalTime / kTestRuns)), std::to_string((collapseTotalTime / kTestRuns))},
                kSubWidths);
        }
    }

    return 0;
}