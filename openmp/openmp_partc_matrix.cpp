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

void initMatrix(std::mt19937& rng, std::uniform_int_distribution<int>& dist,
    Matrix& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(rng);
        }
    }
}

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

int main() {
    // Console UI elements
    constexpr int kLineLength = 50;
    const std::string kSingleLine(kLineLength, '-');
    const std::string kDoubleLine(kLineLength, '=');
    const std::vector<std::string> kTableHeaders = {"NumThreads", "Outer Total (s)", "Inner Total (s)", "Outer Average (s)", "Inner Average (s)"};
    const std::vector<int> kTableColWidths = {13, 18, 18, 20, 20};

    // RNG
    std::mt19937 rng(42);   // Mersenne Twister engine with fixed seed
    std::uniform_int_distribution<int> dist(1, 100);    // Random int from 1-100

    // Program configurations
    const std::vector<int> kMatrixSizeOptions = {50, 500};
    const std::vector<int> kNumThreadsOptions = {1, 4, 8, 16};
    constexpr int kTestCount = 100;

    // Display configurations
    std::cout << "Configuration\n" << kSingleLine;
    std::cout << "\nMatrix Size Options:";
    printVector(kMatrixSizeOptions);
    std::cout << "\nNumThreads Options:";
    printVector(kNumThreadsOptions);
    std::cout << "\nTest Count per NumThreads: " << kTestCount << std::endl;

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
        std::cout << "\n[" << i + 1 << "] " << kMatrixSize << "x" << kMatrixSize << " Matrix Multiplication\n" << kSingleLine << kSingleLine << std::endl;
        printTableHeader(kTableHeaders, kTableColWidths, 100);

        for (const int kNumThreads : kNumThreadsOptions) {
            double outerTotalTime = 0;
            double innerTotalTime = 0;

            for (int j = 0; j < kTestCount; ++j) {
                Matrix outerResult(kMatrixSize, std::vector<int>(kMatrixSize));
                Matrix innerResult(kMatrixSize, std::vector<int>(kMatrixSize));

                outerTotalTime += multiplyOuterParallel(matrix1, matrix2, outerResult, kMatrixSize, kNumThreads);
                innerTotalTime += multiplyInnerParallel(matrix1, matrix2, innerResult, kMatrixSize, kNumThreads);
            }

            printTableRow({std::to_string(kNumThreads), std::to_string(outerTotalTime), std::to_string(innerTotalTime),
                std::to_string((outerTotalTime / kTestCount)), std::to_string((innerTotalTime / kTestCount))}, kTableColWidths);
        }
    }

    return 0;
}