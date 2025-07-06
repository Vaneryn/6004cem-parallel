#include <iostream>
#include <mpi.h>
#include <thread>

int main(int argc, char** argv) {
    // Console UI elements
    constexpr int kLineLength = 50;
    const std::string kSingleLine(kLineLength, '-');
    const std::string kDoubleLine(kLineLength, '=');

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Get the rank of the current process
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Get the name of the processor this process is running on
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(processorName, &nameLen);

    // Master process configurations
    constexpr int kMasterRank = 0;

    if (worldRank == kMasterRank) {
        const unsigned int numCores = std::thread::hardware_concurrency();
        
        // Display program configurations
        std::cout << kDoubleLine << "\nMPI Task Distribution: Hello World (b)\n" << kDoubleLine << std::endl;
        std::cout << "Configuration\n" << kSingleLine << std::endl
                << "Number of cores: " << numCores << std::endl
                << "Number of MPI processes: " << worldSize << std::endl;

        if (numCores > 0 && worldSize > static_cast<int>(numCores)) {
            std::cout << "\n- - - Warning: More processes than cores - expect performance impacts due to context switching - - -\n";
        } else {
            std::cout << "\n+ + + Good: Each process can independently run on a separate core + + +\n";
        }
        std::cout << std::endl;
    }

    // Synchronize processes before executing distributed work
    MPI_Barrier(MPI_COMM_WORLD);

    // Print a Hello World message from each process
    std::cout << "[Process " << worldRank
            << " - " << processorName << "]"
            << " Hello world\n";

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}