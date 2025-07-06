#include <iostream>
#include <mpi.h>

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

    // Enforce that this program must run with 4 processes
    if (worldRank == kMasterRank && worldSize != 4) {
        std::cerr << "Error: this program must be run with 4 processes.\n";
        std::cerr << "Usage: mpirun -np 4 ./<program_name>\n\n";
        MPI_Abort(MPI_COMM_WORLD, 1);   // Stop the program for all processes
    }
    
    // Master process displays program info
    if (worldRank == kMasterRank) {
        std::cout << kDoubleLine << "\nMPI Task Distribution: Hello World (a)\n" << kDoubleLine << std::endl;
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