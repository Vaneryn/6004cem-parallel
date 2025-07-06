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

    // Get total number of processes and current process rank
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Master process configurations
    constexpr int kMasterRank = 0;

    // Enforce that this program must run with at least 2 processes (1 master + 1 slave)
    if (worldRank == kMasterRank && worldSize < 2) {
        std::cerr << "* * * Error: this program must be run with at least processes * * *\n";
        std::cerr << "* * * Usage: mpirun -np <number_of_processes> ./<program_name> * * *\n\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (worldRank == kMasterRank) {
        const unsigned int numCores = std::thread::hardware_concurrency();

        // Display program configurations
        std::cout << kDoubleLine << "\nMPI Master-Slave Communication: Point-to-Point(a)\n" << kDoubleLine << std::endl;
        std::cout << "Configuration\n" << kSingleLine << std::endl
                << "Number of cores: " << numCores << std::endl
                << "Number of MPI processes: " << worldSize << std::endl;

        std::cout << "\nMaster: Hello slaves give me your messages\n" << kSingleLine << std::endl;

        // Master process receives messages from slave processes
        for (int i = 1; i < worldSize; ++i) {
            int receivedRank;
            MPI_Recv(&receivedRank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Message received from process " << receivedRank << ": Hello back" << std::endl;
        }

        std::cout << kSingleLine << "\nMaster: All messages received from slave processes" << std::endl;
    } else {
        // Slave processes send message w/ rank to master process
        MPI_Send(&worldRank, 1, MPI_INT, kMasterRank, 0, MPI_COMM_WORLD);
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}