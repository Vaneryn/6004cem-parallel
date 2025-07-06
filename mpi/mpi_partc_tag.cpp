#include <iostream>
#include <mpi.h>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    // Console UI elements
    constexpr int kLineLength = 60;
    const std::string kSingleLine(kLineLength, '-');
    const std::string kDoubleLine(kLineLength, '=');

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get total number of processes and current process rank
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Main program configurations
    constexpr int kMasterRank = 0;
    constexpr int kMaxMessageLength = 100;
    constexpr int kMasterTag = 100;
    constexpr int kSlaveWaitTag = 101;

    // Enforce that this program must run with at least 2 processes (1 master + 1 slave)
    if (worldRank == kMasterRank && worldSize < 2) {
        std::cerr << "* * * Error: this program must be run with at least processes * * *\n";
        std::cerr << "* * * Usage: mpirun -np <number_of_processes> ./<program_name> * * *\n\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (worldRank == kMasterRank) {
        const unsigned int numCores = std::thread::hardware_concurrency();
        std::string messageTemplate = "Hello, ";
        std::string recipientName;
        std::string finalMessage;

        // Display program configurations
        std::cout << kDoubleLine << "\nMPI Master-Slave Communication: Tags\n" << kDoubleLine << std::endl;
        std::cout << "Configuration\n" << kSingleLine << std::endl
                << "Number of cores: " << numCores << std::endl
                << "Number of MPI processes: " << worldSize << std::endl
                << "Master Tag: " << kMasterTag << std::endl
                << "Slave Wait Tag: " << kSlaveWaitTag << std::endl << std::endl;
        
        // Master process sends custom messages to each slave process
        for (int destRank = 1; destRank < worldSize; ++destRank) {
            switch (destRank) {
                case 1: recipientName = "John"; break;
                case 2: recipientName = "Mary"; break;
                case 3: recipientName = "Susan"; break;
                default: recipientName = "unnamed process"; break;
            }
            finalMessage = messageTemplate + recipientName;

            std::cout << "[Master] Sending to Process " << destRank << " with tag " << kMasterTag << ": " << finalMessage << std::endl;

            MPI_Send(finalMessage.c_str(), finalMessage.size() + 1, MPI_CHAR, destRank, kMasterTag, MPI_COMM_WORLD);
        }
    } else {
        // Slave process configurations
        char recvBuffer[kMaxMessageLength];
        MPI_Status status;

        std::cout << "[Process " << worldRank << "] Waiting to receive message with tag " << kSlaveWaitTag << "...\n";

        // Slave processes receive message with the tag
        MPI_Recv(recvBuffer, kMaxMessageLength, MPI_CHAR, kMasterRank, kSlaveWaitTag, MPI_COMM_WORLD, &status);

        std::cout << "[Process " << worldRank << "] Received from master (actual tag " << status.MPI_TAG << "): " << recvBuffer << std::endl;
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}