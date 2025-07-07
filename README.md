# 6004CEM Parallel Distributed Programming - Coding Portfolio

**Course:** 6004CEM Parallel Distributed Programming  
**Assignment:** Coding Portfolio - OpenMP and MPI Implementation  
**Language:** C++  
**Semester:** April 2025

## Overview

This repository contains a comprehensive coding portfolio demonstrating parallel and distributed programming concepts using OpenMP and MPI. The implementation covers fundamental parallel programming patterns, process communication, and performance optimization techniques.

## Project Structure

```
6004cem-parallel/
├── openmp/                    # OpenMP implementations
│   ├── openmp_parta_helloworld1.cpp    # Fixed thread count
│   ├── openmp_parta_helloworld2.cpp    # Environment variable threads
│   ├── openmp_parta_helloworld3.cpp    # User input threads
│   ├── openmp_partb_schedule.cpp       # Scheduling comparison
│   └── openmp_partc_matrix.cpp         # Matrix multiplication
├── mpi/                      # MPI implementations
│   ├── mpi_parta_helloworld1.cpp       # Basic MPI hello world
│   ├── mpi_parta_helloworld2.cpp       # MPI with system info
│   ├── mpi_partb_slaves1.cpp           # Master-slave communication
│   ├── mpi_partb_slaves2.cpp           # Personalized messages
│   ├── mpi_partc_tag.cpp               # Message 
└── README.md                 # This file
```

## OpenMP Implementation

### Part A: Hello World Variations
- **`openmp_parta_helloworld1.cpp`**: Demonstrates fixed thread count (10 threads)
- **`openmp_parta_helloworld2.cpp`**: Uses environment variable `OMP_NUM_THREADS`
- **`openmp_parta_helloworld3.cpp`**: Interactive user input for thread count

### Part B: Scheduling Comparison
- **`openmp_partb_schedule.cpp`**: Compares static vs dynamic scheduling
- Features balanced and imbalanced workload testing
- Performance measurement and analysis

### Part C: Matrix Multiplication
- **`openmp_partc_matrix.cpp`**: Parallel matrix multiplication
- Compares outer loop vs inner loop parallelization
- Performance analysis with different thread counts

## MPI Implementation

### Part A: Hello World
- **`mpi_parta_helloworld1.cpp`**: Basic MPI process communication
- **`mpi_parta_helloworld2.cpp`**: Enhanced with system information

### Part B: Master-Slave Communication
- **`mpi_partb_slaves1.cpp`**: Basic master-slave pattern
- **`mpi_partb_slaves2.cpp`**: Personalized slave messages
- Demonstrates point-to-point communication

### Part C: Message Tagging
- **`mpi_partc_tag.cpp`**: Advanced message tagging system
- Shows selective message handling

## Key Features

### OpenMP Features
- ✅ Thread management and synchronization
- ✅ Work-sharing constructs (`#pragma omp parallel for`)
- ✅ Critical sections for thread-safe output
- ✅ Dynamic vs static scheduling comparison
- ✅ Performance measurement and analysis

### MPI Features
- ✅ Process initialization and finalization
- ✅ Point-to-point communication (`MPI_Send`/`MPI_Recv`)
- ✅ Process rank and size management
- ✅ Message tagging for selective communication
- ✅ Error handling and validation

## Compilation and Execution

### OpenMP Programs
```bash
# Compile with OpenMP support
g++ -fopenmp -o program_name source_file.cpp

# Set thread count via environment variable
export OMP_NUM_THREADS=4
./program_name

# Examples
g++ -fopenmp -o hello1 openmp_parta_helloworld1.cpp
g++ -fopenmp -o schedule openmp_partb_schedule.cpp
g++ -fopenmp -o matrix openmp_partc_matrix.cpp
```

### MPI Programs
```bash
# Compile with MPI wrapper
mpic++ -o program_name source_file.cpp

# Execute with specified process count
mpirun -np 4 ./program_name

# Examples
mpic++ -o hello_mpi mpi_parta_helloworld1.cpp
mpic++ -o master_slave mpi_partb_slaves1.cpp
mpirun -np 4 ./hello_mpi
mpirun -np 4 ./master_slave
```

## Prerequisites

- **OpenMP**: GCC with OpenMP support (`-fopenmp` flag)
- **MPI**: MPI implementation (MPICH, OpenMPI, or Intel MPI)
- **C++ Compiler**: Supporting C++11 or later
- **Operating System**: Linux, macOS, or Windows with appropriate MPI installation

## Author

Thor Wen Zheng
