#!/bin/bash
# Test script for MPI exercises
# Run all exercises and display results

set -e  # Exit on error

echo "=========================================="
echo "MPI Matrix Multiplication Exercise Tests"
echo "=========================================="
echo ""

# Load required modules
echo "Loading toolchain/foss module..."
module load toolchain/foss 2>/dev/null || true

# Check if executables exist
echo "Checking for compiled executables..."
for exe in exercise_1_1 exercise_1_2 exercise_1_3 exercise_2_1 exercise_2_2; do
    if [ ! -f "$exe" ]; then
        echo "ERROR: $exe not found. Please compile first."
        exit 1
    fi
done
echo "✓ All executables found"
echo ""

# Test Sequential Exercises
echo "=========================================="
echo "SEQUENTIAL EXERCISES (1.1, 1.2, 1.3)"
echo "=========================================="
echo ""

echo "Running Exercise 1.1 (DAXPY and DDOT)..."
if ./exercise_1_1; then
    echo "✓ Exercise 1.1 PASSED"
else
    echo "✗ Exercise 1.1 FAILED"
fi
echo ""

echo "Running Exercise 1.2 (DGEMV)..."
if ./exercise_1_2; then
    echo "✓ Exercise 1.2 PASSED"
else
    echo "✗ Exercise 1.2 FAILED"
fi
echo ""

echo "Running Exercise 1.3 (DGEMM)..."
if ./exercise_1_3; then
    echo "✓ Exercise 1.3 PASSED"
else
    echo "✗ Exercise 1.3 FAILED"
fi
echo ""

# Test MPI Exercises
echo "=========================================="
echo "MPI EXERCISES (2.1, 2.2)"
echo "=========================================="
echo ""
echo "Testing with 4 MPI processes..."
echo ""

echo "Running Exercise 2.1 (Static Scattering)..."
if srun -n 4 ./exercise_2_1; then
    echo "✓ Exercise 2.1 PASSED"
else
    echo "✗ Exercise 2.1 FAILED"
fi
echo ""

echo "Running Exercise 2.2 (Cannon's Algorithm)..."
if srun -n 4 ./exercise_2_2; then
    echo "✓ Exercise 2.2 PASSED"
else
    echo "✗ Exercise 2.2 FAILED"
fi
echo ""

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "All tests completed. Check results above."
echo ""
