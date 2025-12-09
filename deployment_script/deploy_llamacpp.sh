#!/bin/bash
# Allocate the node
# Can/Should we script this?
# Maybe use the script in with srun

# Print usage if arguments are not provided
usage() {
    echo "Usage: $0 <target_path> <version_tag>"
    echo
    echo "Example:"
    echo "  $0 /opt/llama.cpp b7310"
    exit 1
}

# Check arguments
if [ $# -ne 2 ]; then
    usage
fi

# Variables
TARGET_PATH="$1"
VERSION_TAG="$2"

# Ensure path exists
mkdir -p "$TARGET_PATH"

# Clone the repo
git clone https://github.com/ggerganov/llama.cpp.git "$TARGET_PATH"

# Checkout the requested version
cd "$TARGET_PATH"
git checkout "tags/$VERSION_TAG" -b "build_$VERSION_TAG"

echo "llama.cpp cloned to $TARGET_PATH at version $VERSION_TAG"É

# Load the necesarry modules
# This will only work on ORCA HPC cluster
module purge
module load openblas
module load cmake
module load cuda
	
# Build llama.cpp
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_CUDA=ON
cmake --build build --config Release
