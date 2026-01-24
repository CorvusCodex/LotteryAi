#!/bin/bash
# =============================================================================
# LotteryAi - Ubuntu GPU Environment Setup Script
# =============================================================================
# This script sets up the environment for running LotteryAi with GPU support
# on Ubuntu (tested on Ubuntu 20.04, 22.04, and 24.04)
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "LotteryAi - Ubuntu GPU Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. It's recommended to run as a regular user with sudo access."
fi

# Step 1: Update system packages
print_status "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install essential build tools
print_status "Installing essential build tools..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg

# Step 3: Check for NVIDIA GPU
print_status "Checking for NVIDIA GPU..."
if lspci | grep -i nvidia > /dev/null 2>&1; then
    print_status "NVIDIA GPU detected!"
    GPU_DETECTED=true
else
    print_warning "No NVIDIA GPU detected. TensorFlow will run on CPU only."
    GPU_DETECTED=false
fi

# Step 4: Install NVIDIA drivers (if GPU detected)
if [ "$GPU_DETECTED" = true ]; then
    print_status "Installing NVIDIA drivers..."

    # Add NVIDIA PPA for latest drivers
    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt-get update

    # Install recommended NVIDIA driver
    sudo ubuntu-drivers install

    # Alternatively, install specific driver version (uncomment if needed):
    # sudo apt-get install -y nvidia-driver-535

    print_status "NVIDIA drivers installed. A reboot may be required."
fi

# Step 5: Install CUDA Toolkit (if GPU detected)
if [ "$GPU_DETECTED" = true ]; then
    print_status "Installing CUDA Toolkit..."

    # Check Ubuntu version
    UBUNTU_VERSION=$(lsb_release -rs)

    # Install CUDA toolkit from Ubuntu repositories
    sudo apt-get install -y nvidia-cuda-toolkit

    # Set CUDA environment variables
    echo "" >> ~/.bashrc
    echo "# CUDA Environment Variables" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

    # Source the updated bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    print_status "CUDA Toolkit installed."
fi

# Step 6: Install Python and pip
print_status "Installing Python and pip..."
sudo apt-get install -y python3 python3-pip python3-venv python3-dev

# Step 7: Create virtual environment
VENV_DIR="$HOME/lotteryai_venv"
print_status "Creating Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Step 8: Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 9: Install TensorFlow with GPU support
print_status "Installing TensorFlow and dependencies..."
if [ "$GPU_DETECTED" = true ]; then
    # Install TensorFlow with GPU support
    pip install tensorflow[and-cuda]
    print_status "TensorFlow installed with GPU support."
else
    # Install TensorFlow CPU-only
    pip install tensorflow
    print_status "TensorFlow installed (CPU only)."
fi

# Step 10: Install other required packages
print_status "Installing additional required packages..."
pip install numpy keras art

# Step 11: Verify installation
print_status "Verifying TensorFlow installation..."
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('GPUs available:', len(gpus))
    for gpu in gpus:
        print('  -', gpu)
else:
    print('No GPU detected by TensorFlow (will use CPU)')
"

# Step 12: Create a run script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_lotteryai.sh"
print_status "Creating run script at $RUN_SCRIPT..."

cat > "$RUN_SCRIPT" << 'RUNEOF'
#!/bin/bash
# Run script for LotteryAi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
VENV_DIR="$HOME/lotteryai_venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run setup_ubuntu_gpu.sh first."
    exit 1
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Run the program
echo "Starting LotteryAi..."
python3 LotteryAi.py

# Deactivate virtual environment
deactivate
RUNEOF

chmod +x "$RUN_SCRIPT"

# Print summary
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
print_status "Environment Summary:"
echo "  - Python virtual environment: $VENV_DIR"
echo "  - Run script: $RUN_SCRIPT"
if [ "$GPU_DETECTED" = true ]; then
    echo "  - GPU Support: Enabled"
    echo ""
    print_warning "IMPORTANT: You may need to reboot for NVIDIA drivers to take effect."
    echo "  After reboot, verify GPU with: nvidia-smi"
else
    echo "  - GPU Support: Disabled (no NVIDIA GPU detected)"
fi
echo ""
echo "To run LotteryAi:"
echo "  Option 1: ./run_lotteryai.sh"
echo "  Option 2: source $VENV_DIR/bin/activate && python3 LotteryAi.py"
echo ""
echo "=============================================="
