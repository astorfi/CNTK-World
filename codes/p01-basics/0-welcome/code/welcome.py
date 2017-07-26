# Import CNTK library
import cntk

# Version
print('CNTK version:', cntk.__version__)

# Check GPU
from cntk.device import try_set_default_device, gpu
if try_set_default_device(gpu(0)):
    print('GPU device is enabled!')

# Print devices
print('All available devices:', cntk.all_devices())

# Simple math operation
print('Welcome to CNTK world!')