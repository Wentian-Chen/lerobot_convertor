import sys 
sys.path.append('../src')
from lerobot_convertor.utils import print_hdf5_structure
hdf5_file = sys.argv[1] if len(sys.argv) > 1 else "example.hdf5"
print_hdf5_structure(source=hdf5_file, max_depth=10, include_attrs=True)