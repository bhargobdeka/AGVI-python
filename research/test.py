import sys
import os

## Get the path to the current directory
current_directory = os.getcwd()
# print("Current directory:",current_directory)

## Added the src folder to the path
os.chdir(os.path.join(current_directory, "src"))
new_directory = os.getcwd()
# print("New directory:",new_directory)
agvi_python_path = os.path.abspath(new_directory)  # Replace with the actual path
# print("Current path:",agvi_python_path)

## Add the 'AGVI_python' directory to sys.path
sys.path.append(agvi_python_path)

## if want to see the paths
# for path in sys.path:
#     print(path)

## test to see if it works
from AGVIpy.utils.gma import cov1234


## if want to see the modules
# import sys

# for module_name, module in sys.modules.items():
#     print(module_name)


