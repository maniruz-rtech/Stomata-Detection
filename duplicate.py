# Python program to explain shutil.copy2() method

# importing os module
import os

# importing shutil module
import shutil
dest={}

# Source path
source = "/home/maniruz/Downloads/Stomata/output.tif"

# Destination path
destination = "/home/maniruz/Downloads/Stomata/output{:03d}.tif"

# Copy the content of
# source to destination
for i in range(100):
    dest = shutil.copy2(source, destination.format(i))

# List files and directories
# in "/home / User / Desktop"
print("After copying file:")
print(os.listdir(destination))

# Print path of newly
# created file
print("Destination path:", dest)

