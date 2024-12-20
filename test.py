# import torch

# # Check if CUDA is available
# print("Is CUDA available:", torch.cuda.is_available())

# # Check the number of GPUs
# print("Number of GPUs:", torch.cuda.device_count())

# # Get the current GPU index and name
# if torch.cuda.is_available():
#     print("Current GPU:", torch.cuda.current_device())
#     print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# else:
#     print("No GPU available.")

import torch




# Check if CUDA is available
print("Is CUDA available:", torch.cuda.is_available())

# Check the number of GPUs
print("Number of GPUs:", torch.cuda.device_count())

# Get the current GPU index and name
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("No GPU available.")