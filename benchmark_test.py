# Run this line of code to make sure you have the necisary packages.
# If any of the below are missing uncomment and run the below line
# replacing PACKAGE with the needed package.
import torch
from torch import nn
import torch.utils.benchmark as benchmark
from torch.nn import functional as funct
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import densenet121, densenet161, densenet169
from torchvision.models import vgg11, vgg13, vgg16, vgg19

# Function to perform a basic operation on the GPU and return the result
def matrix_addition(dev):
    """Adds to matrixes together"""
    dev = torch.device(dev)
    
    # 10,000x10,000 randomly generate numbers
    a = torch.rand(5000, 5000).to(dev)
    b = torch.rand(5000, 5000).to(dev)

    result = torch.add(a, b).to(dev)

def matrix_multiplication(dev):
    """Multiplies two matrix together"""

    dev = torch.device(dev)
    
    a = torch.rand(5000, 5000).to(dev)
    b = torch.rand(5000, 5000).to(dev)
    
    result = torch.matmul(a, b).to(dev)
    
def convolution(dev):
    """Performs convultion to a using Kernal k"""

    dev = torch.device(dev)
    
    # 10,000x10,000 randomly generate numbers
    # 1 image, 3 for the RGB represenations,
    # and 10,000x10,000 pixels
    c = torch.rand(1, 3, 5000, 5000).to(dev)
    
    # 64, 3 RGB inputs, 3x3 kernel
    k = torch.rand(64, 3, 3, 3).to(dev)
    
    result = funct.conv2d(c, k).to(dev)

# Give device information
def gpu_cuda_info():
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Print the total number of GPUs detected
        gpu_count = torch.cuda.device_count()
        print(f'Total GPUs detected: {gpu_count}\n')
        # Create a list to append benchmark results
        bench =[]
        # Loop through all available GPUs, print their properties, perform computations, and gather results
        for i in range(gpu_count):
            device = torch.device(f'cuda:{i}')
            gpu_properties = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {gpu_properties.name}")
            print(f"  Total memory: {gpu_properties.total_memory / 1e9} GB")
            print(f"  CUDA Capability: {gpu_properties.major}.{gpu_properties.minor}")
            print(f"  MultiProcessor Count: {gpu_properties.multi_processor_count}")
            print(f'  Performing computation on Device {i}...\n')
            
            # Perform the basic operation on the GPU and append the result to the results list
            t1 = benchmark.Timer(stmt = 'matrix_addition(dev)',
                         setup='from __main__ import matrix_addition',
                         globals = {"dev": device},
                         label="Benchmarks",
                         sub_label=f"{device}",
                         description = "matrix_addition"
                        ).blocked_autorange(min_run_time=1)

            t2 = benchmark.Timer(stmt = 'matrix_multiplication(dev)',
                                 setup='from __main__ import matrix_multiplication',
                                 globals = {"dev": device},
                                 label="Benchmarks",
                                 sub_label=f"{device}",
                                 description = "matrix_multiplication"
                                ).blocked_autorange(min_run_time=1)
        
            t3 = benchmark.Timer(stmt = 'convolution(dev)',
                                 setup='from __main__ import convolution',
                                 globals = {"dev": device},
                                 label="Benchmarks",
                                 sub_label=f"{device}",
                                 description = "convolution"
                                ).blocked_autorange(min_run_time=1)

            # Append to list
            bench.append(t1)
            bench.append(t2)
            bench.append(t3)
            
        # Summarize and print the results from each GPU
        print('Summary of Results:')
        #Creating a table to compare results.
        compare = benchmark.Compare(bench)
        compare.trim_significant_figures()
        compare.colorize()
        compare.print()
 
    else:
        print("CUDA is not available. Please check your installation.")

# Function to generate fake image and label tensor data
def generate_data_and_labels(batch_size, num_channels, height, width, num_classes):
    images = torch.rand(batch_size, num_channels, height, width)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels

# Benchmark training
def train(model, data, target, criterion):

    # Sets model to training mode
    model.train()

    # Loops through trains model on the same data
    for step in range(NUM_TEST):
        model.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, target)
        loss.backward()

def intense_benchmark():

    MODEL_LIST = {
        "resnet18": resnet18(),
        "resnet34": resnet34(),
        "resnet50": resnet50(),
        "densenet121": densenet121(),
        "densenet161": densenet161(),
        "densenet169": densenet169(),
        "vgg11": vgg11(),
        "vgg13": vgg13(),
        "vgg16": vgg16(),
        "vgg19": vgg19()
    }
    
    # Benchmarking settings
    NUM_TEST = 50
    BATCH_SIZE = 20 # We are not actually creating batches we are just using the same data as the batch every time
    PRECISION = ['half', 'single', 'double']
    
    # Specifications for the synthetic data
    NUM_CHANNELS = 3
    HEIGHT = 224
    WIDTH = 224
    NUM_CLASSES = 50
    

    
    # This loop does a benchmark for each device present.
    for i in range(gpu_count+1):
        
        # Create a list to append benchmark results
        bench =[]
        
        # Sets Torch Device
        if i>gpu_count:
            dev = torch.device('cuda')
        else:
            dev = torch.device(f'cuda:{i}')
    
        # Loops through our precision type list
        for type in PRECISION:
    
            # Loops through each model in our list and trains it
            for model_name, model in MODEL_LIST.items():
    
                images, labels = generate_data_and_labels(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH, NUM_CLASSES)
    
                criterion = nn.CrossEntropyLoss()
    
                if type == 'double':
                    model=model.double()
                    images=images.double()
                    labels = labels.double()
                    criterion = criterion.double()
    
                elif type == 'single':
                    model=model.float()
                    images=images.float()
                    labels = labels.float()
                    criterion = criterion.float()
    
                elif type == 'half':
                    model=model.half()
                    images=images.half()
                    labels = labels.half()
                    criterion = criterion.half()
    
                # If we have multiple GPUs do model in parallel
                if i>gpu_count:
                    model = nn.DataParallel(model, device_ids=list(range(i-1)))
                else:
                    model=model.to(dev)
    
                images = images.to(dev)
                labels = labels.to(dev).long()
                criterion = criterion.to(dev)
    
                # Get the time of the trainning
                t = benchmark.Timer(stmt = 'train(model, data, target, criterion)',
                                 setup='from __main__ import train',
                                 globals = {'model': model, "data": images, "target": labels, "criterion": criterion},
                                 label=f"Benchmark for {dev}",
                                 sub_label=f"{model_name}",
                                 description = f"{type}"
                                ).blocked_autorange(min_run_time=1)
    
                bench.append(t)
    
                del images
                del labels
                del model
                del criterion
    
        #Creating a table to compare results.
        compare = benchmark.Compare(bench)
        compare.trim_significant_figures()
        compare.colorize()
        compare.print()
        print()


def main():
    # Call your functions here
    gpu_cuda_info()
    #intense_benchmark()

if __name__ == "__main__":
    main()