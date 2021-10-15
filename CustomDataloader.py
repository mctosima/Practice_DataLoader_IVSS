# Import the modules here!


"""
Inherit the proper class from PyTorch module.
"""
class CustomDataset:

    def __init__(self):
        # Load the paths to the images and labels into list/arrays in constructor here
        pass
    
    #Do NOT forget to override the correct methods!
    
    """
    The colours mapping are defined in `labels.py` source code!
    You can import it and use the Label class in this method!
    """
    def shows(self, image, label):
        # Complete the implementation.
        pass


if __name__ == "__main__":
    # Iterate over through the whole dataset, visualise image and colouful label, and save them to output directory!
    # Use `DataLoder` class from torch.utils.data module to load your `CustomDataset` class!
    #                   -> set batch_size to 1!
    pass
