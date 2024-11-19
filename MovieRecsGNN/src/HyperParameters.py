import torch

### Graph Hyperparameters ###
FREEDOM = 3 #Similar to MIN_CONNECTION_THRESHOLD, try to have minimum 3 connections per node
THRESHOLD = .5 #Minimum cosine similarity of THIS PERCENTAGE to create a connection
TRAIN_SPLIT = .8
PRUNE_RATIO = 10 #1 node to 10 edges
MIN_CONNECTION_THRESHOLD = 3 #Don't prune edges if it would leave nodes with less than THIS number of edges
MAX_REMOVAL_THRESHOLD = .85 #Don't prune edges with a cosine similarity score higher than THIS percentage

### MODEL HYPER PARAMETERS ###
#Model name
MODEL_NAME = 'Model_0'

GENRES = ['Romance', 'Animation', 'Documentary', "Children's", 'Drama', 'Action', 'Thriller', 'Comedy', 'War', 'Mystery', 'Crime', 'Film-Noir', 'Horror', 'Western', 'Sci-Fi', 'Fantasy', 'Adventure', 'Musical']
CLASSES = []
BATCH_SIZE = 32
DROPOUT_RATE = .5
HIDDEN_UNITS = 40
NUM_HEADS = 16
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = .0002
EPOCHS = 200
PATIENCE = 10  # Number of epochs to wait before early stopping
input_dim=3+1+1+1 #Color(3), eccentricity(1), aspect_ratio(1), solidity(1)

#Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"