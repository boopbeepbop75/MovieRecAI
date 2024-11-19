How to use:

1. If you want to download and preprocess the images, go to https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download 
and download the dataset into the data_raw folder, 

then run the Data_cleanup.py file

Alternatively, if you want to preprocess the images and train a model of your own, run the GNN_Training.py folder instead.

2. If you want to use the pretrained model, run the Model_Predict.py file. It will automatically open the prediction images dataset and run the model on 10 of them
note: If you would like to input your own images given a URL, change the 'which' variable on line 18 in Model_Predict.py from 'pred' to 'url'
note: If you want to train your own model, you can name it in the HyperParameters.py file (MODEL_NAME)

3. If you want to visualize how the images are segmented and turned into graphs, run the Data_Visualization.py file.

4. If you want to train your own model with new parameters, change the parameters for processing the images or for the model in HyperParameters.py
