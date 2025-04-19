# DA6401 Assignment 02 (P2) (AE21B105)
  This part of the assignment is to fine tune a pre-existing model to our image net classification and report the results.

## Link to WandB Report
WandB link : https://api.wandb.ai/links/A2_DA6401_DL/t7oiddqh
## Requirements
- argparse
- scikit-learn
- pandas
- numpy
- matplotlib
- pytorch, torchvision, torchmetrics etc
- lightning

# Contents of the Repository
The ```Utils.py``` file contains the basic function and methods required for running the fine tuning operations such as lightning wrapper, dataset augmenter, dataloader definition etc. When running the scrips do take care of the paths were the dataset folders are present. The other notebook files as the name suggests were pre-trained weights and finetuning of the model with the Efficient-Net being the best among those.

The path configuration is as follows,

For Training and Validation part...

![Screenshot From 2025-04-19 22-58-16](https://github.com/user-attachments/assets/f727079a-0283-40ec-9000-ee7c7fd75414)

Here in the path_ we will have to give the folder containing the train portion of the dataset

For Testing...

![Screenshot From 2025-04-19 23-00-03](https://github.com/user-attachments/assets/920e762a-abc3-47e3-b197-263a584561a2)

Here in the path_ we will have to give the folder containing the val portion of the dataset (that portion of the dataset is the testing dataset as mention in the question)

## Training and evaluvation of the model
Now the main part of the code is as follows, (The image is taken from the previous part of the question part-A but the process is the same)

![Screenshot From 2025-04-19 22-38-57](https://github.com/user-attachments/assets/76076414-bc73-40df-8542-cdfd8ca9a6c1)

- First two lines is getting the dataloaders (train and val) ready, we pass the path_ of the train folder of the dataset as the argument and the input size as seen in the screenshot of the code. The first line returns the datasets. The second function takes the datasets and the transformations returned from the first fucntion and some other required parameters to build the dataloaders.
- In the next two lines the callbacks for early stopping and saving checkpoints is defined
- Then the lightning module is defined with the convolutional model being defined inside the lightning module and the trainer is trainer for the given configurations for a given number of epochs. The validation and training losses and accuracies are logged in the wandb.
- Then the last 4 lines of the code are simple for testing the model, load the best model, create the test loader by passing the path_ for the val folder of the dataset, then with the trainer defined for the loaded model uses test() method to evaluvate the model on the test dataset.
