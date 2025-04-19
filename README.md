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
