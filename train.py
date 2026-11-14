# USAGE
# python train.py

# Hyperparameter management
import argparse
import os

# import the necessary packages
# from networkx import config
from model.dataset import SegmentationDataset
from model.network import UNet
from model.network import HourGlass
from model.network import SimpleConv

from model import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import pandas as pd

from PyQt5.QtCore import QLibraryInfo
# from PySide2.QtCore import QLibraryInfo

def parse_args():
    parser = argparse.ArgumentParser(description="Train Image Segmentation Models")
    parser.add_argument("--model", type=str, choices=["unet", "hourglass", "simpleconv"], required=True)
    parser.add_argument("--features", type=int, default=64, help="Number of base features")
    parser.add_argument("--depth", type=int, default=4, help="Depth of encoding layers")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.model == "unet":
        model = UNet(num_features=args.features, depth_enc=args.depth)
        model_name = model.__class__.__name__
    elif args.model == "hourglass":
        model = HourGlass(num_features=args.features, depth_enc=args.depth)
        model_name = model.__class__.__name__
    else:
        model = SimpleConv()
        model_name = model.__class__.__name__

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    BASE_OUTPUT, BEST_MODEL_PATH, PLOT_PATH, PLOT_PREDICT_PATH, TEST_PATH, VAL_PATH, TRAIN_PATH = config.get_paths_for_experiment(model_name, args.features, args.depth)
    if(not(os.path.exists(BASE_OUTPUT))):
        os.mkdir(BASE_OUTPUT)
        
    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(config.IMAGES_PATH)))
    maskPaths = sorted(list(paths.list_images(config.MASKS_PATH)))

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths,
                             test_size=config.TEST_SPLIT, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    print("[INFO] saving testing image paths...")
    f = open(TEST_PATH,"w")
    f.write("\n".join(testImages))
    f.close()

    # define transformations (renamed to avoid shadowing the torchvision.transforms module)
    data_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),transforms.ToTensor()])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,transforms=data_transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,transforms=data_transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,num_workers=os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,num_workers=os.cpu_count())

    # initialize the UNet or HourGlass model
    # model = SimpleConv((3,16),(16,1))
    # model = HourGlass((3,16,32,64),(64,32,16,1))
    # model = UNet((3,16,32,64),(64,32,16,1)) # <-- debería dejarse así?
    # model = SimpleConv()
    # model_name = model.__class__.__name__
    cnnet = model.to(config.DEVICE)
    print(model)

    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(cnnet.parameters(), lr=config.INIT_LR)

    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        cnnet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # perform a forward pass and calculate the training loss
            pred = cnnet(x)
            loss = lossFunc(pred, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # use .item() to extract scalar and avoid retaining the computation graph in GPU memory
            totalTrainLoss += loss.item()

        # switch off autograd
        with torch.no_grad():
           # set the model in evaluation mode
           cnnet.eval()
           #
           # loop over the validation set
           for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                #
                # make the predictions and calculate the validation loss
                pred = cnnet(x)
                totalTestLoss += lossFunc(pred, y).item()
        #
        # calculate the average training and validation loss (after full loop)
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        
        # update our training history
        # avgTrainLoss/avgTestLoss are now plain Python floats (from .item()), no need to convert
        H["train_loss"].append(avgTrainLoss)
        H["test_loss"].append(avgTestLoss)           

    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H['train_loss'],'o', ls='-', ms=4, markevery=None, linewidth=2.0,label='train loss')
    plt.plot(H['test_loss'],'o', ls='-', ms=4, markevery=None, linewidth=2.0,label='test loss')
    plt.title("training loss on dataset")
    plt.xlabel("epoch #")
    plt.ylabel("loss")
    plt.legend()
    
    plt.savefig(PLOT_PATH)
    plt.show()

    # save history to csv
    df = pd.DataFrame(H)
    df.to_csv(os.path.join(BASE_OUTPUT,f"training_history_{model_name}.csv"), index=False)

    # serialize the model to disk
    torch.save(cnnet, BEST_MODEL_PATH)
