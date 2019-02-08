from __future__ import print_function
import argparse
import os
import sys
import csv

from utils import query_yes_no
from pytorch import TrainManager

# Check validity of training data
def check_training_directory(training_dir):
    training_dir = training_dir.strip('/') + '/'

    # Ensures training data directory exists
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
        return training_dir

    # Checks that the training data is not empty
    dir_files = os.listdir(training_dir)
    if len(dir_files) is 0:
        sys.exit("Error: Directory %s is empty" % training_dir)


    # Find all valid sequences
    valid_sequences = []
    for i, seq_filename in enumerate(dir_files):
        seq_path = os.path.join(training_dir, seq_filename)
        if not os.path.isdir(seq_path):
            continue

        seq_files = os.listdir(seq_path)

        # Check for velocities.csv file
        if not "velocities.csv" in seq_files:
            print("Sequence %s does not contain velocities.csv file, ignoring" % seq_filename)
            continue

        # Count number of recorded velocities
        vel_count = 0
        with open(os.path.join(seq_path, "velocities.csv"), "r") as f:
            reader = csv.reader(f, delimiter = ",")
            data = list(reader)
            vel_count = len(data) - 1
        
        # Check for validity of frames
        frame_count = 0
        for filename in seq_files:
            if not filename.endswith(".png"):
                continue
            frame_count += 1

        # Skip if the number of .png files doesn't match the amount in velocities.csv
        if vel_count != frame_count:
            print("Sequence %s image count (%d) does not equal its velocity count (%d), ignoring" % (
                seq_filename, frame_count, vel_count))
            continue

        valid_sequences.append(seq_path)

    # Exit if the training directory contains no valid sequences
    if len(valid_sequences) is 0:
        sys.exit("Error: no valid sequences in directory")

    return training_dir, valid_sequences

# Check validity of model directory
def check_model_directory(model_dir):
    model_dir = model_dir.strip('/') + '/'

    # Ensures training data directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return model_dir

    # Checks that the training data is empty
    if len(os.listdir(model_dir)) > 0:
        overrite_query = "Directory: " + model_dir + " already contains a model, would you like to exit?"

        # Model already exists, ask if user wants to exit
        if not query_yes_no(overrite_query):
            continue_query = "Would you like to continue training from last model?"

            # Ask if user wants to continue training, or clear it
            if query_yes_no(continue_query):
                files = os.listdir(model_dir)
                if not "model.pt" in files:
                    sys.exit("Model %s does not contain model.pt file, cannot continue training")
                elif not "last_model.pt" in files:
                    sys.exit("Model %s does not contain last_model.pt file, cannot continue training")

            # Clear existing model to train again
            else:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir)

        else:
            sys.exit()

    return model_dir

# Main function
if __name__ == "__main__" :
    # Argument parsing gets image directory, training data directory, and window size
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="name of directory to load training data from")
    parser.add_argument("models", help="name of directory to save model into")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("-s", "--split", dest="split", type=float, default=0.2, help="percentage of data for validation and test set")
    args = parser.parse_args()

    # Check validity of image and training data directories/files
    training_dir, valid_sequences = check_training_directory(args.training)
    # Check to see if model directory exists
    models_dir = check_model_directory(args.models)

    #Create trainer, load the data, and train the model
    train_manager = TrainManager(training_dir, models_dir, args.split, args.epochs, valid_sequences)
    train_manager.load_data()
    # train_manager.train_model()

