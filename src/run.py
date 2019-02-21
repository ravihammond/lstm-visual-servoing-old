#!/usr/bin/env python2
import argparse
import sys
import os

from pytorch import VisualServo

# Check validity of model directory
def check_model_directory(model_dir):
    model_dir = model_dir.strip('/') + '/'

    # Ensures training data directory exists
    if not os.path.exists(model_dir):
        sys.exit("Error: model directory %s does not exist")

    # model_path = os.path.join(model_dir, "model.pt")
    model_path = os.path.join(model_dir, "model_50.pt")

    # Ensures training data directory exists
    if not os.path.exists(model_path):
        sys.exit("Error: model directory %s does not contail model.py file")

    return model_path

# Main function
if __name__ == "__main__" :
    # Argument parser gets the model directory
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="name of directory to load model from")
    args = parser.parse_args()

    model_path = check_model_directory(args.model)

    visual_servo = VisualServo(model_path)
    visual_servo.run()

