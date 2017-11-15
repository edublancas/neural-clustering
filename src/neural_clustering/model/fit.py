import argparse


def fit():
    """
    Fit an Edward model. Receives a path to a npy file specifying
    the training data and a path to a configuration file
    """
    parser = argparse.ArgumentParser(description='Fit an Edward model')
    parser.add_argument('config', type=str,
                        help='Path to configuration file')
    parser.add_argument('training_data', type=str,
                        help='Path to training data')
    args = parser.parse_args()
