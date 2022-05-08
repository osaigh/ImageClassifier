import argparse

def get_train_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir",type=str, help="Directory contain Train, Validation and Test data")
    parser.add_argument("--save_dir",type=str,help="Directory where model checkpoint should be saved")
    parser.add_argument("--arch",type=str, default="vgg19",help="Model architecture")
    parser.add_argument("--learning_rate",type=float, default=0.003,help="Learning rate")
    parser.add_argument("--hidden_units",type=int, default=512,help="Hidden Units")
    parser.add_argument("--epochs",type=int, default=30,help="Number of epochs to use for training the model")
    parser.add_argument("--gpu", help="Use GPU is available",action="store_true")
    
    
    return parser.parse_args()

def get_predict_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("input",type=str, help="path to image")
    parser.add_argument("checkpoint",type=str, help="path to checkpoint file")
    parser.add_argument("--top_k",type=int,help="Return top K most likely classes")
    parser.add_argument("--category_names",type=str,help="Path to file containing mapping of categories to real names")
    parser.add_argument("--gpu", help="Use GPU is available",action="store_true")
    
    
    return parser.parse_args()