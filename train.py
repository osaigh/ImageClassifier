import model_utility as mu
import input_args as ia


def main():
    in_arg = ia.get_train_input_args()

    if in_arg.data_dir:
        save_dir = "NULL"
        if in_arg.save_dir:
            save_dir = in_arg.save_dir
        
        use_gpu = False
        if in_arg.gpu:
            use_gpu = True

        mu.Build_Train_Model(in_arg.data_dir, in_arg.arch, in_arg.learning_rate, in_arg.hidden_units, in_arg.epochs, use_gpu, save_dir)
    else:
        raise ValueError('data_dir not given')

if __name__ == "__main__":
    main()