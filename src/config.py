
"""Input arguments
"""
import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Enhancing Recipe Retrieval with Foundation Models')

    # paths & logging
    parser.add_argument('--save_dir', type=str,
                        required=True,
                        help='Path to store checkpoints.')
    parser.add_argument('--root', type=str,
                        required=True,
                        help='Dataset path.')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name (used to store checkpoint files under path save_dir/model_name).')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Logging frequency (in iterations).')
    parser.add_argument('--resume_from', type=str, default='',
                        help='Model name to load.')
    # training
    parser.add_argument('--scheduler_name', type=str,
                        default='StepLR',
                        help='Learning rate scheduler',
                        choices=['ReduceLROnPlateau', 'ExponentialLR', 'StepLR'])
    parser.add_argument('--es_metric', type=str, default='recall_1',
                        choices=['loss', 'medr', 'recall_1',
                                 'recall_5',
                                 'recall_10'],
                        help='Early stopping metric to monitor during training.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Maximum number of epochs.')
    parser.add_argument('--tf_n_heads', type=int, default=4,
                        help='Number of attention heads in Transformer models.')
    parser.add_argument('--tf_n_layers', type=int, default=2,
                        help='Number of layers in Transformer models.')
    parser.add_argument('--hidden_recipe', type=int, default=512,
                        help='Embedding dimensionality for recipe representation.')
    parser.add_argument('--output_size', type=int, default=512,
                        help='Dimensionality of the output embeddings.')
    parser.add_argument('--imsize', type=int, default=224,
                        help='Image size (for center/random cropping)')
    parser.add_argument('--resize', type=int, default=256, help='Image size (for resizing)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--scale_lr', type=float, default=1.0,
                        help='Learning rate multiplier for the image backbone.')
    parser.add_argument('--relaxation_factor', type=float, default=0.25,
                        help='Value of the relaxation factor m for the circle loss.')
    parser.add_argument('--scale_factor', type=int, default=32,
                        help='Value of the scale factor gamma for the circle loss.')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='Weight decay.')
    parser.add_argument('--recipe_loss_weight', type=float, default=1.0,
                        help='Weight value for the loss computed on recipe-only samples.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--patience', type=int, default=-1,
                        help='Maximum number of epochs to allow before early \
                            stopping (-1 will train forever).')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                        help='Learning rate decay factor.')
    parser.add_argument('--lr_decay_patience', type=int, default=30,
                        help='Number of epochs with no improvement to wait in order to reduce learning rate.')


    # flags - training
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
                        help='Enables tensorboard logging (otherwise stdout logging is used).')
    parser.set_defaults(tensorboard=False)
    parser.add_argument('--load_optimizer', dest='load_optimizer', action='store_true',
                        help='Loads optimizer state dict when resuming.')
    parser.set_defaults(load_optimizer=True)
   

    # testing
    parser.add_argument('--eval_split', type=str, default='test',
                        help='Split to extract features for when testing.',
                        choices=['train', 'val', 'test'])
    

    args = parser.parse_args()

    return args


def get_preprocessing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_freq', type=int, default=10,
                        help='Minimum number of occurrences required to keep a word in the dictionary.')
    parser.add_argument('--root', type=str, required=True, help='Path to dataset.')
    parser.add_argument('--force', dest='force', action='store_true',
                        help='Re-compute word counter even if it exists.')
    parser.set_defaults(force=False)

    args = parser.parse_args()
    return args


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str, required=True,
                        help='Full path to embeddings file.')
    parser.add_argument('--retrieval_mode', type=str,
                        default='image2recipe',
                        help='Retrieval mode for evaluation.',
                        choices=['image2recipe', 'recipe2image'])
    parser.add_argument('--medr_N', type=int, default=1000,
                        help='Ranking size to compute median rank.')
    parser.add_argument('--ntimes', type=int, default=10,
                        help='Number of test sample sets to evaluate.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_eval_args()
    print(args)