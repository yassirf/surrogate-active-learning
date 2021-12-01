import argparse
import utils.uncertainty as uncertainties

__all__ = ['get_args']


def get_args(model_names):
    # Basic parameters
    parser = get_init_args(model_names)

    # Get optimization args
    parser = get_optim_args(parser)

    # Get specific cyclic learning rate args
    parser = get_cyclic_lr_args(parser)

    # Get active learning args
    parser = get_active_learning_args(parser)

    return process_args(parser)


def get_init_args(model_names):
    parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10/100/ImageNet Training')

    # Datasets
    parser.add_argument('-d', '--dataset', default = 'cifar10', type = str)
    parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N', help = 'number of data loading workers (default: 4)')
    parser.add_argument('--augment', default = 0, type = int, metavar = 'Bool', help = 'use data augmentation')

    # Optimization options
    parser.add_argument('--epochs', default = 300, type = int, metavar = 'N', help = 'number of total epochs to run')
    parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N', help = 'manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default = 128, type = int, metavar = 'N', help = 'train batchsize')
    parser.add_argument('--test-batch', default = 100, type = int, metavar = 'N', help = 'test batchsize')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default = 'checkpoint', type = str, metavar = 'PATH', help = 'path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'path to latest checkpoint (default: none)')

    # Architecture
    parser.add_argument('--arch', '-a', metavar = 'ARCH', choices = model_names, help = 'model architecture: ' + ' | '.join(model_names))

    # Miscs
    parser.add_argument('--manual-seed', type=int, help='manual seed')
    parser.add_argument('--save-every', type=int, default=0, help='save every nth checkpoint')

    # Device options
    parser.add_argument('--use-gpu', default=1, type=int)
    return parser


def get_optim_args(parser):
    # Optimiser and learning rate
    parser.add_argument('--optim', '--optimizer', default='sgd', type=str, help='optimiser type')
    parser.add_argument('--lr-scheduler', default='multisteplr', type=str, help='type of learning rate schedule')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='decrease learning rate at these epochs.')
    parser.add_argument('--schedule-step', type=int, default=1, help='decrease learning at every number of steps')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    return parser


def get_cyclic_lr_args(parser):
    # Optimiser and learning rate specifically for cyclic learning rate
    parser.add_argument('--batch-lr-update', default=0, type=int, help="Whether to update learning rate per epoch or batch")
    parser.add_argument('--base-lr', default=0, type=float, help='Base learning rate for CyclicLR')
    parser.add_argument('--max-lr', default=0, type=float, help='Max learning rate for CyclicLR')
    parser.add_argument('--total-steps', default=0, type=int, help='Number of steps per period for CLR')
    return parser


def get_active_learning_args(parser):
    # Active learning parameters
    parser.add_argument('--selector', default='lc', type=str, help='Selection method and metric')
    parser.add_argument('--initial-size', default=0.05, type=float, help='Fraction of dataset as initial')
    parser.add_argument('--acquisition-size', default=0.05, type=float, help='Fraction of dataset as acquisition')
    parser.add_argument('--acquisition-iterations', default=4, type=int, help='Number of AL loops')
    return parser


def process_args(parser):
    # Create args class
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    return args, state