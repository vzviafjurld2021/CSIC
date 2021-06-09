import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')


    parser.add_argument('--logname',default='',type=str, required=True, help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--lambda_1', default=0.3, type=float, help='(default=%(default)f)')
    parser.add_argument('--lambda_2', default=1e-2, type=float, help='(default=%(default)f)')
    parser.add_argument('--n_gpus',default=2,type=int)
    args=parser.parse_args()
    return args