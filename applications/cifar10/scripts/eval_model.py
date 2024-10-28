"""
Compute FID and IS from generated samples of a model
"""
import argparse
from pytorch_gan_metrics import (
    get_inception_score_and_fid_from_directory,
    get_inception_score_from_directory
)

parser = argparse.ArgumentParser()
parser.add_argument('images_fpath', type=str, help='The directory where the sampled images are saved')
args = parser.parse_args()

images_fpath = args.images_fpath
print(f'Images from {images_fpath}')
stats_fpath = '/data/gdd/cifar10/data/cifar10.train.npz'

(IS, IS_std), FID = get_inception_score_and_fid_from_directory(
    images_fpath, stats_fpath)
print(f'IS: {IS:.2f} ({IS_std:.2f}), FID: {FID:.2f}')