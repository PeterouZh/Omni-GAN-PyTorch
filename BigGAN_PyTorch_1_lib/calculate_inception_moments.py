''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the 
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.
 
 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import logging
import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser

from template_lib.v2.config import update_parser_defaults_from_yaml
from template_lib.v2.config import get_dict_str, global_cfg


def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)') 
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)') 
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser

def run(config):
  logger = logging.getLogger('tl')

  saved_inception_moments = global_cfg.saved_inception_moments.format(config['dataset'])

  # Get loader
  config['drop_last'] = False
  loaders = utils.get_data_loaders(use_data_root=True, **config)

  # Load inception net
  net = inception_utils.load_inception_net(parallel=config['parallel'])
  net.eval()
  # net.train()
  pool, logits, labels = [], [], []
  device = 'cuda'
  pbar = tqdm(loaders[0], desc='accumulate pool and logits')
  for i, (x, y) in enumerate(pbar):
    x = x.to(device)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(y.cpu())]

  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  # uncomment to save pool, logits, and labels to disk
  # print('Saving pool, logits, and labels to disk...')
  # np.savez(config['dataset']+'_inception_activations.npz',
  #           {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  logger.info('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  logger.info('Training data from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean, IS_std))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default 
  # (the FID code also knows to strip "hdf5")
  logger.info('Calculating means and covariances...')
  mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
  logger.info('Saving calculated means and covariances to disk...')

  logger.info(f'Save to {saved_inception_moments}')
  os.makedirs(os.path.dirname(saved_inception_moments), exist_ok=True)
  np.savez(saved_inception_moments, **{'mu' : mu, 'sigma' : sigma})
  pass


def main():
  # parse command line    
  parser = prepare_parser()

  update_parser_defaults_from_yaml(parser)

  config = vars(parser.parse_args())
  print(get_dict_str(config))
  run(config)


def run1(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  # prepare_dataset(myargs.config.dataset)

  parser = prepare_parser()
  args = parser.parse_args([])
  args = config2args(myargs.config, args)

  args.data_root = os.path.expanduser(args.data_root)

  print(args)
  config = vars(args)
  run(config, stdout=myargs.stdout)


if __name__ == '__main__':    
    main()