""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""
import logging
import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange
from easydict import EasyDict
from collections import defaultdict
import importlib

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback

from template_lib.v2.config import update_parser_defaults_from_yaml
from template_lib.v2.config import get_dict_str, global_cfg
from template_lib.v2.logger import summary_defaultdict2txtfig
from template_lib.v2.logger import global_textlogger as textlogger


import exp.scripts

# try:
#   import pydevd_pycharm
#   pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
#   os.environ['TIME_STR'] = '0'
# except:
#   print(f"import error: pydevd_pycharm")
#   pass

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):
  logger = logging.getLogger('tl')
  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = importlib.import_module(config['model'])
  # model = __import__(config['model'])
  experiment_name = 'exp'
  # experiment_name = (config['experiment_name'] if config['experiment_name']
  #                      else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator(**config, cfg=getattr(global_cfg, 'generator', None)).to(device)
  D = model.Discriminator(**config, cfg=getattr(global_cfg, 'discriminator', None)).to(device)
  
   # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}, cfg=getattr(global_cfg, 'generator', None)).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  logger.info(G)
  logger.info(D)
  logger.info('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    utils.load_weights(G=G, D=D, state_dict=state_dict,
                       weights_root=global_cfg.resume_cfg.weights_root, experiment_name='',
                       name_suffix=config['load_weights'] if config['load_weights'] else None,
                       G_ema=G_ema if config['ema'] else None)
    logger.info(f"Resume IS={state_dict['best_IS']}")
    logger.info(f"Resume FID={state_dict['best_FID']}")

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr'],
                                      **getattr(global_cfg, 'train_dataloader', {})}
                                   )

  val_loaders = None
  if hasattr(global_cfg, 'val_dataloader'):
    val_loaders = utils.get_data_loaders(**{**config, 'batch_size': config['batch_size'],
                                            'start_itr': state_dict['itr'],
                                            **global_cfg.val_dataloader}
                                         )[0]
    val_loaders = iter(val_loaders)
  # Prepare inception metrics: FID and IS
  if global_cfg.get('use_unofficial_FID', False):
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['inception_file'], config['parallel'],
                                                                      config['no_fid'])
  else:
    get_inception_metrics = inception_utils.prepare_FID_IS(global_cfg)
  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])  
  fixed_z.sample_()
  fixed_y.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, 
                                            ema, state_dict, config, val_loaders)
  # Else, assume debugging and use the dummy train fn
  elif config['which_train_fn'] == 'dummy':
    train = train_fns.dummy_training_function()
  else:
    train_fns_module = importlib.import_module(config['which_train_fn'])
    train = train_fns_module.GAN_training_function(G, D, GD, z_, y_,
                                                   ema, state_dict, config, val_loaders)

  # Prepare Sample function for use with inception metrics
  if global_cfg.get('use_unofficial_FID', False):
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                  else G),
                               z_=z_, y_=y_, config=config)
  else:
    sample = functools.partial(utils.sample_imgs,
                                G=(G_ema if config['ema'] and config['use_ema']
                                   else G),
                                z_=z_, y_=y_, config=config)

  state_dict['shown_images'] = state_dict['itr'] * D_batch_size

  if global_cfg.get('resume_cfg', {}).get('eval', False):
    logger.info(f'Evaluating model.')
    G_ema.eval()
    G.eval()
    train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                   get_inception_metrics, experiment_name, test_log)
    return



  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):    
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0], desc=f'Epoch:{epoch}, Itr: ',
                            displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders[0])
    for i, (x, y) in enumerate(pbar):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)

      default_dict = train(x, y)

      state_dict['shown_images'] += D_batch_size

      metrics = default_dict['D_loss']
      train_log.log(itr=int(state_dict['itr']), **metrics)

      summary_defaultdict2txtfig(default_dict=default_dict, prefix='train', step=state_dict['shown_images'],
                                 textlogger=textlogger)
      
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']] 
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ', flush=True)

      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if (state_dict['itr'] % config['test_every'] == 0) or \
            state_dict['itr'] == 1 or \
            not (state_dict['itr'] % (global_cfg.get('test_every_epoch', float('inf')) * len(loaders[0]))) or \
            (state_dict['shown_images'] % global_cfg.get('test_every_images', float('inf'))) < D_batch_size:
        if config['G_eval_mode']:
          print('Switchin G to eval mode...', flush=True)
          G.eval()
        G_ema.eval()
        print('\n' + config['tl_outdir'])
        train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                       get_inception_metrics, experiment_name, test_log)
    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1


def run1(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  parser = utils.prepare_parser()
  args = parser.parse_args([])
  args = config2args(myargs.config.args, args)

  args.base_root = os.path.join(myargs.args.outdir, 'biggan')

  main(config=EasyDict(vars(args)), myargs=myargs)

def main():
  # parse command line and run
  parser = utils.prepare_parser()

  update_parser_defaults_from_yaml(parser)

  args = parser.parse_args()
  args.base_root = os.path.join(args.tl_outdir, 'biggan')
  config = EasyDict(vars(args))
  config_str = get_dict_str(config)
  logger = logging.getLogger('tl')
  logger.info(config_str)
  run(config)

if __name__ == '__main__':
  main()