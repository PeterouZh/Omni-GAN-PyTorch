import collections
import logging
import os
import weakref
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import tqdm

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import (
  MetadataCatalog,
  build_detection_test_loader,
  build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
  COCOEvaluator,
  COCOPanopticEvaluator,
  DatasetEvaluators,
  LVISEvaluator,
  PascalVOCDetectionEvaluator,
  SemSegEvaluator,
  inference_on_dataset,
  print_csv_format,
)
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
  CommonMetricPrinter,
  EventStorage,
  JSONWriter,
  TensorboardXWriter,
)

from template_lib.utils import detection2_utils, get_attr_eval, get_attr_kwargs
from template_lib.utils.detection2_utils import D2Utils
from template_lib.utils import modelarts_utils
from template_lib.utils import seed_utils
from template_lib.utils.modelarts_utils import prepare_dataset
from template_lib.d2.data import build_dataset_mapper
from template_lib.d2template.trainer import build_trainer
from template_lib.d2template.scripts import build_start, START_REGISTRY

# import scripts

logger = logging.getLogger("detectron2")


@START_REGISTRY.register()
def save_FID_CBN_location_figure(cfg, args, myargs):
  import matplotlib.pyplot as plt
  import numpy as np

  cfg = myargs.config

  fig, ax = plt.subplots()
  fig.show()

  ax.set_xticks(range(0, 400, 50))
  ax.tick_params(labelsize=14)
  ax.set_xlabel('Epoch', fontsize=20)
  ax.set_ylabel(r'FID', fontsize=20)

  num_plot = len(list(cfg.fid_files.keys()))
  colors = [plt.cm.cool(i / float(num_plot - 1)) for i in range(num_plot)]

  ax.set(**cfg.properties)
  for idx, (_, data_dict) in enumerate(cfg.fid_files.items()):
    log_file = data_dict.log_file
    data = np.loadtxt(log_file, delimiter=':')
    ax.plot(data[:, 0], data[:, 1], c=colors[idx], **data_dict.properties)
    pass

  ax.legend(prop={'size': 20})
  saved_file = os.path.join(myargs.args.outdir, cfg.saved_file)
  fig.savefig(saved_file, bbox_inches='tight', pad_inches=0.01)

  pass


@START_REGISTRY.register()
def save_FID_CMConv_CBN_location_figure(cfg, args, myargs):
  import matplotlib.pyplot as plt
  import numpy as np

  cfg = myargs.config

  fig, ax = plt.subplots()
  # fig.show()

  ax.set_xticks(range(0, 600, 100))
  ax.tick_params(labelsize=14)
  ax.set_xlabel('Epoch', fontsize=20)
  ax.set_ylabel(r'FID', fontsize=20)

  num_plot = len(list(cfg.fid_files.keys()))
  colors = [plt.cm.cool(i / float(num_plot - 1)) for i in range(num_plot)]

  ax.set(**cfg.properties)
  for idx, (_, data_dict) in enumerate(cfg.fid_files.items()):
    log_file = data_dict.log_file
    data = np.loadtxt(log_file, delimiter=':')
    ax.plot(data[:, 0], data[:, 1], **data_dict.properties)
    # ax.plot(data[:, 0], data[:, 1], c=colors[idx], **data_dict.properties)
    pass

  ax.legend(prop={'size': 12})
  saved_file = os.path.join(myargs.args.outdir, cfg.saved_file)
  fig.savefig(saved_file, bbox_inches='tight', pad_inches=0.01)

  pass



def setup(args, config):
  """
  Create configs and perform basic setups.
  """
  from detectron2.config import CfgNode
  # detectron2 default cfg
  # cfg = get_cfg()
  cfg = CfgNode()
  cfg.OUTPUT_DIR = "./output"
  cfg.SEED = -1
  cfg.CUDNN_BENCHMARK = False
  cfg.DATASETS = CfgNode()
  cfg.SOLVER = CfgNode()

  cfg.DATALOADER = CfgNode()
  cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
  cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

  cfg.MODEL = CfgNode()
  cfg.MODEL.KEYPOINT_ON = False
  cfg.MODEL.LOAD_PROPOSALS = False
  cfg.MODEL.WEIGHTS = ""

  # cfg.merge_from_file(args.config_file)
  cfg.merge_from_list(args.opts)

  cfg = detection2_utils.D2Utils.cfg_merge_from_easydict(cfg, config)

  cfg.freeze()
  default_setup(
    cfg, args
  )  # if you don't like any of the default setup, write your own setup code
  return cfg


def main(args, myargs):
  cfg = setup(args, myargs.config)
  myargs = D2Utils.setup_myargs_for_multiple_processing(myargs)
  # seed_utils.set_random_seed(cfg.seed)

  build_start(cfg=cfg, args=args, myargs=myargs)

  modelarts_utils.modelarts_sync_results(args=myargs.args, myargs=myargs, join=True, end=True)
  return


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  args = default_argument_parser().parse_args(args=[])
  args = config2args(myargs.config.args, args)

  args.opts += ['OUTPUT_DIR', args1.outdir + '/detectron2']
  print("Command Line Args:", args)

  myargs = D2Utils.unset_myargs_for_multiple_processing(myargs, num_gpus=args.num_gpus)

  launch(
    main,
    args.num_gpus,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args, myargs),
  )


if __name__ == "__main__":
  run()