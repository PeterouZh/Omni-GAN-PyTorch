import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys
import unittest

from template_lib import utils


class TestingPlot(unittest.TestCase):

  def test_plot_FID_IS(self):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file none
                    --tl_command none
                    --tl_outdir {outdir}
                    """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from template_lib.utils.plot_results import PlotResults
    import collections

    outfigure = os.path.join(outdir, 'FID_IS.jpg')
    default_dicts = []
    show_max = []

    FID_c100 = collections.defaultdict(dict)
    title = 'FID_c100'
    log_file = 'textdir/evaltf.ma0.FID_tf.log'
    dd = eval(title)
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_Omni_GAN_cifar100_3'] = \
      {'Omni-GAN-c100': log_file, }
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_BigGAN_cifar100'] = \
      {'BigGAN-c100': log_file, }

    dd['properties'] = {'title': title, 'ylim': [0, 30]}
    default_dicts.append(dd)
    show_max.append(False)

    IS_c100 = collections.defaultdict(dict)
    title = 'IS_c100'
    log_file = 'textdir/evaltf.ma1.IS_mean_tf.log'
    dd = eval(title)
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_Omni_GAN_cifar100_3'] = \
      {'Omni-GAN-c100': log_file, }
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_BigGAN_cifar100'] = \
      {'BigGAN-c100': log_file, }

    dd['properties'] = {'title': title, }
    default_dicts.append(dd)
    show_max.append(True)


    FID_c10 = collections.defaultdict(dict)
    title = 'FID_c10'
    log_file = 'textdir/evaltf.ma0.FID_tf.log'
    dd = eval(title)
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_Omni_GAN_cifar10'] = \
      {'Omni-GAN-c10': log_file, }
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_BigGAN_cifar10'] = \
      {'BigGAN-c10': log_file, }

    dd['properties'] = {'title': title, 'ylim': [0, 30]}
    default_dicts.append(dd)
    show_max.append(False)

    IS_c10 = collections.defaultdict(dict)
    title = 'IS_c10'
    log_file = 'textdir/evaltf.ma1.IS_mean_tf.log'
    dd = eval(title)
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_Omni_GAN_cifar10'] = \
      {'Omni-GAN-c10': log_file, }
    dd['/home/user321/user/code_sync/Omni-GAN-PyTorch/results/train_BigGAN_cifar10'] = \
      {'BigGAN-c10': log_file, }

    dd['properties'] = {'title': title, }
    default_dicts.append(dd)
    show_max.append(True)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')
    pass