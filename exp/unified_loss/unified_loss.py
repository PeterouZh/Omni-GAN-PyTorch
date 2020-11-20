import os
import torch
import torch.nn.functional as F
import random


def multilabel_categorical_crossentropy(y_true, y_pred, margin=0., gamma=1.):
  """
  y_true: positive=1, negative=0, ignore=-1

  """
  y_true = y_true.clamp(-1, 1)
  if len(y_pred.shape) > 2:
    y_true = y_true.view(y_true.shape[0], 1, 1, -1)
    _, _, h, w = y_pred.shape
    y_true = y_true.expand(-1, h, w, -1)
    y_pred = y_pred.permute(0, 2, 3, 1)

  y_pred = y_pred + margin
  y_pred = y_pred * gamma

  y_pred[y_true == 1] = -1 * y_pred[y_true == 1]
  y_pred[y_true == -1] = -1e12

  y_pred_neg = y_pred.clone()
  y_pred_neg[y_true == 1] = -1e12

  y_pred_pos = y_pred.clone()
  y_pred_pos[y_true == 0] = -1e12

  zeros = torch.zeros_like(y_pred[..., :1])
  y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
  y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
  neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
  pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
  return neg_loss + pos_loss


class UnifiedLoss(object):

  def __init__(self, default_label=0, margin=0., gamma=1.):
    self.default_label = default_label
    self.margin = margin
    self.gamma = gamma
    pass

  @staticmethod
  def get_one_hot(label_list, one_hot, b, filled_value=0):

    for label in label_list:
      if isinstance(label, int):
        label = torch.empty(b, dtype=torch.int64, device=one_hot.device).fill_(label)
      one_hot.scatter_(dim=1, index=label.view(-1, 1), value=filled_value)
    return one_hot

  def __call__(self, pred, positive=None, negative=None, default_label=None, margin=None, gamma=None):
    default_label = self.default_label if default_label is None else default_label
    margin = self.margin if margin is None else margin
    gamma = self.gamma if gamma is None else gamma

    b, nc = pred.shape[:2]
    label_onehot = torch.empty(b, nc, dtype=torch.int64, device=pred.device).fill_(default_label)

    if positive is not None:
      label_onehot = UnifiedLoss.get_one_hot(label_list=positive, one_hot=label_onehot, b=b, filled_value=1)

    if negative is not None:
      label_onehot = UnifiedLoss.get_one_hot(label_list=negative, one_hot=label_onehot, b=b, filled_value=0)

    loss = multilabel_categorical_crossentropy(
      y_true=label_onehot, y_pred=pred, margin=margin, gamma=gamma)
    loss_mean = loss.mean()
    return loss_mean

  @staticmethod
  def test_case():
    b, nc = 32, 101
    pred = torch.rand(b, nc).cuda().requires_grad_()
    y = torch.randint(0, nc-1, (b, )).cuda()

    unified_loss = UnifiedLoss()
    D_loss_real = unified_loss(pred=pred, positive=(y, nc-1))

    D_loss_fake = unified_loss(pred=pred, positive=None, negative=(y, nc-1))

    adv_loss = unified_loss(pred=pred, positive=(y, nc-1))

    D_loss_real = unified_loss(pred=pred, positive=(y, nc - 1), default_label=0)

    # b, nc, h, w = 32, 101, 8, 8
    # pred = torch.rand(b, nc, h, w).cuda().requires_grad_()
    # y = torch.randint(0, nc - 1, (b, h, w)).cuda()
    # D_loss_real = unified_loss(pred=pred, positive=(y, nc - 1))
    pass


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '7'
  UnifiedLoss.test_case()