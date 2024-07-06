import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
      """
        logits = logits[..., None]
        labels = labels[..., None]
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length], device=logits.device).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

# class Balanced_CE_loss(torch.nn.Module):
#     def __init__(self):
#         super(Balanced_CE_loss, self).__init__()

#     def forward(self, input, target):
#         input = input.view(input.shape[0], -1)
#         target = target.view(target.shape[0], -1)
#         loss = 0.0
#         for i in range(input.shape[0]):
#             beta = 1-torch.sum(target[i])/target.shape[1]
#             for j in range(input.shape[1]):
#                 loss += -(beta*target[i][j] * torch.log(input[i][j]) + (1-beta)*(1 - target[i][j]) * torch.log(1 - input[i][j]))
#         return loss
class Balanced_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        input = input.view(input.shape[0], -1).to(device)
        target = target.view(target.shape[0], -1).to(device)
        loss = 0.0
        # version2
        for i in range(input.shape[0]):
            beta = 1-torch.sum(target[i])/target.shape[1]
            x = torch.max(torch.log(input[i]), torch.tensor([-100.0]).to(device))
            y = torch.max(torch.log(1-input[i]), torch.tensor([-100.0]).to(device))
            l = -(beta*target[i] * x + (1-beta)*(1 - target[i]) * y)
            loss += torch.sum(l).to(device)
        return loss
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=9):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.class_balance_loss = Balanced_CE_loss()
        self.class_balance_loss = nn.CrossEntropyLoss(weight=None, reduction='mean')
        self.focal_loss =FocalLoss().to(device)
        # self.focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma, num_classes=self.num_classes)

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy_loss(inputs, targets)
        cb_loss = self.class_balance_loss(inputs, targets)
        fl_loss = self.focal_loss(inputs, targets)
        return (ce_loss +cb_loss + fl_loss) / 3
        # return (ce_loss + fl_loss) / 2