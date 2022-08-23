import torch.nn as nn
import torch

class Ts_loss_cal(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(size_average=False, reduce=False, reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logit_teacher, logit_student):
        # prob_teacher = self.softmax(logit_teacher) + 1e-10
        prob_teacher = self.softmax(logit_teacher)
        prob_student = self.softmax(logit_student)
        prob_student_log = torch.log(prob_student)
        loss = self.kl(prob_student_log, prob_teacher)
        loss_kl = torch.mean(torch.sum(loss, dim=-1))
        return loss_kl