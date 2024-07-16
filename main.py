import torch #파이토치 기본모듈

from Dataset import get_data
from Variables import path_damaged, path_undamaged, BATCH_SIZE, rep_dims, lr_pretrain, epochs_pretrain, weight_decay_pretrain, num_class, rep_dims, data_saved, pretrained, bandwidth, lpf
from Trainer import train_model
from Tester import test_model

if torch.cuda.is_available():
    device = torch.device('cuda') #GPU이용

else:
    device = torch.device('cpu') #GPU이용안되면 CPU이용

print('Using PyTorch version:', torch.__version__, ' Device:', device)

Data = get_data(path_undamaged, path_damaged, lpf)
train_loader, test_loader = Data.load_data(BATCH_SIZE, data_saved)

B = bandwidth
for rep_dim in rep_dims:
    for bandwidth in B:
        SVDD_trainer = train_model(lr_pretrain, weight_decay_pretrain, epochs_pretrain, device, train_loader, rep_dim, num_class, pretrained)
        net = SVDD_trainer.train()

        SVDD_tester = test_model(net, train_loader, test_loader, device)
        SVDD_tester.confusion_mat(rep_dim, bandwidth)
