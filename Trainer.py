import time
import torch.optim as optim
import torch #파이토치 기본모듈
import numpy as np
from MultiClassDeepSVDD import Autoencoder,DeepSVDD

class train_model:
    def __init__(self, lr_pretrain, weight_decay_pretrain, epochs_pretrain, device, train_loader, rep_dim, num_class, pretrained):
        self.lr_pretrain = lr_pretrain
        self.weight_decay_pretrain = weight_decay_pretrain
        self.epochs_pretrain = epochs_pretrain
        self.device = device
        self.train_loader = train_loader
        self.rep_dim = rep_dim
        self.num_class = num_class
        self.pretrained = pretrained

    def pretrain(self):
        ae_net = Autoencoder(self.rep_dim).to(self.device)

        if self.pretrained:
            ae_net.load_state_dict(torch.load('aefloormodel_repdim'+str(self.rep_dim)+'_state_dict.pt'))
        else:
            lr_milestones = tuple()
            optimizer = optim.Adam(ae_net.parameters(), lr=self.lr_pretrain, weight_decay=self.weight_decay_pretrain,
                                        amsgrad='adam'=='amsgrad')
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

            print('Starting pretraining...')
            start_time = time.time()
            ae_net.train()
            for epoch in range(self.epochs_pretrain):

                scheduler.step()
                if epoch in lr_milestones:
                    print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

                loss_epoch = 0.0
                n_batches = 0
                epoch_start_time = time.time()
                for X, Y in self.train_loader:
                    X = X.to(self.device)

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = ae_net(X)
                    scores = torch.sum((outputs - X) ** 2, dim=tuple(range(1, outputs.dim())))
                    loss = torch.mean(scores)
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()
                    n_batches += 1

                # log epoch statistics
                epoch_train_time = time.time() - epoch_start_time
                print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                            .format(epoch + 1, self.epochs_pretrain, epoch_train_time, loss_epoch / n_batches))

            pretrain_time = time.time() - start_time
            print('Pretraining time: %.3f' % pretrain_time)
            print('Finished pretraining.')

        self.save_weight_to_model(ae_net)

    def save_weight_to_model(self, ae_net):

        ae_net.eval()
        net = DeepSVDD(self.rep_dim).to(self.device)
        net_dict = net.state_dict()
        ae_net_dict = ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        net.load_state_dict(net_dict)

        torch.save(net.state_dict(), 'floormodel_repdim'+str(self.rep_dim)+'_state_dict.pt')
        torch.save(ae_net.state_dict(), 'aefloormodel_repdim'+str(self.rep_dim)+'_state_dict.pt')

    def train(self):
           
        self.pretrain()
        net = DeepSVDD(self.rep_dim).to(self.device)
        net.load_state_dict(torch.load('floormodel_repdim'+str(self.rep_dim)+'_state_dict.pt'))

        self.net = net

        return self.net



