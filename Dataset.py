from torch.utils.data import Dataset
from scipy import io
from scipy import signal
import torch #파이토치 기본모듈
import numpy as np


class MyBaseDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data.astype(np.float32)
        self.y_data = y_data.astype(np.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]

class get_data:
    def __init__(self, path_undamaged, path_damaged, lpf):
        self.path_undamaged = path_undamaged
        self.path_damaged = path_damaged
        self.lpf = lpf

    def get_undamaged(self):
        file_name_undamaged = [[] for i in range(len(self.path_undamaged))]
        for k in range(len(self.path_undamaged)):
            for i in range(3):
                for j in range(10):
                    if k == 0:
                        file_name_undamaged[k].append("L00_D00_V0"+str(i*3+2)+"_071200_"+'{:0>2}'.format(str(j+1+k*10)))
                    elif k == 4:
                        file_name_undamaged[k].append("L00_D00_V0"+str(i*3+2)+"_071800_"+'{:0>2}'.format(str(j+1+k*10)))
                    else:
                        file_name_undamaged[k].append("L00_D00_V0"+str(i*3+2)+"_071300_"+'{:0>2}'.format(str(j+1+k*10)))
        file_undamaged = []
        Y_undamaged = []
        file_undamaged = np.array(file_undamaged)
        for k in range(len(self.path_undamaged)):
            for j in range(len(file_name_undamaged[k])):
                f = io.loadmat(self.path_undamaged[k]+file_name_undamaged[k][j])['Data_'+file_name_undamaged[k][j]].T
                f[8]=f[25]
                f = np.delete(f,25, axis = 0)
                f = np.delete(f,24, axis = 0)
                ff = []
                for m in range(8):
                    if self.lpf:
                        p = []
                        for u in range(0, len(f[0])-4096,512):
                            pp = abs(np.fft.fft(f[:,u+m:u+4096:8])/512)
                            p_ = signal.firwin(5, cutoff=150, fs=512, pass_zero='lowpass')
                            pp = signal.lfilter(p_,[1.0],pp)
                            p.append(pp.T)
                        ff.append(np.array(p))
                    else:
                        ff.append(np.array([abs(np.fft.fft(f[:,i+m:i+4096:8])/512).T for i in range(0, len(f[0])-4096,512)]))
                    Y_undamaged.extend([1]*int(4096/512))
                if file_undamaged.shape[0] == 0:
                    file_undamaged = ff[0]
                    for m in range(7):
                        file_undamaged = np.append(file_undamaged, ff[m+1], axis=0)
                else:
                    for m in range(8):
                        file_undamaged = np.append(file_undamaged, ff[m], axis=0)

        #print(file)
        X_undamaged = np.array(file_undamaged)
        X_undamaged = np.expand_dims(X_undamaged, axis=1)
        Y_undamaged = np.array(Y_undamaged)
        return X_undamaged, Y_undamaged

    def get_damaged(self):
        Y = [[] for i in range(len(self.path_damaged))]
        file_name_damaged = [[] for i in range(len(self.path_damaged))]
        for k in range(len(self.path_damaged)):
            for i in range(3):
                for j in range(5):
                    if k == 0:
                        file_name_damaged[k].append("L1C_DB0_V0"+str(i*3+2)+"_071200_"+'{:0>2}'.format(str(j+1)))
                        Y[k].append(2)
                        file_name_damaged[k].append("L1C_DBB_V0"+str(i*3+2)+"_071200_"+'{:0>2}'.format(str(j+1)))
                        Y[k].append(2)
                    elif k == 1:
                        if i == 0:
                            file_name_damaged[k].append("L1C_D05_V08"+"_071700_"+'{:0>2}'.format(str(j+1)))
                            Y[k].append(2)
                            file_name_damaged[k].append("L1C_D05_V08"+"_071700_"+'{:0>2}'.format(str(j+6)))
                            Y[k].append(2)
                        elif i == 1:
                            file_name_damaged[k].append("L1C_D10_V08"+"_071700_"+'{:0>2}'.format(str(j+1)))
                            Y[k].append(2)
                            file_name_damaged[k].append("L1C_D10_V08"+"_071700_"+'{:0>2}'.format(str(j+6)))
                            Y[k].append(2)
                        else:
                            file_name_damaged[k].append("L1C_DHT_V08"+"_071700_"+'{:0>2}'.format(str(j+1)))
                            Y[k].append(2)
                            file_name_damaged[k].append("L1C_DHT_V08"+"_071700_"+'{:0>2}'.format(str(j+6)))
                            Y[k].append(2)
                    elif k == 2:
                        file_name_damaged[k].append("L3A_DB0_V0"+str(i*3+2)+"_071300_"+'{:0>2}'.format(str(j+1)))
                        Y[k].append(3)
                        file_name_damaged[k].append("L3A_DBB_V0"+str(i*3+2)+"_071300_"+'{:0>2}'.format(str(j+1)))
                        Y[k].append(3)
                    else:
                        file_name_damaged[k].append("L13_DB0_V0"+str(i*3+2)+"_071700_"+'{:0>2}'.format(str(j+1)))
                        Y[k].append(0)
                        file_name_damaged[k].append("L13_DBB_V0"+str(i*3+2)+"_071700_"+'{:0>2}'.format(str(j+1)))
                        Y[k].append(0)
        file_damaged = []
        file_damaged = np.array(file_damaged)
        Y_damaged = []
        for k in range(len(self.path_damaged)):
            for i in range(len(file_name_damaged[k])):
                f = io.loadmat(self.path_damaged[k]+file_name_damaged[k][i])['Data_'+file_name_damaged[k][i]].T
                f[8]=f[25]
                f = np.delete(f,25, axis = 0)
                f = np.delete(f,24, axis = 0)
                ff = []
                for m in range(8):
                    if self.lpf:
                        p = []
                        for u in range(0, len(f[0])-4096,512):
                            pp = abs(np.fft.fft(f[:,u+m:u+4096:8])/512)
                            p_ = signal.firwin(5, cutoff=150, fs=512, pass_zero='lowpass')
                            pp = signal.lfilter(p_,[1.0],pp)
                            p.append(pp.T)
                        ff.append(np.array(p))
                    else:
                        ff.append(np.array([abs(np.fft.fft(f[:,i+m:i+4096:8])/512).T for i in range(0, len(f[0])-4096,512)]))
                    Y_damaged.extend([Y[k][i]]*int(4096/512))
                if file_damaged.shape[0] == 0:
                    file_damaged = ff[0]
                    for m in range(7):
                        file_damaged = np.append(file_damaged, ff[m+1], axis=0)
                else:
                    for m in range(8):
                        file_damaged = np.append(file_damaged, ff[m], axis=0)
        X_damaged = np.array(file_damaged)
        X_damaged = np.expand_dims(X_damaged, axis=1)
        Y_damaged = np.array(Y_damaged)
        return X_damaged, Y_damaged

    def load_data(self, BATCH_SIZE, data_saved):
        if data_saved == True:
            X_train = np.load('X_train.npy')
            X_test = np.load('X_test.npy')
            Y_train = np.load('Y_train.npy')
            Y_test = np.load('Y_test.npy')
        else:
            X_undamaged, Y_undamaged = self.get_undamaged()
            X_damaged, Y_damaged = self.get_damaged()
            X_damaged_2 = X_damaged[0:3840]                          #1 - 9600개 2 - 3840개 3 - 1920개 0 - 1920개
            X_damaged_3 = X_damaged[3840:3840+1920]                  #1층 결함 - 2, 3층 결함 - 3, 결함 x - 1
            X_damaged_0 = X_damaged[3840+1920:3840+3840]
            Y_damaged_2 = Y_damaged[0:3840]
            Y_damaged_3 = Y_damaged[3840:3840+1920]+1
            Y_damaged_0 = Y_damaged[3840+1920:3840+3840]

            idx1 = np.arange(X_undamaged.shape[0])
            idx2 = np.arange(X_damaged_2.shape[0])
            idx3 = np.arange(X_damaged_3.shape[0])
            idx4 = np.arange(X_damaged_0.shape[0])

            np.random.shuffle(idx1)
            np.random.shuffle(idx2)
            np.random.shuffle(idx3)
            np.random.shuffle(idx4)

            X_undamaged = X_undamaged[idx1]
            Y_undamaged = Y_undamaged[idx1]
            X_damaged_2 = X_damaged_2[idx2]
            Y_damaged_2 = Y_damaged_2[idx2]
            X_damaged_3 = X_damaged_3[idx3]
            Y_damaged_3 = Y_damaged_3[idx3]
            X_damaged_0 = X_damaged_0[idx4]
            Y_damaged_0 = Y_damaged_0[idx4]

            X_undamaged_1 = X_undamaged[:8000,:,:,16:24]
            X_undamaged_2 = X_undamaged[:8000,:,:,8:16]
            X_undamaged_3 = X_undamaged[:8000,:,:,0:8]
            Y_undamaged_1 = Y_undamaged[:8000]*1
            Y_undamaged_2 = Y_undamaged[:8000]*2
            Y_undamaged_3 = Y_undamaged[:8000]*3

            X_train = np.concatenate((X_undamaged_1,X_undamaged_2,X_undamaged_3),axis=0)
            Y_train = np.concatenate((Y_undamaged_1,Y_undamaged_2,Y_undamaged_3),axis=0)

            X_test = np.concatenate((X_undamaged[8000:],X_damaged_2[:1920],X_damaged_3),axis=0)
            Y_test = np.concatenate((Y_undamaged[8000:],Y_damaged_2[:1920],Y_damaged_3),axis=0)

            np.save('X_train.npy', X_train)
            np.save('Y_train.npy', Y_train)
            np.save('X_test.npy', X_test)
            np.save('Y_test.npy', Y_test)

        print(X_train.shape,Y_train.shape)

        train_data = MyBaseDataset(X_train, Y_train)
        test_data = MyBaseDataset(X_test, Y_test)


        train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                                batch_size = BATCH_SIZE,
                                                shuffle = False)
        return train_loader, test_loader