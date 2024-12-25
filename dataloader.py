import os
import csv
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import h5py
def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_patient_label_no_cross_fold(csv_file):
    patients_list=[]
    labels_list=[]
    patients_train_list=[]
    label_train_list=[]
    patients_test_list=[]
    label_test_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
        #if label_file[i][0]=='test_*':
        if label_file[i][0].startswith('test_'):
            patients_test_list.append(label_file[i][0])
            label_test_list.append(label_file[i][1])
        else:
            patients_train_list.append(label_file[i][0])
            label_train_list.append(label_file[i][1])        
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("train_patient_len:{} train_label_len:{}".format(len(patients_train_list), len(label_train_list)))
    print("test_patient_len:{} test_label_len:{}".format(len(patients_test_list), len(label_test_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object),np.array(patients_train_list,dtype=object), np.array(label_train_list,dtype=object),np.array(patients_test_list,dtype=object), np.array(label_test_list,dtype=object),


def get_patient_label(csv_file):
    patients_list=[]
    labels_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
    """
    dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
    :param full_list: 
    :param ratio:     
    :param shuffle:  
    """
    # select the val-set based on the label ratio
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []

        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])
        return sublist_1, sublist_2
    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        val_set = full_list[:offset]
        train_set = full_list[offset:]

        return val_set, train_set


def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [],[]
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)
        
    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)

def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()

class TCGADataset(Dataset):
    def __init__(self, file_name=None, file_label=None,root=None,persistence=True,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset, self).__init__()
        self.patient_name = file_name
        print('self.patient_name:',self.patient_name)
        self.patient_label = file_label
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'h5_files')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
        self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]

    def __len__(self):
        return len(self.slide_name)
    
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        label = self.slide_label[idx]
        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path))
        return features , int(label)

class C16Dataset(Dataset):
    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        print('self.file_name:',self.file_name)
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"pt")

            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)

        label = int(self.slide_label[idx])
        return features , label
    
class C16Dataset_no_cross_fold(Dataset):
    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset_no_cross_fold, self).__init__()
        self.file_name = file_name
        print('self.file_name:',file_name)
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        self.cluster_npy_path=cluster_npy_path
        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]
        a=0
        b=0
        for i in range(len(self.slide_label)):
            label = int(self.slide_label[i])
            if label==0:
                a=a+1
            elif label==1:
                b=b+1
            else:
                print('error')
        print('a:',a)
        print('b:',b)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"pt_files")
            #dir_path = os.path.join(self.root,"pt")
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)
            file_name=self.file_name[idx]
            h5_file_path = os.path.join(self.root, 'h5_files',self.file_name[idx]+'.h5')
            
            with h5py.File(h5_file_path,'r') as hdf5_file:
                coord = hdf5_file['coords'][:]
        label = int(self.slide_label[idx])
        return features , label ,coord,file_name
     

class GastricCancer(Dataset):
    def __init__(self, file_name,file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(GastricCancer, self).__init__()
        self.file_name = file_name
        print('self.file_name:',file_name)
        self.slide_label = file_label
        self.cluster_npy_path=cluster_npy_path
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        a=0
        b=0
        c=0
        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]
        for i in range(len(self.slide_label)):
            label = int(self.slide_label[i])
            if label==0:
                a=a+1
            elif label==1:
                b=b+1
            else:
                print('error')
        print('a:',a)
        print('b:',b)
       
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"pt")
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)
            file_name=self.file_name[idx]
            h5_file_path = os.path.join(self.root, 'h5_files',self.file_name[idx]+'.h5')
            
            with h5py.File(h5_file_path,'r') as hdf5_file:
                coord = hdf5_file['coords'][:]
                #print('coord:',coord)
        label = int(self.slide_label[idx])
        return features , label ,coord ,file_name
    
class JIANGFENG_DATA(Dataset):

    def __init__(self, file_name,file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(JIANGFENG_DATA, self).__init__()
        self.file_name = file_name
        print('self.file_name:',file_name)
        self.slide_label = file_label

        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        a=0
        b=0
        c=0
        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]
        for i in range(len(self.slide_label)):
            label = int(self.slide_label[i])
            if label==0:
                a=a+1
            elif label==1:
                b=b+1
            elif label==2:
                c=c+1
            else:
                print('error')
        print('a:',a)
        print('b:',b)
        print('c:',c)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"pt")
            
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)
            
            h5_file_path = os.path.join(self.root, 'h5_files',self.file_name[idx]+'.h5')
            
            with h5py.File(h5_file_path,'r') as hdf5_file:
                coord = hdf5_file['coords'][:]
                #print('coord:',coord)
            
        label = int(self.slide_label[idx])
        return features , label ,coord 