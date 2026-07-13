import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')




class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        if self.args.use_closedllm==0:
            text_name='Final_Search_'+str(self.args.text_len)
        else:
            print("!!!!!!!!!!!!Using output of closed source llm and Bert as encoder!!!!!!!!!!!!!!!")
            text_name="Final_Output"
        df_raw = df_raw[['date'] + cols + [self.target]+['prior_history_avg']+['start_date']+['end_date']+[text_name]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values[:,-1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]


        self.data_stamp = data_stamp
        self.date=df_raw[['date']][border1:border2].values
        self.start_date=df_raw[['start_date']][border1:border2].values
        self.end_date=df_raw[['end_date']][border1:border2].values
        self.text=df_raw[[text_name]][border1:border2].values
    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends 
        r_ends = r_begins + self.pred_len
        prior_y=np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y
    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        
        text=np.array([self.text[s_end] for s_end in s_ends])
        return text
    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends - self.label_len
        r_ends = r_begins + self.label_len + self.pred_len

        x_start_dates = np.array([self.start_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        x_end_dates = np.array([self.end_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        
        

        return x_start_dates, x_end_dates
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


# ========== Contrastive Learning Dataset ==========

class Dataset_Traffic_Contrastive(Dataset):
    """
    交通数据对比学习数据集
    读取 Traffic_samples_with_context.csv 或 traffic_generated_all.csv

    数据格式:
        - sample_id: 样本ID
        - window_start: 窗口起始日期
        - window_end: 窗口结束日期
        - x_ot: 时间序列（字符串形式的列表）
        - y_ot: 未来序列（可选，可为空）
        - context_text: 文本描述
        - llm_material: LLM材料（月度报告+辅助文本）
        - split: train/val/test（可选，若存在则按该列切分）

    Args:
        root_path: 数据根目录
        data_path: CSV文件名
        flag: 'train'/'val'/'test'
        split_ratio: (train, val, test) 比例，默认 (0.7, 0.1, 0.2)
        scale: 是否标准化时间序列数据
        scaler: 预训练的scaler（用于val/test）
    """
    def __init__(self, root_path, data_path, flag='train', split_ratio=(0.7, 0.1, 0.2), scale=True, scaler=None):
        assert flag in ['train', 'val', 'test']
        self.flag = flag

        # 读取CSV
        file_path = os.path.join(root_path, data_path)
        df = pd.read_csv(file_path)
        print("原始数据长度为：", len(df))

        print(f"Loading {flag} data from {file_path}")

        # 仅支持包含 split 列的已标准化样本文件
        if 'split' not in df.columns:
            raise ValueError("Contrastive dataset requires 'split' column in CSV.")
        df = df[df['split'] == flag].reset_index(drop=True)

        # 解析时间序列（字符串转列表）
        # df['x_ot_parsed'] = df['x_ot'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        import ast
        df['x_ot_parsed'] = df['x_ot'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        print(df['x_ot_parsed'].shape)

        # 转为numpy数组
        data_x = np.array(df['x_ot_parsed'].tolist())  # [N, 8]
        data_x = data_x[:, :, np.newaxis]  # [N, 8, 1] 添加特征维度

        # 获取文本（允许为空）
        self.text = df['context_text'].fillna('').astype(str).tolist()
        self.data_x = data_x
        self.sample_ids = df['sample_id'].tolist() if 'sample_id' in df.columns else list(range(len(df)))

        print(f'{flag} samples: {len(self.data_x)}')

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return {
            'x_ot': torch.FloatTensor(self.data_x[index]),  # [8, 1]
            'context_text': self.text[index],                # str
            'sample_id': int(self.sample_ids[index])
        }


def data_provider_contrastive(args, flag, train_scaler=None):
    """
    对比学习数据加载器工厂函数

    Args:
        args: 参数对象，需包含:
            - root_path: 数据根目录
            - data_path: CSV文件名
            - batch_size: batch大小
            - num_workers: 数据加载线程数
        flag: 'train'/'val'/'test'
        train_scaler: 训练集的scaler（用于val/test，可选）

    Returns:
        dataset: Dataset对象
        dataloader: DataLoader对象
    """
    dataset = Dataset_Traffic_Contrastive(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        scale=getattr(args, 'scale', True),
        scaler=train_scaler
    )

    shuffle = (flag == 'train')
    drop_last = (flag == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last
    )

    print(f"[contrastive:{flag}] data_x shape: {dataset.data_x.shape}")
    return dataset, dataloader






class Dataset_Traffic_Forecast(Dataset):
    """
    交通数据预测数据集

    与对比学习数据集的区别:
    - 返回 x_enc, x_dec, y (而不是仅 x_ot)
    - x_enc: 输入序列 [seq_len, features]
    - x_dec: 解码器输入 [label_len + pred_len, features] (起始标记)
    - y: 目标序列 [pred_len, features]
    - context_text: 文本描述 (可选)
    """
    def __init__(self, root_path, data_path, flag='train',
                 seq_len=8, label_len=4, pred_len=6,
                 split_ratio=(0.7, 0.1, 0.2), scale=True, scaler=None):
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.scaler = scaler

        # 读取CSV
        file_path = os.path.join(root_path, data_path)
        df = pd.read_csv(file_path)
        print(f"Loading {flag} data from {file_path}")
        print(f"Original samples: {len(df)}")

        # 预处理：删除无效样本（y_ot 为空的样本无法训练/评估）
        def _is_empty(val):
            return pd.isna(val) or str(val).strip() == ''

        if 'y_ot' in df.columns:
            df = df[~df['y_ot'].apply(_is_empty)]

        # 优先使用 split 列；若不存在则回退使用 split_ratio
        if 'split' in df.columns:
            df = df[df['split'] == flag].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        print(f"Valid samples: {len(df)}")
        self.sample_ids = df['sample_id'].tolist() if 'sample_id' in df.columns else list(range(len(df)))

        # 解析时间序列
        import ast
        df['x_ot_parsed'] = df['x_ot'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df['y_ot_parsed'] = df['y_ot'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # 转为numpy数组
        data_x = np.array(df['x_ot_parsed'].tolist())  # [N, seq_len]
        data_y = np.array(df['y_ot_parsed'].tolist())  # [N, pred_len]

        # 添加特征维度
        data_x = data_x[:, :, np.newaxis]  # [N, seq_len, 1]
        data_y = data_y[:, :, np.newaxis]  # [N, pred_len, 1]

        # 获取文本（允许为空）
        text = df['context_text'].fillna('').astype(str).tolist() if 'context_text' in df.columns else [''] * len(df)

        # 若没有 split 列，按照比例切分（保持旧逻辑）
        if 'split' not in df.columns:
            n_samples = len(data_x)
            train_end = int(n_samples * split_ratio[0])
            val_end = int(n_samples * (split_ratio[0] + split_ratio[1]))

            if flag == 'train':
                data_x = data_x[:train_end]
                data_y = data_y[:train_end]
                text = text[:train_end]
                self.sample_ids = self.sample_ids[:train_end]
            elif flag == 'val':
                data_x = data_x[train_end:val_end]
                data_y = data_y[train_end:val_end]
                text = text[train_end:val_end]
                self.sample_ids = self.sample_ids[train_end:val_end]
            else:  # test
                data_x = data_x[val_end:]
                data_y = data_y[val_end:]
                text = text[val_end:]
                self.sample_ids = self.sample_ids[val_end:]

        self.data_x = data_x
        self.data_y = data_y
        self.text = text

        # 标准化
        if self.scale:
            if scaler is None:
                if flag != 'train':
                    raise ValueError("Scaler is None for non-train split. Please provide train scaler.")
                self.scaler = StandardScaler()
                self.scaler.fit(self.data_x.reshape(-1, 1))
            else:
                self.scaler = scaler
            self.data_x = self.scaler.transform(self.data_x.reshape(-1, 1)).reshape(self.data_x.shape)
            self.data_y = self.scaler.transform(self.data_y.reshape(-1, 1)).reshape(self.data_y.shape)
        else:
            self.scaler = None

        print(f'{flag} samples: {len(self.data_x)}')

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        # x_enc: 输入序列 [seq_len, features]
        x_enc = self.data_x[index]  # [seq_len, 1]

        # y: 目标序列 [pred_len, features]
        y = self.data_y[index]  # [pred_len, 1]

        # x_dec: 解码器输入 [label_len + pred_len, features]
        # 前 label_len 来自输入序列的末尾，后 pred_len 初始化为0
        x_dec = np.zeros((self.label_len + self.pred_len, x_enc.shape[-1]))
        x_dec[:self.label_len] = x_enc[-self.label_len:]  # 起始标记

        return {
            'x_enc': torch.FloatTensor(x_enc),       # [seq_len, features]
            'x_dec': torch.FloatTensor(x_dec),       # [label_len + pred_len, features]
            'y': torch.FloatTensor(y),               # [pred_len, features]
            'context_text': self.text[index],        # str
            'sample_id': int(self.sample_ids[index]) # 用于邻居检索
        }


def data_provider(args, flag):
    """
    预测任务数据加载器工厂函数

    Args:
        args: 参数对象，需包含:
            - root_path: 数据根目录
            - data_path: CSV文件名
            - seq_len: 输入序列长度
            - label_len: 起始标记长度
            - pred_len: 预测长度
            - batch_size: batch大小
            - num_workers: 数据加载线程数
            - scale: 是否标准化
        flag: 'train'/'val'/'test'

    Returns:
        dataset: Dataset对象
        dataloader: DataLoader对象
    """
    scale_flag = getattr(args, 'scale', True)
    print("Scale flag（标准化标志）:", scale_flag)

    # 创建数据集
    if flag == 'train':
        dataset = Dataset_Traffic_Forecast(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='train',
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            scale=scale_flag,
            scaler=None
        )
        train_scaler = dataset.scaler if scale_flag else None
    else:
        train_scaler = None
        if scale_flag:
            train_dataset = Dataset_Traffic_Forecast(
                root_path=args.root_path,
                data_path=args.data_path,
                flag='train',
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.pred_len,
                scale=True,
                scaler=None
            )
            train_scaler = train_dataset.scaler

        dataset = Dataset_Traffic_Forecast(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            scale=scale_flag,
            scaler=train_scaler
        )

    shuffle = (flag == 'train')
    drop_last = (flag == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last
    )

    return dataset, dataloader
