import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN
        }
        self.device = self._acquire_device()
        # self.model = self._build_model().to(self.device)
        try:
            self.model = self._build_model()
            if self.model is not None:
                self.model = self.model.to(self.device)
        except NotImplementedError:
            self.model = None

    def _build_model(self):
        # raise NotImplementedError
        # return None
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    # def _acquire_device(self):
    #     if self.args.use_gpu:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #             self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device
    def _acquire_device(self):
        if self.args.use_gpu:
            # 修改这行，添加默认值防止 AttributeError
            device_ids = getattr(self.args, 'devices', '0') if getattr(self.args, 'use_multi_gpu', False) else str(
                self.args.gpu)
            print(f'Use GPU: cuda:{device_ids}')
            return torch.device('cuda:{}'.format(self.args.gpu))
        else:
            print('Use CPU')
            return torch.device('cpu')

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
