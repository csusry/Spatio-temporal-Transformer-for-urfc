#coding=utf-8
import warnings
import os
import time
from shutil import copyfile

class DefaultConfigs(object): 
    model_name = "Spatio-temporal Transformer"
    num_classes = 9
    img_weight = 100
    img_height = 100
  
    channels = 3
    vis_channels=7
    vis_weight=24
    vis_height=26

    lr = 0.0015
    
    lr_decay = 0.5
    weight_decay =0e-5
    batch_size =16
    epochs = 30
    
    def __init__(self):
        if not os.path.exists("./bak/"):
            os.mkdir("./bak/")
        time_now = time.strftime("%m-%d-%H_%M_%S", time.localtime())
        path = f"bak/{time_now+self.model_name}/"
        os.makedirs(path)
        copyfile("multimodal.py",path+"multimodal.py")
        copyfile("multimain.py",path+"multimain.py")
        copyfile("config.py",path+"config.py")
        print('备份到'+path)
        self.weights = f"{path}checkpoints/"
        self.best_models = f"{path}checkpoints/best_models/"
        self.logs = f"{path}"
        self.debug_file=f'{path}tmp/debug'
        self.submit = f"{path}submit/"
        
    train_data = "./data/train/" # where is your train images data
    test_data = "./data/test/"   # your test data
    train_vis="./data/npy/train_visit"  # where is your train visits data
    test_vis="./data/npy/test_visit"
    load_model_path = None
    
    
def parse(self, kwargs):
    """
    update config by kwargs
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfigs.parse = parse
config = DefaultConfigs()
