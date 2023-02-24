
import numpy as np
from easydict import EasyDict as edict

__C = edict()
NNcfg = __C

__C.CHECKPOINT_DIR = ''
__C.DATASET_NAME = ''
__C.CONFIG_NAME = ''

__C.GPU_ID = ''
__C.CUDA = False
__C.ngpu = 1

__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 2048
__C.TRAIN.MAX_EPOCH = 40
__C.TRAIN.TRAIN_ITER = 6
__C.TRAIN.NORM = 'Standard'

__C.DATA_GENERATOR = edict()
__C.DATA_GENERATOR.FLAG = False

__C.MODEL = edict()
__C.MODEL.lr = 0.01
__C.MODEL.lr_decay = 0.96
__C.MODEL.steplr_decay = 0.5
__C.MODEL.NUM_INPUT = 3
__C.MODEL.NUM_NODES = 10







def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering theoptions in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
        # IF ERROR, CHANGE ABOVE LINE TO "yaml_cfg = edict(yaml.safe_load(f))"

    _merge_a_into_b(yaml_cfg, __C)
