from mgooglenetv7 import create_googlenet as mgv7
from mgooglenetv8 import create_googlenet as mgv8
from mgooglenetv10 import create_googlenet as mgv10
from mgooglenetv11 import create_googlenet as mgv11
from mgooglenetv12 import create_googlenet as mgv12
from mgooglenetv13 import create_googlenet as mgv13
from mgooglenetv14 import create_googlenet as mgv14
from mgooglenetv15 import create_googlenet as mgv15
from malexnet import mAlexNet
from mXception import Xception as mxv1
from mXceptionv2 import Xception as mxv2
from mXception3 import Xception as mxv3

mg_nets = [
		{'model_name': 'mgv7', 'net': mgv7},
		{'model_name': 'mgv8', 'net': mgv8},
		{'model_name': 'mgv10', 'net': mgv10},
		{'model_name': 'mgv11', 'net': mgv11},
		{'model_name': 'mgv12', 'net': mgv12},
		{'model_name': 'mgv13', 'net': mgv13},
		{'model_name': 'mgv14', 'net': mgv14},
		{'model_name': 'mgv15', 'net': mgv15},
		{'model_name': 'mAlexNet', 'net': mAlexNet.build},
		{'model_name': 'mxv1', 'net': mxv1},
		{'model_name': 'mxv2', 'net': mxv2},
		{'model_name': 'mxv3', 'net': mxv3}
]

def getModel(model_name):
		for net in mg_nets:
				if net['model_name'] == model_name:
						return net['net']

