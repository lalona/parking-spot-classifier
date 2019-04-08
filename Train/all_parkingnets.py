from malexnet import mAlexNet
from parkingnetv1 import create_parkingnet as pnv1
from parkingnetv1_2 import create_parkingnet as pnv1_2
from parkingnetv2 import create_parkingnet as pnv2
from parkingnetv2_2 import create_parkingnet as pnv2_2
from parkingnetv3 import create_parkingnet as pnv3
from parkingnetv3_2 import create_parkingnet as pnv3_2
from parkingnetv4 import create_parkingnet as pnv4
from parkingnetv4_2 import create_parkingnet as pnv4_2
from parkingnetv5 import create_parkingnet as pnv5
from parkingnetv6 import create_parkingnet as pnv6
from parkingnetv7 import create_parkingnet as pnv7
from parkingnetv8 import create_parkingnet as pnv8
from parkingnetv9 import create_parkingnet as pnv9
from parkingnetv10 import create_parkingnet as pnv10
from siamese_net import siamesenet
from siamese_net2 import siamesenet2
from siamese_net3 import siamesenet3
from siamese_net4 import siamesenet4
from siamese_net5 import siamesenet5

mg_nets = [
		{'model_name': 'malexnet', 'net': mAlexNet.build},
		{'model_name': 'siamesenet', 'net': siamesenet.build},
		{'model_name': 'siamesenet2', 'net': siamesenet2.build},
		{'model_name': 'siamesenet3', 'net': siamesenet3.build},
		{'model_name': 'siamesenet4', 'net': siamesenet4.build},
		{'model_name': 'siamesenet5', 'net': siamesenet5.build},
		{'model_name': 'pnv1', 'net': pnv1},
		{'model_name': 'pnv1_2', 'net': pnv1_2},
		{'model_name': 'pnv2', 'net': pnv2},
		{'model_name': 'pnv2_2', 'net': pnv2_2},
		{'model_name': 'pnv3', 'net': pnv3},
		{'model_name': 'pnv3_2', 'net': pnv3_2},
		{'model_name': 'pnv4', 'net': pnv4},
		{'model_name': 'pnv4_2', 'net': pnv4_2},
		{'model_name': 'pnv5', 'net': pnv5},
		{'model_name': 'pnv6', 'net': pnv6},
		{'model_name': 'pnv7', 'net': pnv7},
		{'model_name': 'pnv8', 'net': pnv8},
		{'model_name': 'pnv9', 'net': pnv9},
		{'model_name': 'pnv10', 'net': pnv10}
]

def getModel(model_name):
		for net in mg_nets:
				if net['model_name'] == model_name:
						return net['net']

