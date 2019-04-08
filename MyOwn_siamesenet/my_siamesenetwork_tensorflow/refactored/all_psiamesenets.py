from pSiameseNetv1 import psiamesenet_v1
from pSiameseNetv2 import psiamesenet_v2
from pSiameseNetv3 import psiamesenet_v3
from pSiameseNetv4 import psiamesenet_v4

mg_nets = [
		{'model_name': 'psv1', 'net': psiamesenet_v1},
		{'model_name': 'psv2', 'net': psiamesenet_v2},
		{'model_name': 'psv3', 'net': psiamesenet_v3},
		{'model_name': 'psv4', 'net': psiamesenet_v4}
]

def getModel(model_name):
		for net in mg_nets:
				if net['model_name'] == model_name:
						return net['net']

