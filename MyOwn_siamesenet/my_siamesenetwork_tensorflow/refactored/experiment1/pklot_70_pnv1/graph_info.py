import json
import matplotlib.pyplot as plt

def main():
		with open('info.json') as f:
				info = json.load(f)
		loss_history = info['loss_history']
		loss = [float(l) * 1000 for l in loss_history]
		plt.plot(loss)
		plt.ylabel('some numbers')
		plt.show()

if __name__ == "__main__":
		main()