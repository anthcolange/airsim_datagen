"""

Author: Jianxiong Cai
"""

import torch

def calc_accuracy(model, testloader, device):
	"""
	# TODO (jianxiong): double check the labels format? is it 1-d or 2-d?
	:param model: The trained model
	:param testloader: The test loader
	:return:
		the accuracy
	"""
	correct = 0
	total = 0

	# set to eval model
	model.eval()

	with torch.no_grad():
		for data in testloader:
			# images, labels = data
			images = data["rgb_images"]
			labels = data["label"]

			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			# print(outputs)
			_, predicted = torch.max(outputs.data, 1)
			# print(predicted)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	# print('Accuracy of the network on the 10000 test images: %d %%' % (
	# 		100 * correct / total))
	accuracy = 100 * correct / total
	# set back to train mode
	model.train()

	return accuracy

