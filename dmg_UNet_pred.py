import os
import torch
import numpy as np
from data_dmg import MaterialsDataset
from net import UNet

model_file = 'UNet_l1.pth'
data_path = 'Results'

dataset = MaterialsDataset(data_path,split='all')


unet = UNet(1)
state_dict_unet = torch.load(os.path.join('model',model_file))['state_dict']
unet.load_state_dict(state_dict_unet)
unet.eval()
unet.cuda()

for mdx in range(len(dataset.examples)):
	material = dataset.examples[mdx]
	distance_field = dataset.get_distance_field(material)
	with torch.no_grad(): 
		predicted_field = unet(distance_field.unsqueeze(0).unsqueeze(0).cuda()).squeeze().detach().cpu()
	
	folder = material['dir']
	predicted_field=torch.tensor(np.array(predicted_field.unsqueeze(0))).to(torch.float32)
	torch.save(predicted_field,folder + "/dmg_pred.th")
	print(folder)