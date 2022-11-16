import torch
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

import os
import sys
from data_dmg import MaterialsDataset
from interpolate import interpolate_to_grid


if __name__=='__main__':
    dataset = MaterialsDataset(sys.argv[1],split='all')
    W = 512
    H = 512
    #for example in tqdm(dataset.examples):
    for example in dataset.examples:
        '''
        resampled_filename = os.path.join(example['dir'],'resampled_displacement.th')
        displacement_field = dataset.get_node_displacement(example).cuda().T
        '''
        # resampled_filename = os.path.join(example['dir'],'resampled_stress.th')
        # field = dataset.get_node_stress(example).cuda().T
        resampled_filename = os.path.join(example['dir'],'resampled_damage.th')
        field = dataset.get_node_result(example).cuda().T
        node_positions = dataset.get_node_positions(example).cuda()/50
        # print(node_positions.shape,field.shape)
        interpolated_grid = interpolate_to_grid(W,H,node_positions,field).cpu()
        print('field',field.min(),field.max(),'grid',interpolated_grid.min(),interpolated_grid.max(),'positions',node_positions.min(),node_positions.max())
        from matplotlib import pyplot as plt
        # plt.imshow(interpolated_grid[3])
        # plt.colorbar()
        # plt.title("interpolated_grid"+example['dir'])
        # plt.savefig("interpolated_grid.png")
        # plt.close()
        # exit()
        torch.save(interpolated_grid,resampled_filename)
    #
#
