import sys
import os
import time
import math
import json
import gzip
import torch
import numpy as np
import random
from torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image

# --- Assumes a domain size of 50 x 50
class MaterialsDataset(Dataset):
    def __init__(self, data_dir, subsample=10000, split='all'):
        self.data_dir = data_dir
        self.subsample = subsample

        examples_filename = os.path.join(self.data_dir,'split.json')
        if not os.path.exists(examples_filename):
            # generate a train/test split -> for now, just take 6 random materials for testing
            examples = []
            for walker in os.walk(self.data_dir):
                base_dir = walker[0]
                for dir_example in walker[1]:
                    metadata = dir_example.split('_')
                    try:
                        item = {'dir':os.path.join(base_dir,dir_example),'volume_fraction':float(metadata[0]),'radius':float(metadata[1])}
                        examples.append(item)
                    #
                    except:
                        print('no material directory',metadata)
                    #
                #
            #
            random.shuffle(examples)
            for idx in range(len(examples)):
                examples[idx]['split'] = 'test' if idx < 200 else 'train'
            #

            json.dump(examples,open(examples_filename,'w'))
        #

        ex = json.load(open(examples_filename,'r'))
        ex = [example for example in ex if split=='all' or example['split']==split]

        self.examples = ex
    #

    def __len__(self):
        #return 100000000
        return len(self.examples)
    #

    def parse_fiber_centers(self,filename):
        fiber_file = open(filename,'r')
        fiber_info = next(fiber_file).rstrip().split(',')
        radius = 1*float(fiber_info[-1])
        point_data = fiber_info[1:-1]
        points = torch.tensor([[float(point_data[2*idx]),float(point_data[2*idx+1])] for idx in range(len(point_data)//2)])
        points[points<0] = points[points<0]+50
        points[points>50] = points[points>50]-50

        unique_ids = []
        for p in range(points.shape[0]-1):
            other_points = points[p+1:]
            dists = ((points[p:p+1]-other_points)**2).sum(dim=1)
            duplicates = (dists < 1e-10).nonzero().squeeze(1)
            if duplicates.shape[0] == 0:
                unique_ids.append(p)
        #
        unique_ids.append(points.shape[0]-1)
        unique_ids = torch.tensor(unique_ids,dtype=torch.long)

        return points[unique_ids],radius
    #

    def parse_mesh(self,filename):
        elems_file = gzip.open(filename,'rt')
        the_lines = [line.rstrip().split() for line in elems_file]
        mesh = torch.tensor([[int(float(tri[1])),int(float(tri[2])),int(float(tri[3]))] for tri in the_lines],dtype=torch.int32)
        return mesh
    #

    def get_fibers(self,example,displacement=None):
        fiber_centers_filename = os.path.join(example['dir'],'fiber_center.txt')
        fiber_points,fiber_radius = self.parse_fiber_centers(fiber_centers_filename)
        if displacement is not None:
            fiber_points[:,0] += displacement
        fiber_points[fiber_points<0] = fiber_points[fiber_points<0]+50
        fiber_points[fiber_points>50] = fiber_points[fiber_points>50]-50
        fibers = torch.zeros(fiber_points.shape[0],3)
        fibers[:,:2] = 2*(fiber_points/50)-1
        fibers[:,2] = 2*(fiber_radius/50)
        #fibers[:,2] = 1*(fiber_radius/50)
        return fibers #[x,y,r]
    #

    def get_fiber_geometry(self,geom):
        geom_filename = geom['filename']
        fiber_points,fiber_radius = self.parse_fiber_centers(geom_filename)
        fibers = torch.zeros(fiber_points.shape[0],3)
        fibers[:,:2] = 2*(fiber_points/50)-1
        fibers[:,2] = fiber_radius/50
        return fibers
    #

    def form_distance_field(self,fibers,res):
        x_spacing = torch.linspace(-1,1,res)
        x_spacing = x_spacing.to(device=fibers.device)
        y_spacing = torch.linspace(-1,1,res)
        y_spacing = y_spacing.to(device=fibers.device)

        grid = torch.stack(torch.meshgrid(x_spacing,y_spacing),dim=0).view(2,-1).T
        fiber_pos = fibers[:,:2]

        diff = grid.unsqueeze(1)-fiber_pos.unsqueeze(0)
        periodic_diff = diff.unsqueeze(-1)+torch.tensor([2,0,-2],device=fibers.device).view(1,1,1,3)
        sqd_diff = periodic_diff**2
        periodic_ssd,_ = sqd_diff.min(dim=-1)
        fiber_dists,_ = (torch.sqrt(periodic_ssd.sum(dim=-1))-fibers[0,-1]).min(dim=1)

        fiber_dists[fiber_dists<0] = 0
        fiber_dists[fiber_dists>0] = 1

        return fiber_dists.view(res,res)
    #

    def has_mesh(self,m):
        return False
    #

    def get_mesh(self,example):
        mesh_filename = os.path.join(example['dir'],'elems.txt.gz')
        mesh_tensor_filename = os.path.join(example['dir'],'mesh.th')
        if os.path.exists(mesh_tensor_filename):
            mesh = torch.load(mesh_tensor_filename)
        else:
            mesh = self.parse_mesh(mesh_filename)
            torch.save(mesh,mesh_tensor_filename)
        #
        return mesh
    #

    def get_node_positions(self,example):
        positions_tensor_filename = os.path.join(example['dir'],'IntPoint_positions.th')
        node_positions = torch.load(positions_tensor_filename)
        return node_positions
    #

    def get_resampled_result(self,example,result_name = 'resampled_damage.th'):
        filename = os.path.join(example['dir'],result_name)
        return torch.load(filename)
    #

    def get_node_result(self,example):
        tensor_filename = os.path.join(example['dir'],'damage.th')
        node_result = torch.load(tensor_filename)
        return node_result 
    #

    def __getitem__(self,idx):
        proper_id = idx%len(self.examples)
        np.random.seed(seed=int(time.time() + proper_id))
        example = self.examples[proper_id]

        # --- fiber centers
        # fibers = self.get_fibers(example)

        # --- distance field
        # distance_field = self.form_distance_field(fibers,512)
        distance_field = self.get_resampled_result(example,result_name = 'distance_field.th')

        # --- stress field
        stress_field = self.get_resampled_result(example,result_name ='max_f.th')

        return idx,distance_field,stress_field
    #
#

if __name__=='__main__':
    data_path = "Results_values"
    dataset = MaterialsDataset(data_path,split='all')
    example = dataset.examples[0]
    stress_field = dataset.get_resampled_result(example,result_name ='max_f.th')
    print(stress_field)
    
#
