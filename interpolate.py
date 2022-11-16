import torch
import numpy as np
from PIL import Image
import time

import sys
# from data_dmg import MaterialsDataset
# from data_disp import MaterialsDataset

'''
* `grid`: D x W x H
* `positions`: B x 2 \in [0,1]

* output: B x D
'''
def bilinear_interpolation(grid,positions):
    B = positions.shape[0]

    res = torch.tensor([grid.shape[1],grid.shape[2]],device=positions.device).unsqueeze(0) # 1 x 2
    lattice = (res-1)*positions

    # -> give us a B x 2 x 2 tensor of control points
    floorpunch = torch.floor(lattice).long()
    x_stencil = torch.tensor([ [0,0],[1,1] ],dtype=torch.long,device=positions.device)
    y_stencil = torch.tensor([ [0,1],[0,1] ],dtype=torch.long,device=positions.device)
    x_control = floorpunch[:,0:1].unsqueeze(2)+x_stencil.unsqueeze(0)
    y_control = floorpunch[:,1:2].unsqueeze(2)+y_stencil.unsqueeze(0)

    # clamp
    x_control[x_control<0] = 0
    x_control[x_control>=res[0,0]] = res[0,0]-1
    y_control[y_control<0] = 0
    y_control[y_control>=res[0,1]] = res[0,1]-1

    # -> access values in grid, yielding D x B x 2 x 2 control points
    F_grid = grid[:,x_control,y_control]

    # interpolation matrices
    x_coords = torch.ones(B,2,device=positions.device)
    x_coords[:,0] = lattice[:,0]-floorpunch[:,0].float() # vary in position, B x 2
    y_coords = torch.ones(B,2,device=positions.device)
    y_coords[:,0] = lattice[:,1]-floorpunch[:,1].float() # vary in position, B x 2

    basis = torch.tensor([[-1,1],[1,0]],dtype=positions.dtype,device=positions.device) # 2 x 2

    x_interpolation = x_coords.mm(basis).unsqueeze(0).unsqueeze(3)
    y_interpolation = y_coords.mm(basis).unsqueeze(0).unsqueeze(2)
    full_interpolation = x_interpolation*y_interpolation
    # print('full interpolation',full_interpolation.shape)
    interpolated_grid = (full_interpolation*F_grid).sum(dim=(2,3))

    return interpolated_grid
#

def interpolate_to_grid(x_res,y_res,positions,field):
    B = positions.shape[0]

    res = torch.tensor([x_res,y_res],device=positions.device).unsqueeze(0) # 1 x 2
    lattice = (res-1)*positions

    # -> give us a B x 2 x 2 tensor of control points
    floorpunch = torch.floor(lattice).long()
    x_stencil = torch.tensor([ [0,0],[1,1] ],dtype=torch.long,device=positions.device)
    y_stencil = torch.tensor([ [0,1],[0,1] ],dtype=torch.long,device=positions.device)
    x_control = floorpunch[:,0:1].unsqueeze(2)+x_stencil.unsqueeze(0)
    y_control = floorpunch[:,1:2].unsqueeze(2)+y_stencil.unsqueeze(0)

    # clamp
    x_control[x_control<0] = 0
    x_control[x_control>=res[0,0]] = res[0,0]-1
    y_control[y_control<0] = 0
    y_control[y_control>=res[0,1]] = res[0,1]-1

    # interpolation matrices
    x_coords = torch.ones(B,2,device=positions.device)
    x_coords[:,0] = lattice[:,0]-floorpunch[:,0].float() # vary in position, B x 2
    y_coords = torch.ones(B,2,device=positions.device)
    y_coords[:,0] = lattice[:,1]-floorpunch[:,1].float() # vary in position, B x 2

    basis = torch.tensor([[-1,1],[1,0]],dtype=positions.dtype,device=positions.device) # 2 x 2

    x_interpolation = x_coords.mm(basis).unsqueeze(0).unsqueeze(3)
    y_interpolation = y_coords.mm(basis).unsqueeze(0).unsqueeze(2)
    full_interpolation = (x_interpolation*y_interpolation)

    grid_field = full_interpolation*field.unsqueeze(-1).unsqueeze(-1)
    flat_grid_field = torch.reshape(grid_field,(grid_field.shape[0],grid_field.shape[1]*grid_field.shape[2]*grid_field.shape[3]))
    # print(flat_grid_field.shape)

    x_control = x_control.view(-1)
    y_control = y_control.view(-1)
    control = y_control+y_res*x_control
    full_interpolation = full_interpolation.view(-1)

    normalization = 1e-12+torch.zeros(x_res*y_res,device=positions.device)
    normalization.scatter_add_(0,control,full_interpolation)

    # For the damage field flat_grid_field.shape[0] = 1
    n_field = flat_grid_field.shape[0]
    grid_field_list = []
    for idx_field in range(n_field):
        interpolated_grid = torch.zeros(x_res*y_res,device=positions.device)
        interpolated_grid.scatter_add_(0,control,flat_grid_field[idx_field])
        grid_field_list.append(interpolated_grid)
    # interpolated_grid_1 = torch.zeros(x_res*y_res,device=positions.device)
    # interpolated_grid_1.scatter_add_(0,control,flat_grid_field[0])
    # interpolated_grid_y = torch.zeros(x_res*y_res,device=positions.device)
    # interpolated_grid_y.scatter_add_(0,control,flat_grid_field[1])
    # interpolated_grid_z = torch.zeros(x_res*y_res,device=positions.device)
    # interpolated_grid_z.scatter_add_(0,control,flat_grid_field[2])

    normalization = normalization.view(1,x_res,y_res)
    # full_grid = torch.stack((interpolated_grid_x,interpolated_grid_y,interpolated_grid_z),dim=0).view(3,x_res,y_res) / normalization
    # full_grid = interpolated_grid_x.view(1,x_res,y_res) / normalization
    full_grid = torch.stack(grid_field_list,dim=0).view(n_field,x_res,y_res) / normalization
    '''
    normalization = torch.zeros(1,x_res,y_res,device=positions.device)
    interpolated_grid = torch.zeros(field.shape[0],x_res,y_res,device=positions.device)
    for x,y,a,f in zip(x_control.view(-1),y_control.view(-1),full_interpolation.view(-1),flat_grid_field):
        normalization[0,x,y]+=a
        interpolated_grid[:,x,y]+=f
    #
    interpolated_grid /= normalization
    '''

    return full_grid
#


# if __name__=='__main__':
#     '''
#     positions = torch.rand(B,2)
#     rand_field = torch.rand(D,B)
#     rand_grid = torch.rand(D,W,H)

#     tick = time.time()
#     interp_func = bilinear_interpolation(rand_grid,positions)
#     print('interpolate to grid...',rand_field.min(),rand_field.max())
#     interpolated_grid = interpolate_to_grid(W,H,positions,rand_field)
#     tock = time.time()
#     print('time to interpolate',(tock-tick),interp_func.shape)
#     print('interpolated grid',interpolated_grid.shape,interpolated_grid.min(),interpolated_grid.max())
#     '''

#     dataset = MaterialsDataset(sys.argv[1],split='all')
#     example = dataset.examples[0]
#     displacement_field = dataset.get_node_displacement(example).cuda().T
#     # displacement_field = dataset.get_node_stress(example).T
#     print('displacement',displacement_field.shape,displacement_field.min(),displacement_field.max())
#     node_positions = dataset.get_node_positions(example)/50
#     print('node positions',node_positions.shape,node_positions.min(),node_positions.max())

#     W = 512
#     H = 512

#     tick = time.time()
#     interpolated_grid = interpolate_to_grid(W,H,node_positions,displacement_field).cpu()
#     tock = time.time()
#     print('time to interpolate to grid',(tock-tick),interpolated_grid.shape)
#     interpolated_x_grid = (340*(interpolated_grid[0]-interpolated_grid[0].min())).char().numpy()
#     print(interpolated_x_grid.shape)
#     pil_arr = np.zeros((W,H,3),dtype=np.uint8)
#     print(pil_arr.shape)
#     pil_arr[:,:,0] = interpolated_x_grid
#     pil_arr[:,:,1] = interpolated_x_grid
#     pil_arr[:,:,2] = interpolated_x_grid
#     # Image.fromarray(pil_arr).save('img.png')
#     # interpolated_x_grid_im = interpolated_x_grid.astype('uint8')
#     # Image.fromarray(interpolated_x_grid_im).save('img1.png')
# #
