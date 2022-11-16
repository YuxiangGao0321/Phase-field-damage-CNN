import argparse
import os
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from net import UNet
from net import Image2value2
from interpolate import bilinear_interpolation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', required=True, help='path to materials dataset')

    parser.add_argument('--batchSize', type=int, default=256000, help='batch size for coordinates')

    parser.add_argument('--start_lr', type=float, default=2e-4, help='learning rate, default=1e-4')
    parser.add_argument('--end_lr', type=float, default=0, help='learning rate, default=1e-4')
    parser.add_argument('--n_passes', type=float, default=60000, help='number of passes to make, default=10000')
    parser.add_argument('--pass_decay', type=float, default=10000, help='frequency at which to decay learning rate')
    parser.add_argument('--lr_decay', type=float, default=.2, help='learning rate decay, default=.2')
    parser.add_argument('--warmup_pass', type=float, default=10000, help='number of passes to make, default=10000')

    parser.add_argument('--model_dir', default='model', help='directory to write out model')
    parser.add_argument('--field', default='value', help='type of field to predict (stress or displacement)')

    parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
    parser.set_defaults(cuda=False)


    opt = parser.parse_args()
    print(opt)


    opt.device = 'cuda' #if opt.cuda else 'cpu'
    from data_load_pred_dmg import MaterialsDataset as TheDataset
    d_in = 1
    #

    if not os.path.exists(opt.model_dir):
        os.mkdir(opt.model_dir)
    #
    data_path = 'Results'
    # dataset = TheDataset(opt.data,subsample=opt.batchSize,split='train')
    dataset = TheDataset(data_path,subsample=opt.batchSize,split='train')
    dataloader = DataLoader(dataset, batch_size=30, num_workers=4, shuffle=True)


    dataset_testing = TheDataset(data_path,subsample=opt.batchSize,split='test')
    dataloader_testing = DataLoader(dataset_testing, batch_size=30, num_workers=4, shuffle=True)

    # --- UNet
    # discr = PatchDiscriminator(d_in)
    img2peak = Image2value2(d_in)

    if opt.cuda:
        img2peak.cuda()
    #
    img2peak.train()

    # optimization
    model_optimizer = optim.Adam(img2peak.parameters(), lr=opt.start_lr, betas=(0.9, 0.999), weight_decay=0.0)

    # loss
    MSE_loss = nn.MSELoss()
    # l1_loss = nn.L1Loss()
    if opt.cuda:
        MSE_loss.cuda()
        # l1_loss.cuda()
    #

    n_iter = 1
    n_epochs = 0

    total_loss_curve = []
    loss_curve =[]#storing the history of the loss
    testing_error_curve = []#storing the history of the testing error

    while True:
        for (mdx,distance,field) in dataloader:
            if opt.cuda:
                distance = distance.cuda()
                field = (field).cuda()
            #

            # -- Generator(Unet)

            # --- zero out gradienrts
            model_optimizer.zero_grad()

            # --- push distance fields through UNet
            # distance = distance.unsqueeze(1)
            predicted_value = img2peak(distance)#.squeeze()

            # print(field,field.shape)
            # print(predicted_value,predicted_value.shape)
            # exit()

            # --- loss
            field = field.unsqueeze(1)
            mse_value = MSE_loss(predicted_value,field)
            # l1_value = l1_loss(predicted_value,field)
            # MSE = mse_value*0.5 + l1_value * 0.5
            MSE = mse_value
            # unet_loss = BCE_loss(discr_gen,torch.ones_like(discr_gen))

            # --- Training loss

            # --- backprop
            MSE.backward()

            # --- step
            model_optimizer.step()

            print('loss[',n_iter,']',"MSE:",MSE.item(),'->',mdx.shape)
            # print('loss[',n_iter,']',"total:",MSE.item(),"mse:",mse_value.item(),"l1:",l1_value.item(),'->',mdx.shape)
            # loss_curve.append(mse_value.item())
            loss_curve.append(MSE.item())

            if n_iter%100==0 or n_iter==1:
                img2peak.eval()
                with torch.no_grad():
                    MSE_sum = []
                    for (mdx_t,distance_t,field_t) in dataloader_testing:
                        if opt.cuda:
                            distance_t = distance_t.cuda()
                            field_t = (field_t).cuda()
                        # predicted_fields_t = img2peak(distance_t.unsqueeze(1)).squeeze()
                        predicted_fields_t = img2peak(distance_t).squeeze()
                        MSE_testing = MSE_loss(predicted_fields_t,field_t)
                        # MSE_testing = l1_loss(predicted_fields_t,field_t)
                        MSE_sum.append(MSE_testing.item())
                testing_error_curve.append(np.array(MSE_sum).mean())
                # test_loss_step.append(n_iter)
                print('n_iter:',n_iter,"Test:",testing_error_curve[-1])
                # torch.save({'state_dict':unet.state_dict(),'loss':loss_curve,'test':testing_error_curve},os.path.join(opt.model_dir,'cGAN.pth'))
                img2peak.train()

            if n_iter%100==0:
                torch.save({'state_dict':img2peak.state_dict(),'training_loss':loss_curve,'test_loss':testing_error_curve},
                    os.path.join(opt.model_dir,'dmg2peak_pred_dmg.pth'))           
            #,'test':testing_error_curve
            #

            # --- learning rate decay --- #
            if n_iter%opt.pass_decay==0:
                print('------ learning rate decay ------',n_iter)
                for param_group in model_optimizer.param_groups:
                    param_group['lr'] *= opt.lr_decay
                #
            #



            n_iter+=1
            if n_iter==opt.n_passes:
                break
            #
        #
        
        n_epochs+=1
        if n_iter==opt.n_passes:
            break
        #
    #
#
