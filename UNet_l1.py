import argparse
import os
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net import UNet
from net import PatchDiscriminator
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
    parser.add_argument('--field', default='damage', help='type of field to predict (stress or displacement)')

    parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
    parser.set_defaults(cuda=False)


    opt = parser.parse_args()
    print(opt)


    opt.device = 'cuda' #if opt.cuda else 'cpu'
    if opt.field=='damage':
        from data_dmg import MaterialsDataset as TheDataset
        d_out = 1
    elif opt.field=='stress':
        from data_stress import MaterialsDataset as TheDataset
        d_out = 3
    elif opt.field=='displacement':
        from data_disp import MaterialsDataset as TheDataset
        d_out = 2
    elif opt.field=='damage_stress':
        from data_dmg import MaterialsDataset as TheDataset
        d_out = 4
    #

    if not os.path.exists(opt.model_dir):
        os.mkdir(opt.model_dir)
    #

    data_path = 'Results'


    examples_filename = os.path.join(data_path,'split.json')
    if not os.path.exists(examples_filename):
        dataset = TheDataset(data_path,split='all')


    dataset = TheDataset(data_path,subsample=opt.batchSize,split='train')
    dataloader = DataLoader(dataset, batch_size=30, num_workers=8, shuffle=True)

    dataset_testing = TheDataset(data_path,subsample=opt.batchSize,split='test')
    dataloader_testing = DataLoader(dataset_testing, batch_size=30, num_workers=8, shuffle=True)
    # --- UNet
    unet = UNet(d_out)
    # discr = PatchDiscriminator(d_out+1,d_out+1)

    if opt.cuda:
        unet.cuda()
        # discr.cuda()
    #
    unet.train()
    # discr.train()

    # optimization
    unet_optimizer = optim.Adam(unet.parameters(), lr=opt.start_lr, betas=(0.5, 0.999), weight_decay=0.0)
    # discr_optimizer = optim.Adam(discr.parameters(), lr=opt.start_lr, betas=(0.5, 0.999), weight_decay=0.0)

    # loss
    # BCE_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    # MSE_loss = nn.MSELoss()
    # LAMBDA = 10
    if opt.cuda:
        # BCE_loss.cuda()
        l1_loss.cuda()
        # MSE_loss.cuda()
    #

    n_iter = 1
    n_epochs = 0


    loss_curve =[]#storing the history of the loss
    testing_error_curve = []#storing the history of the testing error

    while True:
        for (mdx,distance,field) in dataloader:
            if opt.cuda:
                distance = distance.cuda()
                field = field.cuda()
            #

            #Generator(Unet)

            # --- zero out gradienrts
            unet_optimizer.zero_grad()

            # --- push distance fields through UNet
            distance = distance.unsqueeze(1)
            predicted_fields = unet(distance)

            # --- loss
            # pred_cat = torch.cat([distance,predicted_fields],dim=1) #Concatenate microstructure and stress fields
            # discr_gen = discr(pred_cat)
            l1 = l1_loss(predicted_fields,field)
            # l2 = MSE_loss(predicted_fields,field)
            # Unet_BCE = BCE_loss(discr_gen,torch.ones_like(discr_gen))
            unet_loss = l1
            # unet_loss = LAMBDA*l1+Unet_BCE
            # unet_loss = BCE_loss(discr_gen,torch.ones_like(discr_gen))

            # --- Training loss
            # Unet_BCE = BCE_loss(discr_gen,torch.ones_like(discr_gen))
            # Unet_mse = MSE_loss(predicted_fields,field)

            # --- backprop
            unet_loss.backward()

            # --- step
            unet_optimizer.step()

            #Discriminator

            # --- zero out gradienrts
            # discr_optimizer.zero_grad()

            # --- push predicted_fields through Discriminator
            # predicted_discr = discr(pred_cat)
            # pred_loss = BCE_loss(predicted_discr.detach(),torch.zeros_like(predicted_discr))
            
            # real_cat = torch.cat([distance,field],dim=1)#Concatenate microstructure and stress fields
            # real_discr = discr(real_cat)
            # real_loss = BCE_loss(real_discr,torch.ones_like(real_discr))


            # --- loss
            # discr_loss = BCE_loss(predicted_discr,torch.zeros_like(predicted_discr))+BCE_loss(real_discr,torch.ones_like(real_discr))
            # discr_loss = (pred_loss+real_loss)/2


            # --- backprop
            # discr_loss.backward()

            # --- step
            # discr_optimizer.step()

            # print('loss[',n_iter,']',"Unet:",l1.item(),"(l1)",unet_loss.item(),"(Loss)","Discr:",discr_loss.item(),'->',mdx.shape)
            # loss_curve.append([l1.item(),Unet_BCE.item(),discr_loss.item()])
            print('loss[',n_iter,']',"Unet:",l1.item(),"(l1)",'->',mdx.shape)
            loss_curve.append([l1.item()])


            if n_iter%100==0 or n_iter==1:
                unet.eval()
                with torch.no_grad():
                    MSE_sum = []
                    for (mdx_t,distance_t,field_t) in dataloader_testing:
                        if opt.cuda:
                            distance_t = distance_t.cuda()
                            field_t = field_t.cuda()
                        predicted_fields_t = unet(distance_t.unsqueeze(1))
                        # MSE_testing = MSE_loss(predicted_fields_t,field_t)
                        MSE_testing = l1_loss(predicted_fields_t,field_t)
                        MSE_sum.append(MSE_testing.item())
                testing_error_curve.append(np.array(MSE_sum).mean())
                # test_loss_step.append(n_iter)
                print('n_iter:',n_iter,"Test:",testing_error_curve[-1])
                # torch.save({'state_dict':unet.state_dict(),'loss':loss_curve,'test':testing_error_curve},os.path.join(opt.model_dir,'cGAN.pth'))
                unet.train()

            if n_iter%100==0:
                torch.save({'state_dict':unet.state_dict(),'training_loss':loss_curve,'test_loss':testing_error_curve},
                    os.path.join(opt.model_dir,'UNet_l1.pth'))           
            #,'test':testing_error_curve
            #

            # --- learning rate decay --- #
            if n_iter%opt.pass_decay==0:
                print('------ learning rate decay ------',n_iter)
                for param_group in unet_optimizer.param_groups:
                    param_group['lr'] *= opt.lr_decay
                #
            #

            n_iter+=1
            if n_iter==opt.n_passes:
                break
            #
        #
        # unet.eval()
        # with torch.no_grad():
        #     MSE_sum = []
        #     for (mdx_t,distance_t,field_t) in dataloader_testing:
        #         if opt.cuda:
        #             distance_t = distance_t.cuda()
        #             field_t = field_t.cuda()
        #         predicted_fields_t = unet(distance_t.unsqueeze(1))
        #         MSE_testing = MSE_loss(predicted_fields_t,field_t)
        #         MSE_sum.append(MSE_testing.item())
        # testing_error_curve.append(np.array(MSE_sum).mean())
        # print("Test:",testing_error_curve[-1])
        # # torch.save({'state_dict':unet.state_dict(),'loss':loss_curve,'test':testing_error_curve},os.path.join(opt.model_dir,'cGAN.pth'))
        # unet.train()
        
        n_epochs+=1
        if n_iter==opt.n_passes:
            break
        #
    #
#
