import os
import random
import torch
import numpy as np
import argparse
import json
import time
import math

from net import UNet
from net import Image2value2
from interpolate import bilinear_interpolation
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import cm

# def evaluate_material(mdx):
#     material = dataset.examples[mdx]
#     fibers = dataset.get_fibers(material).cuda()

#     distance_field = dataset.form_distance_field(fibers,512).unsqueeze(0).unsqueeze(0)
#     with torch.no_grad():
#         predicted_field = unet(distance_field).squeeze().detach()

#     node_positions = dataset.get_node_positions(material).cuda()/50
#     if field_type=='stress':
#         node_field = dataset.get_node_stress(material).cuda()
#     elif field_type=='displacement':
#         node_field = dataset.get_node_displacement(material).cuda()
#     elif field_type=='damage':
#         node_field = dataset.get_node_result(material).cuda()
#         predicted_field = predicted_field.unsqueeze(0)
#     # print(predicted_field.shape,predicted_field.type())
#     # exit()
#     #
#     interpolated_field = bilinear_interpolation(predicted_field,node_positions).T
#     sse = ((node_field-interpolated_field)**2).sum().cpu().item()
#     Normalization_factor = (node_field**2).sum().cpu().item()/node_positions.shape[0]

#     mse = sse / node_positions.shape[0]
#     nmse = mse / Normalization_factor
#     print('MSE',material['volume_fraction'],material['radius'],mse,nmse)

#     return mse,nmse

def output_prediction(mdx):
    material = dataset.examples[mdx]

    distance_field = dataset.get_distance_field(material)
    ground_truth = dataset.get_resampled_result(material)
    with torch.no_grad(): 
        predicted_field = unet(distance_field.unsqueeze(0).unsqueeze(0).cuda()).squeeze().detach()
    return predicted_field,ground_truth.squeeze()

def output_ground_truth(mdx):
    material = dataset.examples[mdx]
    ground_truth = dataset.get_resampled_result(material)
    return ground_truth
if __name__=='__main__':

    from data_dmg import MaterialsDataset
    d_out = 1
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True, help='material dataset directory')
    parser.add_argument('--D', type=int, default=128, help='latent code dimension')
    parser.add_argument('--n_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--model_dir', default='model', help='directory to write out model')
    parser.add_argument('--eval_out', default='evaluation.json', help='path that contains evaluation results')
    opt = parser.parse_args()

    #model = opt.model_dir[:-1] if opt.model_dir[-1]=='/' else opt.model_dir
    # model = 'cGAN'

    evl_Rf = 4.0

    result_path = 'output_results_{}/'.format(int(evl_Rf))
    evl_set = 'test'
    model_file = 'UNet_l1.pth'
    dmg2peak_file = "dmg2peak_pred_dmg.pth"
    data_path = 'Results'

    dataset = MaterialsDataset(data_path,split=evl_set)
    # all_materials = []
    
    # for mdx in range(len(dataset.examples)):
    #     material = dataset.examples[mdx] # {'dir': 'Results_sigs/Results_sigs/0.4_4.0_73', 'volume_fraction': 0.4, 'radius': 4.0, 'split': 'test'}
    #     fibers = dataset.get_fibers(material).tolist() # [[x1,x2,r1],[x2,y2,r2],...]
    #     all_materials.append(fibers)
    #

    
    # if model == "cGAN":
    #     d_out = d_out +1 
    #
    ## -----------------------------------
    unet = UNet(d_out)
    state_dict_unet = torch.load(os.path.join(opt.model_dir,model_file))['state_dict']
    unet.load_state_dict(state_dict_unet)
    unet.eval()
    unet.cuda()

    dmg2peak = Image2value2(d_out)
    state_dict_dmg = torch.load(os.path.join(opt.model_dir,dmg2peak_file))['state_dict']
    dmg2peak.load_state_dict(state_dict_dmg)
    dmg2peak.eval()
    dmg2peak.cuda()
    ## -----------------------------------
    do_eval = True
    output_mse = True
    plot_diff = True
    if do_eval:
        from my_lib import my_mkdir
        my_mkdir(result_path + 'test')
        my_mkdir(result_path + 'train')
        if output_mse:
            if evl_set == 'test':
                rmse_list = []
                rnmse_list = []
            with open(result_path+evl_set+"/mse_"+evl_set+".txt",'w+'):
                pass
        # all_eval = []
        # if os.path.exists(opt.eval_out):
        #     all_eval = json.load(open(opt.eval_out,'r'))
        #
        # tick = time.time()
        load_pred_list = []
        peak_load_list = []
        test_info = ''
        for mdx in range(len(dataset.examples)):
            # vf = dataset.examples[mdx]['volume_fraction']
            Rf = dataset.examples[mdx]['radius']
            # if  vf == 0.1 or vf == 0.6:
            #     pass
            # else:
            #     continue
            if Rf == evl_Rf:
                pass
            else:
                continue
            info = str(dataset.examples[mdx]['dir'].split("\\")[-1])
            fig_info = str("dmg_"+info)
            peak_load_info = 'IntD/{}/maxf.txt'.format(info)
            peak_load_list.append(np.loadtxt(peak_load_info))
            print(info)
            idx = info.split("_")[-1]
            ## -----------------------------------
            predicted_field,ground_truth = output_prediction(mdx)
            s_pred = predicted_field.cpu()
            s_real = ground_truth.cpu()
            ## -----------------------------------
            # ground_truth = output_ground_truth(mdx)
            rmse = np.sqrt((s_pred-s_real).square().mean())
            Normalization_factor = np.sqrt(s_real.square().mean())
            rnmse = rmse/Normalization_factor
            if output_mse:
                if evl_set == 'test':
                    rmse_list.append(rmse)
                    rnmse_list.append(rnmse)                   
                with open(result_path+evl_set+"/rmse_"+evl_set+".txt",'a+') as f:
                    f.write(info+","+str(rmse)+","+str(rnmse)+"\n")
            # ## -----------------------------------

            # ## -----------------------------------
            # ## -----------------------------------
            if plot_diff == True:
                s_real = np.rot90(np.array(s_real.squeeze()),2).T
                s_pred = np.rot90(np.array(s_pred),2).T
                fig = plt.figure(figsize = (20,5))
                plt.rcParams.update({'font.size': 20})
                plt.subplot(1,3,3)
                diff = s_pred - s_real
                # diff = np.abs(s_real-s_pred)
                # plt.imshow(diff,vmin=0, vmax=1)
                fmax = float(np.format_float_positional(diff.max(),precision = 2,unique=False, fractional=False, trim='k'))
                fmin = float(np.format_float_positional(diff.min(),precision = 2,unique=False, fractional=False, trim='k'))
                ct = (fmin, 0.5*fmin, 0, 0.5*fmax, fmax)
                im_diff = plt.imshow(diff,
                    norm=colors.TwoSlopeNorm(vcenter=0,vmin = fmin,vmax = fmax),
                    cmap = cm.RdBu_r)
                value = str(np.around(float(rnmse),5))
                plt.title('The difference (NRMSE:'+ value +')')
                plt.axis('off')
                plt.colorbar(im_diff,ticks=ct)
                plt.tight_layout(pad = 0.6)
                # ## -----------------------------------
                cm_field = cm.binary
                plt.subplot(1,3,2)
                plt.imshow(s_real,vmin=0, vmax=1,cmap = cm_field)
                plt.title("Ground truth")
                plt.axis('off')
                # plt.savefig("figs/"+evl_set+"/real_"+info+".png")
                # plt.close()
                # ## -----------------------------------
                plt.subplot(1,3,1)
                im = plt.imshow(s_pred,vmin=0, vmax=1,cmap = cm_field)
                # plt.colorbar()
                # plt.title("pred_"+info)
                plt.title("Model prediction")
                plt.axis('off')
                plt.subplots_adjust(wspace = -0.65,left = -0.2)
                mpl.rcParams['figure.dpi'] = 200
                # ## -----------------------------------
                cb_ax = fig.add_axes([0.52, 0.07, 0.011, 0.845])
                fig.colorbar(im, cax = cb_ax)
                plt.savefig(result_path+evl_set+"/"+info+".jpg")
                plt.close()
            
            # ## -----------------------------------
            with torch.no_grad():
                pred_load = dmg2peak(predicted_field.unsqueeze(0).unsqueeze(0)).squeeze().detach().cpu()

            load_pred_list.append(pred_load)
            test_info = test_info + info + ','
            # if mdx > 15:
            #     break
        peak_load_list,load_pred_list = np.array(peak_load_list),np.array(load_pred_list)
        np.savetxt(result_path+'maxf_list_'+evl_set+'.txt',peak_load_list)
        np.savetxt(result_path+'maxf_pred_'+evl_set+'.txt',load_pred_list)
        # print(test_info)
        with open(result_path+evl_set+'_info.txt','w+') as f:
            f.write(test_info)
        
        if evl_set == 'test' and output_mse:
            np.savetxt(result_path+'rmse.txt',rmse_list)
            np.savetxt(result_path+'rnmse.txt',rnmse_list)
            with open(result_path+"/error_avg.txt",'w+') as f:
                f.write("avg rmse: {}, avg nrmse: {}".format(np.mean(rmse_list),np.mean(rnmse_list)))
                f.write("\npeak load rmse:{}".format(np.sqrt(((peak_load_list-load_pred_list)**2).mean())))
                f.write("\npeak load nrmse:{}".format(np.sqrt(((peak_load_list-load_pred_list)**2/peak_load_list**2).mean())))
    else:
        print("error")
    #
#
