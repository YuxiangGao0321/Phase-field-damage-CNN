import os

def job2task(path = 'joblist.txt'):
    task_list=[]
    with open("joblist.txt") as jf:
        for line in jf:
            dataline = line.split(",")
            if dataline[0] == "vf":
                vf_list = dataline[1:]
            elif dataline[0] == "R":
                R_list = dataline[1:]
            elif dataline[0] == "n":
                # each_num = dataline[1]
                idx_start = int(dataline[1])
                idx_end = int(dataline[2])
                n_list = [i for i in range(idx_start+1,idx_end+1)]
            else:
                continue
    print("vf:")
    print(vf_list)
    print("R:")
    print(R_list)
    print("idx:{},{}".format(idx_start,idx_end))
    for volfrac in vf_list:
        for R in R_list:
            for num in n_list:
                task_list.append([float(volfrac),float(R),num])
    return task_list


def my_mkdir(directory):
	if not os.path.exists(directory):
	    os.makedirs(directory)


def plot_dmg(max_info_1,max_value_1,results_title):
    from data_dmg import MaterialsDataset
    from net import UNet
    import torch
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    import matplotlib.colors as colors
    from matplotlib import cm
    model_file = 'UNet_l1.pth'
    data_file = 'Results_peak'
    results_path = 'output_results/'

    unet = UNet(1)
    state_dict = torch.load(os.path.join('model',model_file))['state_dict']
    unet.load_state_dict(state_dict)
    unet.eval()

    dataset = MaterialsDataset(data_file,split='test')
    for material in dataset.examples:
        if material['dir'].split('\\')[-1] == max_info_1:
            fibers = dataset.get_fibers(material)
            distance_field = dataset.form_distance_field(fibers,512).unsqueeze(0).unsqueeze(0)
            ground_truth = dataset.get_resampled_result(material)
            with torch.no_grad(): 
                predicted_field = unet(distance_field).squeeze().detach()
            break     

    value = str(np.around(max_value_1,3))
    s_real = np.array(ground_truth.squeeze())
    s_pred = np.array(predicted_field)
    fig = plt.figure(figsize = (20,5))
    plt.rcParams.update({'font.size': 20})
    plt.subplot(1,3,3)
    diff = s_pred - s_real
    fmax = float(np.format_float_positional(diff.max(),precision = 2,unique=False, fractional=False, trim='k'))
    fmin = float(np.format_float_positional(diff.min(),precision = 2,unique=False, fractional=False, trim='k'))
    ct = (fmin, 0.5*fmin, 0, 0.5*fmax, fmax)
    im_diff = plt.imshow(diff,
        norm=colors.TwoSlopeNorm(vcenter=0,vmin = fmin,vmax = fmax),
        cmap = cm.RdBu_r)
    plt.title('The difference (NMSE:'+ value +')')
    plt.axis('off')
    # cb_diff = plt.colorbar(im_diff)
    # cb_diff.ax.locator_params(nbins=6)
    plt.colorbar(im_diff,ticks=ct)
    plt.tight_layout(pad = 0.6)
    # ## -----------------------------------
    cm_field = cm.Reds
    plt.subplot(1,3,2)
    plt.imshow(s_real,vmin=0, vmax=1,cmap = cm_field)
    plt.title("Grounded truth")
    plt.axis('off')
    # ## -----------------------------------
    plt.subplot(1,3,1)
    im = plt.imshow(s_pred,vmin=0, vmax=1,cmap = cm_field)
    # plt.colorbar()
    plt.title("Model prediction")
    plt.axis('off')
    plt.subplots_adjust(wspace = -0.65,left = -0.2)
    mpl.rcParams['figure.dpi'] = 200
    # ## -----------------------------------
    cb_ax = fig.add_axes([0.52, 0.07, 0.011, 0.845])
    fig.colorbar(im, cax = cb_ax)
    plt.savefig(results_path + 'jpg/'+results_title+'{}-{}.jpg'.format(max_info_1,value),dpi = 200)
    plt.savefig(results_path + 'pdf/'+results_title+'{}-{}.pdf'.format(max_info_1,value),dpi = 200)
    # plt.show()
    plt.close()

if __name__ == "__main__":
	# print(job2task())
    print(norm_kendall_tau_dist([1,2,3,4],[2,1,3,4]))