import matplotlib.pyplot as plt
import numpy as np


def vis_pxpy_post_train(data1, data2, nstep, save_dir, tag=['student', 'teacher'], pos_per_img_1=None, pos_per_img_2=None, rt=None, loss=None): 

    plt.rcParams.update({'font.size': 20})
    data1 = data1.cpu().detach().numpy()
    data2 = data2.cpu().detach().numpy()
    xs_1 = data1[:,0]
    ys_1 = data1[:,1]
    xs_2 = data2[:,0]
    ys_2 = data2[:,1]

    start1 = 0
    start2 = 0
    if len(pos_per_img_1) == 1:
        plt.figure(figsize=(20, 20))
        
        end1 = start1 + pos_per_img_1[0]*8
        end2 = start2 + pos_per_img_2[0]*8
        plt.scatter(xs_1[start1:end1], ys_1[start1:end1], 50, c='g', edgecolors="none", alpha=0.6, label=tag[0])
        plt.scatter(xs_2[start2:end2], ys_2[start2:end2], 50, c='r', edgecolors="none", alpha=0.6, label=tag[1])
        
        plt.title(f'loss_kd: {loss[0].item():.4f}')
        
    else:
        n_ = 4
        fig, axs = plt.subplots(n_, n_, figsize=(20, 20))
        for i in range(len(pos_per_img_1)):
            i_ = i//4
            j_ = np.mod(i, 4) 
            end1 = start1 + pos_per_img_1[i]*8
            end2 = start2 + pos_per_img_2[i]*8
            axs[i_, j_].scatter(xs_1[start1:end1], ys_1[start1:end1], 50, c='g', edgecolors="none", alpha=0.6, label=tag[0])
            axs[i_, j_].scatter(xs_2[start2:end2], ys_2[start2:end2], 50, c='r', edgecolors="none", alpha=0.6, label=tag[1])
            try:
                axs[i_, j_].set_title(f'loss_kd: {loss[i].item():.4f}')
            except:
                continue

            start1 = end1
            start2 = end2

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{nstep}_img_2d.png")
    plt.close()


def vis_pxpy_post_train_weight(data1, data2, cls1, cls2, nstep, save_dir, tag=['student', 'teacher'], pos_per_img_1=None, pos_per_img_2=None, rt=None, loss=None): #tag = ['post', 'train]

    plt.rcParams.update({'font.size': 20})
    data1 = data1.cpu().detach().numpy()
    data2 = data2.cpu().detach().numpy()
    cls1 = cls1.cpu().detach().numpy()
    cls2 = cls2.cpu().detach().numpy()
    xs_1 = data1[:,0]
    ys_1 = data1[:,1]

    xs_2 = data2[:,0]
    ys_2 = data2[:,1]
    start1 = 0
    start2 = 0
    
    if len(pos_per_img_1) == 1:
        plt.figure(figsize=(20, 20))
        
        end1 = start1 + pos_per_img_1[0]*8
        end2 = start2 + pos_per_img_2[0]*8


        plt.scatter(xs_1[start1:end1], ys_1[start1:end1], s=cls1[start1:end1]*100, c='g', edgecolors="none", alpha=0.3, label=tag[0])
        plt.scatter(xs_2[start2:end2], ys_2[start2:end2], s=cls2[start2:end2]*100, c='r', edgecolors="none", alpha=0.3, label=tag[1])
        
        plt.title(f'loss_kd: {loss[0].item():.4f}')
        
    else:
        n_ = 4
        fig, axs = plt.subplots(n_, n_, figsize=(20, 20))
        for i in range(len(pos_per_img_1)):
            i_ = i//4
            j_ = np.mod(i, 4) 
            end1 = start1 + pos_per_img_1[i]*8
            end2 = start2 + pos_per_img_2[i]*8

            axs[i_, j_].scatter(xs_1[start1:end1], ys_1[start1:end1], s=cls1[start1:end1]*100, c='g', edgecolors="none", alpha=0.6, label=tag[0])
            axs[i_, j_].scatter(xs_2[start2:end2], ys_2[start2:end2], s=cls2[start2:end2]*100, c='r', edgecolors="none", alpha=0.6, label=tag[1])
            try:
                axs[i_, j_].set_title(f'loss_kd: {loss[i].item():.4f}')
            except:
                continue

            start1 = end1
            start2 = end2

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{nstep}_img_2d.png")
    plt.close()