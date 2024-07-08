# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 18:10:50 2022

@author: kossi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


## For range 
hid_dim = 2
epsilon = 2
gamma = 1
pi = 3.14#np.pi

rang = (gamma + epsilon)/hid_dim

## Switch
Type =False # True if interested in Relation otherwise False for Entities
angle = True # vector
not_a_tree = False # True
TransE =  False#True. False for ETrans
all_figs_in_1 = True # True False
step=2000

model_foldername = '_tree0_0/'
data_foldername = 'tree0'
#########
model_name = 'TransE' if TransE else 'ETrans' 
folder_results = "models/" + model_name + "_"+data_foldername+"_0/"# "_toy_0/"
folder_data = "data/" + data_foldername + "/" #"data/toy/"

os.makedirs(folder_results+"Figs", exist_ok=True)

## Embeddings relations and entities
Ent= np.load(folder_results+'entity_embedding.npy')
Rel= np.load(folder_results+'relation_embedding.npy')
## Entyties and relations Data
Ent_txt= pd.read_table(folder_data+'entities.dict', header=None).iloc[:, 1]
Rel_txt= pd.read_table(folder_data+'relations.dict', header=None).iloc[:, 1]

## KG 
labels = Rel_txt.iloc[:] if Type else Ent_txt.iloc[:] 
fig_count =0

fig = plt.figure(figsize=(15,15))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fig_size = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = fig_size
#plt.title('step'+str(step))
done = []
Tail = set()
if all_figs_in_1:
    plt.figure(0)
with open(folder_data+"KG.txt") as kg:
    for triple in kg.readlines():
        h, rel, t = triple.split()
        Tail.add(t) # save all the tails
        
with open(folder_data+"entities.txt") as kg:
    for triple in kg.readlines():
        _ , h = triple.split()
        if h in done:
            continue
        else:
            done.append(h)
            pass
        
        try:
            with open(folder_data+"KG"+h+".txt") as kg1:
                pass
        except:
            continue
        
        fig_count = 0 if all_figs_in_1 else fig_count + 1
        title = model_name+"at step" + str(step) if all_figs_in_1 else model_name+" with focus on "+h+" at step" + str(step)
        if not all_figs_in_1:
            plt.figure(fig_count)
            plt.title(title)
            
        with open(folder_data+"KG"+h+".txt") as kg1:
            for triple1 in kg1.readlines():
                _, rel, t = triple1.split()
                for i, head in enumerate(Ent_txt):
                    if head == h:
                        for j, tail in enumerate(Ent_txt):
                            if tail == t:
                                head_x, head_y = Ent[i] if TransE else Ent[i, :hid_dim]
                                forceh_x, forceh_y = (0, 0) if TransE else Ent[i, hid_dim:]
                                
                                tail_x, tail_y = Ent[j] if TransE else Ent[j, :hid_dim]
                                forcet_x, forcet_y = (0, 0) if TransE else Ent[j, hid_dim:]
                                
                                plt.plot(head_x, head_y, label = h, linewidth=2, marker='s' )
                                # if TransE:
                                #    plt.plot([head_x, trasn_h_x], [head_y, trasn_h_y], '-->', linewidth=1)
                                plt.plot(tail_x, tail_y, label = t, linewidth=2, marker='o' )
                                
                                plt.text(head_x, head_y, h, fontsize= 16)
                                plt.text(tail_x, tail_y, t, fontsize= 16)
                                #plt.plot(head_x + forceh_x, head_y + forceh_y, '-->',label = h, linewidth=6 )
                                
                                for k, rels in enumerate(Rel_txt):
                                    if rels== rel:
                                        #trasn_h_x = head_x + forcet_x
                                        #trasn_h_y = head_y + forcet_y
                                        
                                        rel_x, rel_y = Rel[k] if TransE else Rel[k, :hid_dim]
                                        is_h_a_tail = (h in Tail) 
                                        if is_h_a_tail and all_figs_in_1 and not_a_tree:
                                            continue
                                        rhead_x =  head_x + (-1)**(is_h_a_tail*not_a_tree)*( rel_x + forcet_x - forceh_x)
                                        rhead_y = head_y + (-1)**(is_h_a_tail*not_a_tree)*(rel_y + forcet_y - forceh_y)
                                        #rhead_x, rhead_y  = Rel[k] + Ent[i] if TransE else Rel[k, :hid_dim] + Ent[j, hid_dim:] - Ent[i, hid_dim:]
                                        plt.plot([head_x, rhead_x], [head_y, rhead_y],'-->', label =rel, linewidth=1 )
                #plt.show()

        if all_figs_in_1:
            continue
        else:
            fig.savefig(folder_results+'Figs/'+title+'.png', dpi=300, bbox_inches='tight')   
    if all_figs_in_1:
        plt.show()
        fig.savefig(folder_results+'Figs/'+title+'.png', dpi=300, bbox_inches='tight')                 
################################################################################

start=0
## TITLE
title = 'Relations R from ' if Type else 'Entities E from '
title += str(start) + " to "+str(9+start)

title = "Vectors of " + title #if angle else "Azimuts of " + title

## Choose between relation and entities embeddings
data = Rel if Type else Ent 


#dt_theta = data[:, :hid_dim].T/(rang/pi)
#dt_azimut = data[:, hid_dim:].T
#dt = dt_azimut

## Choose between theta or azimuts plots
# dt = data[:, :hid_dim].T/(rang/pi) if angle else data[:, hid_dim:].T
dt = data[:, :hid_dim] 

## Plots entities
# =============================================================================
# for i, pos in enumerate(Ent):
#     plt.plot(pos[0], pos[1], label =Ent_txt.iloc[i], linewidth=6, marker='*' )
#     plt.text(pos[0], pos[1], labels[i])
# for i, pos in enumerate(Rel):
#     plt.plot([0, pos[0]], [0, pos[1]], label =Rel_txt.iloc[i], linewidth=2 )
#     plt.text(pos[0], pos[1], labels[i])
# =============================================================================
#plt.legend()


name = 'R' if Type else 'E'
names = [name +str(i) for i in range(dt.shape[1])]
df = pd.DataFrame(data= dt, index=labels)

#### Read KG 
#C:\Users\kossi\Documents\KGE Codes\KGEmbeddingRotatE-master\data\FB15k
#dd=pd.read_table('relations.dict')
#dd.head(10)