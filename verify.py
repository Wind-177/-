import os
import numpy as np
import nibabel as nib
import torch
import glob
# In[1]
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

mov_path = r'D:/交接文件/数据/配准数据/人工仿射配准后' #mr label 配准后的文件 mov是mr, fix是us
fix_path = r'D:/交接文件/数据/配准数据/us_label'

move_path_id = os.listdir(mov_path)
move_path_id = [file for file in move_path_id if file.startswith('Case')]
list_len = len(move_path_id)
a = 0
out = np.zeros(list_len)
np.set_printoptions(precision=3)
for k in range(0, list_len):
    fix_name = os.path.join(fix_path, move_path_id[k])
    mov_name = os.path.join(mov_path, move_path_id[k])
    fix_img = nib.load(fix_name)
    spacing = fix_img.header.get_zooms()
    fix_img_data = fix_img.get_fdata()
    mov_img = nib.load(mov_name)
    mov_img_data = mov_img.get_fdata()
    fix_img_data = torch.from_numpy(fix_img_data).cuda()
    mov_img_data = torch.from_numpy(mov_img_data).cuda()
    unique = torch.unique(fix_img_data).cuda()
    positions = torch.zeros((len(unique) - 1, 3))
    positions2 = torch.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = torch.where(fix_img_data == unique[i])
        positions[i - 1, 0] = label[0].float().mean()
        positions[i - 1, 1] = label[1].float().mean()
        positions[i - 1, 2] = label[2].float().mean() #计算US坐标的平均值
    for i in range(1, len(unique)):
        label2 = torch.where(mov_img_data == unique[i])
        positions2[i - 1, 0] = label2[0].float().mean()
        positions2[i - 1, 1] = label2[1].float().mean()
        positions2[i - 1, 2] = label2[2].float().mean()#计算MR坐标的平均值

    positions = positions.numpy()
    positions2 = positions2.numpy()
    tre = np.linalg.norm((positions - positions2) * spacing, axis=1)
    print(tre.mean(), move_path_id[k])
    a += tre.mean()
    out[k] = tre.mean()
print(out.mean(), np.std(out))
