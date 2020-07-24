import os
import numpy as np
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch
#import matplotlib.pyplot as plt
print('1-5')
from torchvision import transforms as T
print('1-6')
from torchvision.utils import save_image
from PIL import Image
from default_config import config
print('1-7')
from detect_face import detect_face
print('1-8')

print('success')

def build_target():
    while True:
        try:
            print('')
            # 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'
            hair = input('Choose your desired hair color: 0 for Black_Hair, 1 for Blond_Hair, 2 for Brown_Hair ')
            hair = int(hair)
            assert hair in [0, 1, 2]
            male = input('Choose the gender you desired: 0 for Female, 1 for Male')
            male = int(male)
            assert male in [0, 1]
            age = input('Choose whether to generate aged, 0 for No, 1 for Yes: ')
            age = int(age)
            assert age in [0, 1]
            target = [0, 0, 0]
            target[hair] = 1
            target.append(male)
            target.append(1 - age)
            return target
        except Exception:
            pass
        
def showimage(img):
    npimg = img.numpy().squeeze()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def generate_face(face):
    if face is not None:
        new_face = Image.fromarray(face)
        new_face = transform(new_face)
        new_face = torch.unsqueeze(new_face, 0) #(1,3,256,256)

        # 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'
        new_face = new_face.to(solver.device)
        target_cls = torch.tensor(build_target()).float()
        target_cls = torch.unsqueeze(target_cls, 0)
        target_cls = target_cls.to(solver.device)
        gen_face = solver.G(new_face, target_cls)
        gen_face = solver.denorm(gen_face.detach().cpu())
        showimage(gen_face)

    else:
        print("No face detected!")



# set config
config.mode = 'test'
config.dataset = 'CelebA'
config.image_size = 256
config.c_dim = 5
config.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
config.model_save_dir = 'stargan_celeba_256/models'
config.result_dir = 'stargan_celeba_256/results'


# solver
solver = Solver(None, None, config)
solver.restore_model(solver.test_iters)
transform = []
transform.append(T.Resize(config.image_size))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)


print('3')

face = detect_face('./images/5.jpg')
#plt.imshow(face)
#plt.show()


print('4')

generate_face(face)