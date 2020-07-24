import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from default_config import config
import argparse
from detect_face import detect_face

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    return args
    

def build_target():
    while True:
        try:
            print('')
            # 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'
            hair = input('Input your desired hair color, 0 for Black_Hair, 1 for Blond_Hair, 2 for Brown_Hair: ')
            hair = int(hair)
            assert hair in [0, 1, 2]
            male = input('Input the gender of the generated image, 0 for Female, 1 for Male: ')
            male = int(male)
            assert male in [0, 1]
            age = input('Input whether to generate aged image, 0 for No, 1 for Yes: ')
            age = int(age)
            assert age in [0, 1]
            target = [0, 0, 0]
            target[hair] = 1
            target.append(male)
            target.append(1 - age)
            return target
        except Exception:
            pass

if __name__ == '__main__':
    args = parse_args()

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
    
    
    face = detect_face(args.image)
    if face is not None:
        face = Image.fromarray(face)
        face = transform(face)
        face = torch.unsqueeze(face, 0)
        # 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'
        face = face.to(solver.device)
        target_cls = torch.tensor(build_target()).float()
        target_cls = torch.unsqueeze(target_cls, 0)
        target_cls = target_cls.to(solver.device)
        gen_face = solver.G(face, target_cls)
        gen_face = solver.denorm(gen_face.detach().cpu())
        save_image(gen_face, 'images/gen.png')
        print('Please check images/gen.png.')
    else:
        print("No face detected!")

    
    

    
