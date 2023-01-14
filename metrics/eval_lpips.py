from lpips import calculate_lpips_given_images
import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pdb
st = pdb.set_trace

def get_image_tensors(folder_path, transform):
    names = os.listdir(folder_path)
    images = []
    for name in names:
        image = transform(
                Image.open(os.path.join(folder_path, name)).convert('RGB'))
        images.append(image)
    return torch.stack(images)

if __name__ == '__main__':
    batch = 16
    transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
    images = get_image_tensors('../outputs/cat-id1', transform)
    images = images[:200]
    n_group = images.shape[0] // batch
    group_of_images = images[:n_group*batch].reshape(n_group,batch,*images.shape[1:])
    lpips_value = calculate_lpips_given_images(group_of_images)
    print(f'lpips score with batch {batch}: {lpips_value}')

