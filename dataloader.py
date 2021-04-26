import random
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from utils import get_labels


class RotationPartLoader(Dataset):
    def __init__(self, root_dir, num_parts, num_angles):
        super(RotationPartLoader, self).__init__()
        self.num_parts = num_parts
        self.root_dir = root_dir
        file_ind = open(self.root_dir)
        self.data = [line.rstrip() for line in file_ind.readlines()]
        file_ind.close()
        self.color_transform = torchvision.transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.4)
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.orig_size = 300
        self.crop_size = 224
        self.parts = [x + 1 for x in range(self.num_parts)]
        self.sq = int(self.num_parts ** (0.5))
        self.alt = self.orig_size // self.sq
        self.crop_parts = [(i * self.alt, j * self.alt, (i + 1) * self.alt, (j + 1) * self.alt) for i in range(self.sq) for j in range(self.sq)]
        self.res_crop_size = self.crop_size // self.sq
        self.num_angles = num_angles
        self.angle_ratio = 360 // num_angles
        self.angles = [i * self.angle_ratio for i in range(self.num_angles)]
        self.res_labels = get_labels(self.num_parts, self.num_angles)
        

        
    def get_image(self, path):
        image = Image.open(path).convert('RGB')
        return image
    
    def __len__(self):
        return len(self.data)
       
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        original = self.get_image(self.data[index])
        image = torchvision.transforms.Resize((self.orig_size, self.orig_size))(original)
        part = random.choice(self.parts)
        image_part = image.crop(self.crop_parts[part - 1])
        image_part = torchvision.transforms.RandomCrop((self.res_crop_size, self.res_crop_size))(image_part)
        rotation = torchvision.transforms.Resize((self.res_crop_size, self.res_crop_size))(image_part)
        rotation = self.color_transform(rotation)
        angle = random.choice(self.angles)
        angle_index = self.angles.index(angle)
        res_index = self.res_labels[part - 1][angle_index]
        rotation = torchvision.transforms.functional.rotate(rotation, angle)
        rotation = torchvision.transforms.functional.to_tensor(rotation)
        rotation = self.normalize(rotation)
        return {'rotation': rotation, 'res_index': res_index}