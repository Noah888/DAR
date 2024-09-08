# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Dataset and dataloader functions
"""

import os
import json
import random
random.seed(1234)
from random import choice
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.utils import get_token_ids, list2Tensors,listtensor2Tensors
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
from transformers import AutoTokenizer,AutoProcessor
import lmdb
import  io
import  glob

class Recipe1M(Dataset):
    """Dataset class for Recipe1M

    Parameters
    ----------
    root : string
        Path to Recipe1M dataset.
    transform : (callable, optional)
        A function/transform that takes in a PIL image and returns a transformed version.
    sam_transform : (callable, optional)
        A function/transform that takes in the segment PIL images and returns a transformed version.
    split : string
        Dataset split (train, val, or test).
    max_ingrs : int
        Maximum number of ingredients to use.
    max_instrs : int
        Maximum number of instructions to use.
    max_length_ingrs : int
        Maximum length of ingredient sentences.
    max_length_instrs : int
        Maximum length of instruction sentences.
    text_only_data : bool
        Whether to load paired or text-only samples.
    """

    def __init__(self, root, transform=None,sam_transform=None, split='train',
                 max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15,
                 text_only_data=False):

        """ self.env = lmdb.open(os.path.join(root,'traindata',split+"_lmdb"),
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False) """
        
        # suffix to load text only samples or paired samples
        suf = '_noimages' if text_only_data else ''
        self.data = pickle.load(open(os.path.join(root, 'traindata', split + suf + '.pkl'),
                                     'rb'))
        self.root = root
        self.ids = list(self.data.keys())
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

        self.split = split
        self.transform = transform
        self.sam_tranform = sam_transform

        self.max_ingrs = max_ingrs
        self.max_instrs = max_instrs
        self.max_length_ingrs = max_length_ingrs
        self.max_length_instrs = max_length_instrs

        self.text_only_data = text_only_data

    def __getitem__(self, idx):

        entry = self.data[self.ids[idx]]

        if not self.text_only_data:
            # loading images
            if self.split == 'train':
                # if training, pick an image randomly
                img_name = choice(entry['images'])

            else:
                # if test or val we pick the first image
                img_name = entry['images'][0]
            
            """ with self.env.begin(write=False) as txn:
                buf = txn.get(img_name.encode('ascii'))
            img = Image.open(io.BytesIO(buf)) """
            img_name_file = '/'.join(img_name[:4])+'/'+ img_name
            
            img = Image.open(os.path.join(self.root, self.split, img_name_file))
            if self.transform is not None:
                img = self.transform(img)
            
            img_single_name,_ = os.path.splitext(img_name)
            sam_img_file = os.path.join(self.root,"segment", self.split, '/'.join(img_name[:4]) , img_single_name)
            sam_file_list = glob.glob(os.path.join(sam_img_file, "*.*"))
            sam_img_list = []
            for filename in sam_file_list:
                image_sam = Image.open(filename)
                if self.transform is not None:
                    image_sam =self.sam_tranform(image_sam)
                    sam_img_list.append(image_sam)
            sam_img_whole = torch.stack(sam_img_list, dim=0)
            llama_description = entry['llama_13b_generation']
            llama_description = self.tokenizer(llama_description,max_length=77, padding=True,truncation=True, return_tensors="pt")['input_ids'].squeeze(0)  
        else:
            img = None
            sam_img_whole = None
            llama_description =None

        title = entry['title']
        ingrs = entry['ingredients']
        instrs = entry['instructions']
      
        # turn text into indexes  
        title  = self.tokenizer(title, padding=True, return_tensors="pt")['input_ids'].squeeze(0)            
        instrs = listtensor2Tensors([self.tokenizer(instr[:self.max_length_instrs], padding=True, return_tensors="pt")['input_ids'][:,:self.max_length_instrs+2]  for instr in instrs[:self.max_instrs]])
        ingrs  = listtensor2Tensors([self.tokenizer(ingr[:self.max_length_instrs], padding=True, return_tensors="pt")['input_ids'][:,:self.max_length_ingrs+2]   for ingr in ingrs[:self.max_ingrs]])
              
        return img, sam_img_whole,title, ingrs, instrs,llama_description,self.ids[idx]

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

   


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """
    
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long()
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_fn(data):
    """ collate to consume and batchify recipe data
    """

    # Sort a data list by caption length (descending order).
    image, sam_img_whole,titles, ingrs, instrs,llama_description, ids = zip(*data)

    if image[0] is not None:
        # Merge images (from tuple of 3D tensor to 4D tensor).
        image = torch.stack(image, 0)
        sam_img_whole = torch.stack(sam_img_whole, 0)
        llama_targets  = pad_input(llama_description)
    else:
        image = None
        sam_img_whole = None
        llama_targets  = None
    
    title_targets = pad_input(titles)
    ingredient_targets = pad_input(ingrs)
    instruction_targets = pad_input(instrs)
    
    return image,sam_img_whole, title_targets, ingredient_targets, instruction_targets,llama_targets, ids


def get_loader(root, batch_size, resize, im_size, augment=True,
               split='train', mode='train',
               drop_last=True,
               text_only_data=False):
    """Function to get dataset and dataloader for a data split

    Parameters
    ----------
    root : string
        Path to Recipe1M dataset.
    batch_size : int
        Batch size.
    resize : int
        Image size for resizing (keeps aspect ratio)
    im_size : int
        Image size for cropping.
    augment : bool
        Description of parameter `augment`.
    split : string
        Dataset split (train, val, or test)
    mode : string
        Loading mode (impacts augmentations & random sampling)
    drop_last : bool
        Whether to drop the last batch of data.
    text_only_data : type
        Whether to load text-only or paired samples.

    Returns
    -------
    loader : a pytorch DataLoader
    ds : a pytorch Dataset

    """

    transforms_list = [transforms.Resize((resize))]

    if mode == 'train' and augment:
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomCrop(im_size))

    else:
        transforms_list.append(transforms.CenterCrop(im_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                               (0.26862954, 0.26130258, 0.27577711)))

    transforms_ = transforms.Compose(transforms_list)

    sam_transforms_list = [transforms.Resize((224,224))]

    if mode == 'train' and augment:
        sam_transforms_list.append(transforms.RandomHorizontalFlip())
    sam_transforms_list.append(transforms.ToTensor())
    sam_transforms_list.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                    (0.26862954, 0.26130258, 0.27577711)))
    sam_transforms_ = transforms.Compose(sam_transforms_list)

    ds = Recipe1M(root, transform=transforms_,sam_transform=sam_transforms_, split=split,
                  text_only_data=text_only_data)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=12,
                        collate_fn=collate_fn, drop_last=drop_last,pin_memory=True)

    return loader, ds
