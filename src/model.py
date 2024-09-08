# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torchvision.models import resnet18, resnet50, resnet101, resnet152, inception_v3, resnext50_32x4d, resnext101_32x8d
import timm
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import math
#import  clip
from transformers.adapters import LoRAConfig
from transformers import AutoProcessor, CLIPModel,CLIPVisionConfig,AdapterType, PfeifferConfig,AdapterConfig,AutoTokenizer,CLIPVisionModelWithProjection,CLIPTextModelWithProjection
from einops import rearrange

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x




class LearnedPositionalEncoding(nn.Module):
    """ Positional encoding layer

    Parameters
    ----------
    dropout : float
        Dropout value.
    num_embeddings : int
        Number of embeddings to train.
    hidden_dim : int
        Embedding dimensionality
    """

    def __init__(self, dropout=0.1, num_embeddings=50, hidden_dim=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.hidden_dim)
        x = x + embeddings
        return self.dropout(x)


def AvgPoolSequence(attn_mask, feats, e=1e-12):
    """ The function will average pool the input features 'feats' in
        the second to rightmost dimension, taking into account
        the provided mask 'attn_mask'.
    Inputs:
        attn_mask (torch.Tensor): [batch_size, ...x(N), 1] Mask indicating
                                  relevant (1) and padded (0) positions.
        feats (torch.Tensor): [batch_size, ...x(N), D] Input features.
    Outputs:
        feats (torch.Tensor) [batch_size, ...x(N-1), D] Output features
    """

    length = attn_mask.sum(-1)
    # pool by word to get embeddings for a sequence of words
    mask_words = attn_mask.float()*(1/(length.float().unsqueeze(-1).expand_as(attn_mask) + e))
    feats = feats*mask_words.unsqueeze(-1).expand_as(feats)
    feats = feats.sum(dim=-2)

    return feats



class SingleTransformerEncoder(nn.Module):
    """A transformer encoder with masked average pooling at the output

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.

    """
    def __init__(self, dim, n_heads, n_layers):
        super(SingleTransformerEncoder, self).__init__()

        self.pos_encoder = LearnedPositionalEncoding(hidden_dim=dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=n_heads)

        self.tf = nn.TransformerEncoder(encoder_layer,
                                        num_layers=n_layers)

    def forward(self, feat, ignore_mask):

        if self.pos_encoder is not None:
            feat = self.pos_encoder(feat)
        # reshape input to t x bs x d
        feat = feat.permute(1, 0, 2)
        out = self.tf(feat, src_key_padding_mask=ignore_mask)
        # reshape back to bs x t x d
        out = out.permute(1, 0, 2)

        out = AvgPoolSequence(torch.logical_not(ignore_mask), out)

        return out

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
      
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
      

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
    
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn



class RecipeTransformerEncoder(nn.Module):
    """The recipe text encoder. Encapsulates encoders for all recipe components.

    Parameters
    ----------
    hidden_size : int
        Output embedding size.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.

    """
    def __init__(self, hidden_size, n_heads,
                 n_layers):
        super(RecipeTransformerEncoder, self).__init__()

        
        
        self.tfs = nn.ModuleDict()

        # independent transformer encoder for each recipe component
        for name in ['title', 'ingredients', 'instructions']:
            self.tfs[name] = get_cliptext_adapter()

        self.merger = nn.ModuleDict()
        for name in ['ingredients', 'instructions']:
            self.merger[name] = SingleTransformerEncoder(dim=hidden_size,
                                                         n_heads=n_heads,
                                                         n_layers=n_layers)



    def forward(self, input, name=None):
        '''
        Extracts features for an input using the corresponding encoder (by name)
        '''
        # check if input is a sequence or a sequence of sequences
        if len(input.size()) == 2:
            # if it is a sequence, the output of a single transformer is used
            #ignore_mask = (input == 0)
            out_dic = self.tfs[name](input)
            out =  out_dic.text_embeds
            

        else:
            # if it's a sequence of sequences, the first encoder is applied
            # to each sentence, and the second on
            
            input_rs = input.view(input.size(0)*input.size(1), input.size(2))
            out_dic = self.tfs[name](input_rs)
            out =  out_dic.text_embeds
            out = out.view(input.size(0), input.size(1), out.size(-1))

            attn_mask = input > 0
            mask_list = (attn_mask.sum(dim=-1) > 0).bool()
            out = self.merger[name](out, torch.logical_not(mask_list))
                    
        return out



def get_clipvision_lora():
    image_project = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    config = LoRAConfig(r=8, alpha=16)
    image_project.add_adapter("lora_adapter", config=config)
    image_project.merge_adapter("lora_adapter")
    image_project.reset_adapter()
    #adpt_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=4, non_linearity="relu")
    image_project.train_adapter("lora_adapter")
    
    return  image_project


def get_cliptext_lora():
    text_project = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    config = LoRAConfig(r=8, alpha=16)
    text_project.add_adapter("lora_adapter", config=config)
    text_project.merge_adapter("lora_adapter")
    text_project.reset_adapter()
    #adpt_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=4, non_linearity="relu")
    text_project.train_adapter("lora_adapter")

    return text_project


def get_clipvision_adapter():
    image_project = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    adpt_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=4, non_linearity="relu")
    image_project.add_adapter("bottleneck_adapter",config=adpt_config)
    image_project.train_adapter("bottleneck_adapter")
    
    return  image_project

def get_clipvision_sam_adapter():
    
    image_project = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    adpt_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=4, non_linearity="relu")
    image_project.add_adapter("bottleneck_adapter",config=adpt_config)
    image_project.train_adapter("bottleneck_adapter")
    
    return  image_project

def get_clipvision_freeze():
    image_project = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    for param in image_project.parameters():
        param.requires_grad = False
    return  image_project

def get_cliptext_adapter():
    text_project = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    adpt_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=4, non_linearity="relu")
    text_project.add_adapter("bottleneck_adapter",config=adpt_config)   
    text_project.train_adapter("bottleneck_adapter")

    return text_project

def get_cliptext_freeze():
    text_project = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    for param in text_project.parameters():
        param.requires_grad = False

    return text_project





class JointEmbedding(nn.Module):
    """A joint embedding of ingredients and recipes
    Parameters
    ----------
    output_size : int
        Embedding output size.
    image_model : string
        Name of image model.
    hidden_recipe : int
        Embedding size for recipe components
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    """

    def __init__(self, output_size,
                 hidden_recipe=512,
                 n_heads=4, n_layers=2):
        super(JointEmbedding, self).__init__()


        self.text_encoder = RecipeTransformerEncoder(hidden_size=hidden_recipe,
                                                     n_heads=n_heads,
                                                     n_layers=n_layers)

        
        self.image_encoder = get_clipvision_adapter()

        self.sam_img_encoder = get_clipvision_freeze()

        self.llama_text_encoder = get_cliptext_adapter()

        self.merger_recipe = nn.ModuleList()

        # linear layer to merge features from all recipe components.
        self.merger_recipe = nn.Linear(hidden_recipe*(3), output_size)
        #self.merger_pic = nn.Linear(hidden_recipe*(2), output_size) 


        # projection layers for self supervised recipe loss
        self.projector_recipes = nn.ModuleDict()
        names = ['title', 'ingredients', 'instructions']
        for name in names:
            self.projector_recipes[name] = nn.ModuleDict()
            for name2 in names:
                if name2 != name:
                    self.projector_recipes[name][name2] = nn.Linear(hidden_recipe, hidden_recipe)

    def forward(self, img,sam_img_whole, title, ingrs, instrs,llama_desc,
                freeze_backbone=True):

        text_features = []
        projected_text_features = {'title': {},
                                   'ingredients': {},
                                   'instructions': {},
                                   'raw': {}}

        elems = {'title': title, 'ingredients': ingrs, 'instructions': instrs}

        names = list(elems.keys())

        for name in names:
            # for each recipe component, extracts features and projects them to all other spaces
            input_source = elems[name]
           
            text_feature = self.text_encoder(input_source, name)
            text_features.append(text_feature)
            projected_text_features['raw'][name] = text_feature
            for name2 in names:
                if name2 != name:
                    projected_text_features[name][name2] = self.projector_recipes[name][name2](text_feature)
        
        
        if img is not None:

            out_img = self.image_encoder(img)
            img_feat = out_img.image_embeds


            b,n_img, c,h,w = sam_img_whole.shape
            sam_img_whole_x = rearrange(sam_img_whole,'b n_img c h w -> (b n_img) c h w')
            out_sam_img = self.sam_img_encoder(sam_img_whole_x)
            sam_img_whole_feat = out_sam_img.image_embeds

            sam_img_whole_feat = rearrange(sam_img_whole_feat,'(b n_img) dim -> b n_img dim',n_img = n_img)
            sam_img_whole_feat = sam_img_whole_feat.mean(dim=1)

            #sam_img_whole_feat = torch.max(sam_img_whole_feat, dim=1)

            llama_feat = self.llama_text_encoder(llama_desc)
            llama_feat = llama_feat.text_embeds
        else:
            img_feat = None
            sam_img_whole_feat = None
            llama_feat = None

        

        recipe_feat = self.merger_recipe(torch.cat(text_features, dim=1))
        recipe_feat = nn.Tanh()(recipe_feat)
      
       
        return img_feat,sam_img_whole_feat,recipe_feat, projected_text_features,llama_feat


def get_model(args):
    model = JointEmbedding(output_size=args.output_size,
                           hidden_recipe=args.hidden_recipe,
                           image_model=args.backbone,
                           n_heads=args.tf_n_heads,
                           n_layers=args.tf_n_layers,
                           )
    return model