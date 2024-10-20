import os
import multiprocessing
from dataset import get_loader
from models import get_model
import torch.backends.cudnn as cudnn
from config import get_args
from tqdm import tqdm
import torch
import numpy as np
import pickle
from utils.utils import load_checkpoint, count_parameters
from transformers import AutoProcessor, CLIPModel,AdapterType, PfeifferConfig,AdapterConfig,AutoTokenizer,CLIPVisionModelWithProjection,CLIPTextModelWithProjection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'


def test(args):
    torch.set_num_threads(8)
    if device != 'cpu':
        cudnn.benchmark = True
    checkpoints_dir = os.path.join(args.save_dir, args.model_name)
    # make sure these arguments are kept from commandline and not from loaded args
    vars_to_replace = ['batch_size', 'eval_split', 'imsize', 'root', 'save_dir']
    store_dict = {}
    for var in vars_to_replace:
        store_dict[var] = getattr(args, var)
    args, model_dict, _ = load_checkpoint(checkpoints_dir, 'best', map_loc,
                                          store_dict)
    for var in vars_to_replace:
        setattr(args, var, store_dict[var])

    loader, dataset = get_loader(args.root, args.batch_size, args.resize,
                                 args.imsize,
                                 augment=False,
                                 split=args.eval_split, mode='test',
                                 drop_last=False)
    print("Extracting features for %d samples from the %s set..."%(len(dataset),
                                                                   args.eval_split))
    model = get_model(args)

    print("recipe encoder", count_parameters(model.text_encoder))
    print("image encoder", count_parameters(model.image_encoder))


    model.load_state_dict(model_dict, strict=False)

    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()
    
    """ text_project = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    text_project.to(device) """

    

    total_step = len(loader)
    loader = iter(loader)
    print("Loaded model from %s ..."%(checkpoints_dir))
    all_f1, all_f2 = None, None
    all_sam_feat,all_llama_feat = None,None
    allids = []

    for _ in tqdm(range(total_step)):

        img,sam_img_whole, title, ingrs, instrs,llama_desc, ids = next(loader)

        img = img.to(device)
        sam_img_whole = sam_img_whole.to(device) 
        title = title.to(device)
        ingrs = ingrs.to(device)
        instrs = instrs.to(device)
        llama_desc =llama_desc.to(device)
        with torch.no_grad():
            out = model(img,sam_img_whole, title, ingrs, instrs,llama_desc)
            f1, sam_feat,f2, _,llama_feat = out
        allids.extend(ids)
        if all_f1 is not None:
            all_f1 = np.vstack((all_f1, f1.cpu().detach().numpy()))
            all_f2 = np.vstack((all_f2, f2.cpu().detach().numpy()))

            all_sam_feat = np.vstack((all_sam_feat, sam_feat.cpu().detach().numpy()))
            all_llama_feat = np.vstack((all_llama_feat, llama_feat.cpu().detach().numpy()))
        else:
            all_f1 = f1.cpu().detach().numpy()
            all_f2 = f2.cpu().detach().numpy()
            
            all_sam_feat = sam_feat.cpu().detach().numpy()
            all_llama_feat = llama_feat.cpu().detach().numpy()



    print("Done.")

    file_to_save = os.path.join(checkpoints_dir,
                                'feats_' + args.eval_split+'.pkl')
    
    with open(file_to_save, 'wb') as f:
        pickle.dump(all_f1, f)
        pickle.dump(all_f2, f)
        pickle.dump(all_sam_feat, f)
        pickle.dump(all_llama_feat, f)
        pickle.dump(allids, f)
    print("Saved features to disk.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    args = get_args()
    test(args)
