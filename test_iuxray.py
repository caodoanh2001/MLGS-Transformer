import random
from data import ImageDetectionsField, TextField, RawField, ImageDetectionsFieldIUXRay, ImageDetectionsFieldIUXRayAndSegmentation
from data import COCO, DataLoader, IUXray
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention, OriginalTransformer, TransformerWithMasks, MemoryAugmentedEncoderWithMasks
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
import json
import re
def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def predict_captions(model, dataloader, text_field, name_json=None):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    lst_gts, lst_res = [], []
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            caps_gt = [clean_report_iu_xray(cap_gt[0]) for cap_gt in caps_gt]
            with torch.no_grad():
                out, _ = model.beam_search(images, 60, text_field.vocab.stoi['<eos>'], 3, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            lst_caps_gen_batch_i = []
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                lst_caps_gen_batch_i.append(gen_i)
            
            lst_res.extend(lst_caps_gen_batch_i)
            lst_gts.extend(caps_gt)
            pbar.update()

    scores, _ = evaluation.compute_scores({i: [gt] for i, gt in enumerate(lst_gts)}, 
                                            {i: [re] for i, re in enumerate(lst_res)})

    if name_json:
        f = open('./' + name_json + '.json', 'w')
    else:
        f = open('./results.json', 'w')
    json.dump({i: {'re': re} for i, (re, gt) in enumerate(zip(lst_res, lst_gts))}, f, indent=4)
    f.close()
    return scores

def predict_captions_with_masks(model, dataloader, text_field, name_json=None):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    lst_gts, lst_res = [], []
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images, masks = images['images'].to(device), images['masks'].to(device)
            caps_gt = [clean_report_iu_xray(cap_gt[0]) for cap_gt in caps_gt]
            with torch.no_grad():
                out, _ = model.beam_search(images, masks, 60, text_field.vocab.stoi['<eos>'], 3, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            lst_caps_gen_batch_i = []
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                lst_caps_gen_batch_i.append(gen_i)
            
            lst_res.extend(lst_caps_gen_batch_i)
            lst_gts.extend(caps_gt)
            pbar.update()

    scores, _ = evaluation.compute_scores({i: [gt] for i, gt in enumerate(lst_gts)}, 
                                            {i: [re] for i, re in enumerate(lst_res)})

    if name_json:
        f = open('./' + name_json + '.json', 'w')
    else:
        f = open('./results.json', 'w')
    json.dump({i: {'re': re, 'gt': gt} for i, (re, gt) in enumerate(zip(lst_res, lst_gts))}, f, indent=4)
    f.close()
    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='MLGS Transformer')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--features_path', type=str, default='./iu_xray/images/')
    parser.add_argument('--annotation_folder', type=str, default='./iu_xray/')
    parser.add_argument('--exp_name', type=str, default='iuxray_msaf_lsg')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--grid_size', type=int, default=7)
    parser.add_argument('--with_masks', type=bool, default=False)
    args = parser.parse_args()

    print('MLGS Transformer Evaluation')

    # Pipeline for image regions
    image_field_test = ImageDetectionsFieldIUXRay(detections_path=args.features_path, load_in_tmp=False, data_set='test')
    # image_field_test = ImageDetectionsFieldIUXRayAndSegmentation(detections_path=args.features_path, load_in_tmp=False, data_set='test', seg_path='./lungVAE/segment_lung_masks/')

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = IUXray(image_field_test, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab/vocab_' + args.exp_name + '.pkl', 'rb'))

    # Model and dataloaders
    # encoder = MemoryAugmentedEncoderWithMasks(3, 0, attention_module=ScaledDotProductAttention, d_in=256)
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttention, d_in=256)
    decoder = MeshedDecoder(len(text_field.vocab), 180, 3, text_field.vocab.stoi['<pad>'])
    

    # model = TransformerWithMasks(text_field.vocab.stoi['<bos>'], encoder, decoder, pretrained_backbone=args.backbone, grid_size=args.grid_size).to(device)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, pretrained_backbone=args.backbone).to(device)

    data = torch.load('./checkpoints/' + args.exp_name + '/' + args.exp_name + '_best.pth')
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field_test, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    # scores = predict_captions_with_masks(model, dict_dataloader_test, text_field, args.exp_name)
    scores = predict_captions(model, dict_dataloader_test, text_field, args.exp_name)
    
    print(scores)