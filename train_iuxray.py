import random
from data import ImageDetectionsField, TextField, RawField, ImageDetectionsFieldIUXRay, TextFieldIUXRay
from data import COCO, DataLoader, IUXray
import evaluation
from evaluation import PTBTokenizer, Cider, Rouge, Bleu
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, EnsembleMultiLevelTransformerDecoder, ScaledDotProductAttention, OriginalTransformer
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

import torch
from torch import nn
import torch.nn.functional as F

def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

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

def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}

    lst_gts, lst_res = [], []
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
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
    return scores

def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, metric_scst, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 60
    beam_size = 3

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()
            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = [clean_report_iu_xray(cap_gt[0]) for cap_gt in caps_gt]
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))

            caps_gen = {i: [v] for i, v in enumerate(caps_gen)}
            caps_gt = {i: [v] for i, v in enumerate(caps_gt)}

            # reward = np.array(metric_scst.compute_score(caps_gt, caps_gen)[1]).astype(np.float32)[-1, :]
            reward = metric_scst.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline

def build_optimizer(model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, 'Adam')(
        [{'params': model.visual_extractor.parameters(), 'lr': 1},
         {'params': ed_params, 'lr': 1}],
        weight_decay=5e-5,
        amsgrad=True
    )
    return optimizer

def build_lr_scheduler(optimizer, lambda_lr):
    lr_scheduler = LambdaLR(optimizer, lambda_lr)
    return lr_scheduler

if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Experiments on Transformer')
    parser.add_argument('--exp_name', type=str, default='iu_xray_exp2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str, default='./iu_xray/images/')
    parser.add_argument('--annotation_folder', type=str, default='./iu_xray/')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--rl_at', type=int, default=20)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()
    print(args)

    print('IU-Xray Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field_train_val = ImageDetectionsFieldIUXRay(detections_path=args.features_path, load_in_tmp=False, data_set='train')
    image_field_test = ImageDetectionsFieldIUXRay(detections_path=args.features_path, load_in_tmp=False, data_set='test')

    # Pipeline for text
    text_field = TextFieldIUXRay(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=False, nopoints=False)

    # Create the dataset
    dataset = IUXray(image_field_train_val, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, _ = dataset.splits

    dataset = IUXray(image_field_test, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits

    if not os.path.isfile('vocab/vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=2)
        pickle.dump(text_field.vocab, open('vocab/vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab/vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttention, d_in=2048)
    decoder = MeshedDecoder(len(text_field.vocab), 180, 3, text_field.vocab.stoi['<pad>'])
    model = OriginalTransformer(text_field.vocab.stoi['<bos>'], encoder, decoder, pretrained_backbone=args.backbone).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field_train_val, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    metric_scst = Cider(PTBTokenizer.tokenize(ref_caps_train))
    # metric_scst = Bleu()
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field_train_val, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field_test, 'text': RawField()})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    # Initial conditions
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    optim = build_optimizer(model)
    scheduler = build_lr_scheduler(optim, lambda_lr)

    use_rl = False
    best_monitor_score = .0
    patience = 0
    start_epoch = 0

    if not os.path.exists('./checkpoints/' + args.exp_name):
        os.mkdir('./checkpoints/' + args.exp_name)

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = './checkpoints/' + args.exp_name + '/%s_last.pth' % args.exp_name
        else:
            fname = './checkpoints/' + args.exp_name + '/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_monitor_score = data['best_monitor_score']
            patience = data['patience']
            use_rl = True
            print('Resuming from epoch %d, validation loss %f, and best monitor score %f' % (
                data['epoch'], data['val_loss'], data['best_monitor_score']))

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, metric_scst, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_monitor_score = scores['CIDEr'] # BLEU@4 as monitor score.
        writer.add_scalar('data/val_cider', val_monitor_score, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_monitor_score >= best_monitor_score:
            best_monitor_score = val_monitor_score
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == args.patience:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('./checkpoints/' + args.exp_name + '/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best monitor score %f' % (
                data['epoch'], data['val_loss'], data['best_monitor_score']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_monitor_score': val_monitor_score,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_monitor_score': best_monitor_score,
            'use_rl': use_rl,
        }, './checkpoints/' + args.exp_name + '/%s_last.pth' % args.exp_name)

        if best:
            copyfile('./checkpoints/' + args.exp_name + '/%s_last.pth' % args.exp_name, './checkpoints/' + args.exp_name + '/%s_best.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break