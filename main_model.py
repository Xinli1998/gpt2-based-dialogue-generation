from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup,GenerationConfig,top_k_top_p_filtering
from custom_dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
import re
import torch
import os, sys
import numpy as np
import argparse
import copy
import math
import random
from torch.cuda.amp import autocast, GradScaler
import pdb
import bisect
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from gptmodel import *
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import re
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import wordnet
# from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import math
# import ipdb
class Manager():
    def __init__(self, args):
        self.args = args
        
        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            # 'eos_token': self.args.eos_token,
            'bos_token': self.args.bos_token,
            'additional_special_tokens': [self.args.sp1_token, self.args.sp2_token]
        }
        # self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        # pdb.set_trace()
        if self.args.vae_type == 'gpt2model':
            gpt2_vae_prior = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
            gpt2_vae_prior.resize_token_embeddings(self.args.vocab_size)
            gpt2_vae_post = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
            gpt2_vae_post.resize_token_embeddings(self.args.vocab_size)
        else:
            gpt2_vae_prior = None
            gpt2_vae_post = None          
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        if self.args.model_fusion == 'encoder_h_concat' or self.args.model_fusion == 'com_latent_seg':
            add_cross_attention=True
        else:
            add_cross_attention=False
            # output_attentions=True
        #define new model ,add_cross_attention=True
        if self.args.use_old:
            print('load model from previous GPT2.ckpt')
            gpt_model = GPT2LMHeadModel.from_pretrained(self.args.model_type,output_attentions = True,output_hidden_states=True ).to(self.args.device)
            gpt_model.resize_token_embeddings(self.args.vocab_size)
            #load old model
            ckpt = torch.load(f'{args.ckpt_dir}/best_ckpt_epoch=14_valid_loss=2.6137.ckpt', map_location=self.args.device)
            gpt_model.load_state_dict(ckpt['model_state_dict'])
        else:
            #define new model ,add_cross_attention=True
            print('new model from with cross_attention')
            gpt_model = GPT2LMHeadModel.from_pretrained(self.args.model_type,add_cross_attention=add_cross_attention,output_attentions = True,output_hidden_states=True ).to(self.args.device)
            gpt_model.resize_token_embeddings(self.args.vocab_size)  
                
        config = GPT2Config(
            output_attentions = True,
            vocab_size = args.vocab_size,
            # eos_id = args.eos_id,
            sp1_id = self.args.sp1_id,
            sp2_id = self.args.sp2_id)
        # if self.args.model_fusion == 'encoder_h_concat' or self.args.model_fusion == 'com_latent_seg':
        config.add_cross_attention=add_cross_attention
        config.output_attentions=True

        self.model = model_vae(config,gpt_model,with_latent = self.args.with_latent,with_hier = self.args.with_hier,hier_type = self.args.hier_type,model_fusion= self.args.model_fusion,vae_type = self.args.vae_type,gpt2_vae_prior = gpt2_vae_prior,gpt2_vae_post = gpt2_vae_post ).to(self.args.device)
        print(f"This model type is {self.args.vae_type} with_latent:{self.args.with_latent} with_hier:{self.args.with_hier}")
        # self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
        ppd = PadCollate(eos_id=self.tokenizer.eos_token_id)
        test_set = CustomDataset(self.args.test_prefix, self.args)
        self.test_loader = DataLoader(test_set, 
                                       collate_fn=ppd.pad_collate,
                                       batch_size=self.args.batch_size, 
                                       num_workers=self.args.num_workers, 
                                       pin_memory=True)
        self.loss_fct = CrossEntropyLoss()
        if self.args.mode == 'train':            
            # Load optimizer
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            self.best_loss = sys.float_info.max
            self.last_epoch = 0
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset(self.args.train_prefix, self.args)
            valid_set = CustomDataset(self.args.valid_prefix, self.args)
            
            
            self.train_loader = DataLoader(train_set, 
                                           collate_fn=ppd.pad_collate, 
                                           shuffle=True, 
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            self.valid_loader = DataLoader(valid_set, 
                                           collate_fn=ppd.pad_collate,
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)

            
            if not os.path.exists(self.args.ckpt_dir):
                os.makedirs(self.args.ckpt_dir)
                
            # Calculate total training steps
            num_batches = len(self.train_loader)
            args.total_train_steps = args.num_epochs * num_batches
            args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)
            
            self.sched = get_polynomial_decay_schedule_with_warmup(
                self.optim,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.total_train_steps,
                power=2
            )
            
            # self.writer = SummaryWriter()
        
        if self.args.ckpt_name is not None:
            if self.args.loadtype == 'ckpt':
                ckpt_path = f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt"
                if os.path.exists(ckpt_path):
                    print("Loading the trained checkpoint...")
                    ckpt = torch.load(ckpt_path, map_location=self.args.device)
                    self.model.load_state_dict(ckpt['model_state_dict'])
                    
                    if self.args.mode == 'train':
                        print(f"The training restarts with the specified checkpoint: {self.args.ckpt_name}.ckpt.")
                        self.optim.load_state_dict(ckpt['optim_state_dict'])
                        self.sched.load_state_dict(ckpt['sched_state_dict'])
                        self.best_loss = ckpt['loss']
                        self.last_epoch = ckpt['epoch']
                    else:
                        print("The inference will start with the specified checkpoint.")
                else:
                    print(f"Cannot fine the specified checkpoint {ckpt_path}.")
                    if self.args.mode == 'train':
                        print("Training will start with the initialized model.")
                    else:
                        print("Cannot inference.")
                        exit()
            elif self.args.loadtype == 'pt':
                ckpt_path = f"{self.args.ckpt_dir}/{self.args.ckpt_name}.pt"
                if os.path.exists(ckpt_path):
                    print('Loading the previous model...')
                    if self.args.mode == 'train':
                        print(f"The training restarts with the specified checkpoint: {self.args.ckpt_name}.pt.")
                        
                        self.model = torch.load(ckpt_path)
                        digits = re.findall(r"\d+\.?\d*",self.args.ckpt_name)
                        self.last_epoch = int(digits[0])
                        self.best_loss = float(digits[1])
                        print(digits)
                    else:
                        print("The inference will start with the specified checkpoint.")
                else:
                    print(f"Cannot fine the specified checkpoint {ckpt_path}.")
                    if self.args.mode == 'train':
                        print("Training will start with the initialized model.")
                    else:
                        print("Cannot inference.")
                        exit()
              
        print("Setting finished.")
              
    def train(self):
        self.model.train()
        scaler = GradScaler()
        
        self.fix_seed(self.args.seed)  # Fix seed before training
        print("Training starts.")
        # print("Initial Testing")
        # self.test()
        start_epoch = self.last_epoch+1
        global_t = 0
        train_loss_list = []
        for epoch in range(start_epoch, start_epoch+self.args.num_epochs):
            
            
            print(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
            train_losses = []
            train_ppls = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                global_t += 1
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), labels.to(self.args.device)
                self.optim.zero_grad()
                with autocast():
                    model_output = self.model(
                        input_ids=input_ids,
                        token_type_ids = token_type_ids,
                        labels = labels,
                        # with_latent= self.args.with_latent
                    )
                    if self.args.model_fusion == 'com_latent_seg':
                        outputs,kl_loss,latent_embeds,hier_embeds,inputs_embeds,boundary = model_output
                    else:
                        outputs,kl_loss,latent_embeds,inputs_embeds ,boundary= model_output
                    
                    # pdb.set_trace()s
                    ce_loss, logits = outputs[0], outputs[1]
                    loss = ce_loss.mean() + min(global_t/self.args.full_kl_step,0.8) *kl_loss
                    # if i % 5 == 0:
                    #     print(outputs.cross_attentions)
                
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                self.sched.step()
                
                train_losses.append(loss.item())
                train_loss_list.append(loss.item())
                ppl = torch.exp(loss.detach())
                train_ppls.append(ppl)
                if i%5000 == 0:
                    print(f"Step{i} Loss{loss.detach()} C_loss{ce_loss.mean().detach()}")
                # if (i>0) and (i%40000==0):
                #     self.test(global_t)
                # if i == 20:
                #     break
                if i == 100000:
                    valid_loss, valid_ppl = self.validation(global_t)
                      
                    if valid_loss < self.best_loss:
                        self.best_loss = valid_loss
                        state_dict = {
                            'model_state_dict': self.model.state_dict(),
                            'optim_state_dict': self.optim.state_dict(),
                            'sched_state_dict': self.sched.state_dict(),
                            'loss': self.best_loss,
                            'epoch': self.last_epoch,
                            'global_t':global_t
                        }
                      
                        torch.save(state_dict, f"{self.args.ckpt_dir}/{self.args.special_words}_{self.args.vae_type}_{self.args.model_fusion}_{self.args.hier_type}_best_half_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
                        print("*"*10 + "Current best checkpoint is saved." + "*"*10)
                        print(f"{self.args.ckpt_dir}/best_half_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
                #     pdb.set_trace()
            del input_ids, token_type_ids, labels,outputs
            torch.cuda.empty_cache()

            
            # train_losses = [loss.item() for loss in train_losses]
            train_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls]
            train_loss = np.mean(train_losses)
            train_ppl = np.mean(train_ppls)
            print(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")
            
            # self.writer.add_scalar("Loss/train", train_loss, epoch)
            # self.writer.add_scalar("PPL/train", train_ppl, epoch)
            
            self.last_epoch += 1
            
            valid_loss, valid_ppl = self.validation(global_t)
            state_dict = {
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'sched_state_dict': self.sched.state_dict(),
                'loss': valid_loss,
                'epoch': self.last_epoch,
                'global_t':global_t
            }
            torch.save(state_dict, f'{self.args.ckpt_dir}/{self.args.special_words}_{self.args.vae_type}_{self.args.model_fusion}_{self.args.hier_type}_best_ckpt_epoch={epoch}_valid_loss={round(valid_loss, 4)}.ckpt')
            print(f"{self.args.ckpt_dir}/ckpt_epoch={epoch}_valid_loss={round(valid_loss, 4)}.ckpt")

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss

                print("*"*10 + "Current best checkpoint is saved." + "*"*10)
                              
            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")
            np.save(f'hier_{self.args.vae_type}_{self.args.model_fusion}_train_loss_list_epoch={epoch}_valid_loss={round(valid_loss, 4)}.npy',np.array(train_loss_list))
            self.test(global_t)
              
        print("Training finished!")
        
    
    def validation(self,global_t):
        print("Validation processing...")
        self.model.eval()
              
        valid_losses = []
        valid_ppls = []
        with torch.no_grad():
            for i, batch in enumerate(self.valid_loader):
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), labels.to(self.args.device)
                
                model_output= self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = labels,
                    # with_latent = self.args.with_latent
                    )
                if self.args.model_fusion == 'com_latent_seg':
                    outputs,kl_loss,latent_embeds,hier_embeds,inputs_embeds,boundary = model_output
                else:
                    outputs,kl_loss,latent_embeds,inputs_embeds ,boundary= model_output
                        
                # pdb.set_trace()
                ce_loss, logits = outputs[0], outputs[1]
                
                loss = ce_loss + min(global_t/self.args.full_kl_step,1) * kl_loss
            
                valid_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                valid_ppls.append(ppl)

            valid_losses = [loss.item() for loss in valid_losses]
            valid_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls]
            valid_loss = np.mean(valid_losses)
            valid_ppl = np.mean(valid_ppls)
            del input_ids, token_type_ids, labels,outputs
            torch.cuda.empty_cache()
            if math.isnan(valid_ppl):
                valid_ppl = 1e+8
        self.model.train()
              
        return valid_loss, valid_ppl
        
              
    # def infer(self):
    #     print("Let's start!")
    #     print(f"If you want to quit the conversation, please type \"{self.args.end_command}\".")
    #     self.model.eval()
    #     self.fix_seed(self.args.seed)
        
    #     with torch.no_grad():
    #         input_hists = []
            
    #         while True:
    #             utter = input("You: ")
    #             if utter == self.args.end_command:
    #                 print("Bot: Good bye.")
                #     break
                
                # input_ids = [self.args.sp1_id] + self.tokenizer.encode(utter)
                # input_hists.append(input_ids)
                
                # if len(input_hists) >= self.args.max_turns:
                #     num_exceeded = len(input_hists) - self.args.max_turns + 1
                #     input_hists = input_hists[num_exceeded:]
                    
                # input_ids = [self.args.bos_id] + list(chain.from_iterable(input_hists)) + [self.args.sp2_id]
                # start_sp_id = input_hists[0][0]
                # next_sp_id = self.args.sp1_id if start_sp_id == self.args.sp2_id else self.args.sp2_id
                # assert start_sp_id != next_sp_id
                # token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
                # assert len(token_type_ids) == len(input_hists)
                # token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.args.sp2_id]
                # assert len(input_ids) == len(token_type_ids)
                # input_len = len(input_ids)
                
                # input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args.device)
                # token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args.device)
                
                # output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)                
                # # output_ids = self.model.generate(
                # #     input_ids=input_ids, token_type_ids=token_type_ids, pad_token_id=self.args.eos_id,
                # #     do_sample=True, top_p=self.args.top_p, max_length=self.args.max_len,
                # #     output_hidden_states=True, output_scores=True, return_dict_in_generate=True,
                # # ).sequences
                # # output_ids = output_ids[0].tolist()[input_len:]
                # res = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                # print(f"Bot: {res}")
                # input_hists.append([self.args.sp2_id] + self.tokenizer.encode(res))
                
    def nucleus_sampling(self, input_ids_list, token_type_ids_list, next_speaker_id,latent_embeds,latent_inp_embeds,hier_embeds = None):

        output_id = []
        res_id = [next_speaker_id]
        res_type_id = [next_speaker_id]
        device = token_type_ids_list.device
        if not isinstance(input_ids_list,list):
            input_ids_list = input_ids_list.tolist()
            token_type_ids_list = token_type_ids_list.tolist()
        # pdb.set_trace()
        for pos in range(256):
            input_ids = list(chain.from_iterable(input_ids_list)) + res_id
            token_type_ids = list(chain.from_iterable(token_type_ids_list)) + res_type_id
            input_len = len(input_ids)
            
            # left = self.args.max_len - len(input_ids)
            # input_ids += [self.args.eos_id] * left
            # token_type_ids += [self.args.eos_id] * left

            assert len(input_ids) == len(token_type_ids), "There is something wrong in dialogue process."
            
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)  # (1, L)
            token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(device)  # (1, L)
            

            torch_res_ids = torch.LongTensor(res_id).unsqueeze(0).to(device)  # (1, L)
            torch_res_type_ids = torch.LongTensor(res_type_id).unsqueeze(0).to(device)  # (1, L)
            # print(latent_inp_embeds.shape)
            res_em = self.model.transformer.transformer.wte(torch_res_ids) + self.model.transformer.transformer.wte(torch_res_type_ids)
            # print(res_em.shape,res_em)
            if self.args.model_fusion == 'proj':
                inputs_embeds = torch.cat([latent_inp_embeds,res_em+latent_embeds],1)
                output = self.model.transformer(
                    inputs_embeds = inputs_embeds
                    )  # (1, vocab_size)
            else:
                inputs_embeds = torch.cat([latent_inp_embeds,res_em],1)
                if self.args.model_fusion == 'concat':
                    output = self.model.transformer(
                        inputs_embeds = inputs_embeds
                        )  # (1, vocab_size)
                elif self.args.model_fusion == 'layer_concat':
                    # pdb.set_trace()
                    output = self.model.transformer(
                        inputs_embeds = inputs_embeds,
                        past_key_values = latent_embeds,
                        )  # (1, vocab_size) 
                elif self.args.model_fusion == 'encoder_h_concat':
                    output = self.model.transformer(
                        inputs_embeds = inputs_embeds,
                        encoder_hidden_states = latent_embeds
                        )  # (1, vocab_size)       
                elif self.args.model_fusion == 'com_latent_seg':
                    output = self.model.transformer(
                        inputs_embeds = inputs_embeds,
                        past_key_values = latent_embeds,
                        encoder_hidden_states = hier_embeds
                        )  # (1, vocab_size)
                # print(output.cross_attentions[-1][0].shape)
            
            outputs = output.logits[:, input_len-1]
            # print(output.cross_attentions[-1][0].mean(dim=0).shape)
            
            # Random sampling
            probs = top_k_top_p_filtering(outputs, top_k=self.args.top_k, top_p=self.args.top_p)
            probs = F.softmax(probs, dim=-1)  # (1, vocab_size)
            idx = torch.multinomial(probs, 1).squeeze(-1).squeeze(0).item()
            
            if len(output_id) == 256 or idx == self.args.eos_id:
                break
            else:
                output_id.append(idx)
                res_id.append(idx)
                res_type_id.append(next_speaker_id)
            
        return output_id,output
    
    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    def distinct(self,hyps):
        intra_dist1, intra_dist2 = [], []
        unigrams_all, bigrams_all = Counter(), Counter()
        # print("hyps",list(hyps.values()))
        for hyp in list(hyps.values()):
            hyp = hyp[0].split()
            unigrams = Counter(hyp)
            bigrams = Counter(zip(hyp, hyp[1:]))
            intra_dist1.append((len(unigrams)+1e-12) / (len(hyp)+1e-5))
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(hyp)-1)+1e-5))
            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)

        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist1 = np.average(intra_dist1)
        intra_dist2 = np.average(intra_dist2)
        output_content = 'Distinct-1/2: {}/{}\n'.format(round(inter_dist1, 4), round(inter_dist2, 4))
        print(output_content)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2
        
    def COCO_evaluate(self,pred , res):
        assert isinstance(res,dict)
        assert isinstance(pred,dict)
    
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]       
        scorers_list = {}
        for scorer, method in scorers:
            _score, _ = scorer.compute_score(pred , res)
            if isinstance(method,list):
                for m,s in zip(method,_score):
                    scorers_list[m] = s 
            else:
                scorers_list[method] = _score
    
        #Rounge
        return scorers_list
        
    # def meteor_score(self, y_true, y_pred, preprocess = str.lower, stemmer = PorterStemmer(), wordnet=wordnet, alpha=0.9,
    #                  beta=3, gamma=0.5):
    #     total_score = 0
    #     # pdb.set_trace()
    #     for y_true_seq, y_pred_seq in zip(y_true.values(), y_pred.values()):
    #         references = [y_true_seq[0].split()]
    #         hypothesis = y_pred_seq[0].split()

    #         total_score += nltk_meteor_score(references, hypothesis,preprocess, stemmer, wordnet, alpha, beta, gamma)
    #         # , preprocess, stemmer, wordnet, alpha, beta, gamma

    #     return total_score / len(y_true)
        
    # def meteor_score1(self, y_true, y_pred, preprocess = str.lower, stemmer = PorterStemmer(), wordnet=wordnet, alpha=0.9,
    #                  beta=3, gamma=0.5):
    #     total_score = 0
    #     # pdb.set_trace()
    #     for y_true_seq, y_pred_seq in zip(y_true.values(), y_pred.values()):
    #         references = [y_true_seq[0].split()]
    #         hypothesis = y_pred_seq[0].split()

    #         total_score += nltk_meteor_score(references, hypothesis)
    #         # , preprocess, stemmer, wordnet, alpha, beta, gamma

    #     return total_score / len(y_true)
        
    def top_k_top_p_filtering_2(self,logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:

            sorted_logits, sorted_indices = torch.sort(logits, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -1:] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logtis = logits.masked_fill(indices_to_remove, filter_value)
        return logits
    

    def heatmap(self,output,res,output_id, boundary,input_ids,ind,segments):
        sns.set()
        if isinstance(segments,list):
            segments = torch.cat(segments,1)
        #utput.hidden_states[-1] = [1, 59, 768])
        attn_weights = torch.matmul(output.hidden_states[-1],segments.transpose(-1, -2))[:,-len(output_id):,:].squeeze().cpu()
        if self.args.hier_type == 'turn':
            column_seg = ['seq'+str(i+1) for i in range(len(segments[0]))]
        else:
            column_seg = ['seg'+str(i+1) for i in range(boundary.count(1))]
        # attn_weights
        p1 = sns.heatmap(pd.DataFrame(attn_weights,columns=column_seg),annot=False,yticklabels=False)
        plt.xticks(rotation=360,fontsize=8)
        p1.set_yticks([])
        s1 = p1.get_figure()
        
        s1.savefig(f'heatmap_turn/dot_seg_{self.args.special_words}_{ind}_{self.args.model_fusion}_{self.args.hier_type}.jpg',dpi=300,bbox_inches='tight')
        plt.show()
        
        # pdb.set_trace()
        if self.args.model_fusion == 'encoder_h_concat' or self.args.model_fusion == 'com_latent_seg':
            # if self.args.hier_type == 'jra_seg':
            if self.args.hier_type == 'block_seg_aggregate':
                attention = output.cross_attentions[-1][0][:,-len(output_id):,:].mean(dim=0).mean(dim=-1).cpu()
                column_seg = ['seg'+str(1)]
            
            else:
                #-input_ids.shape[1]
                # pdb.set_trace()
                attention = output.cross_attentions[-1][0][:,-len(output_id):,:].mean(dim=0).cpu() #30,3
                if boundary is not None:
                    column_seg = ['seg'+str(i+1) for i in range(boundary.count(1))]
                else:
                    column_seg = ['seg'+str(i+1) for i in range(len(segments[0]))]
            # attention[:,-1] = attention[:,-1]*1.5
            index = res.split()
            # pdb.set_trace()
            p1 = sns.heatmap(pd.DataFrame(attention,columns=column_seg),annot=False,yticklabels=False)
            plt.xticks(rotation=360,fontsize=8)
            p1.set_yticks([])
            s1 = p1.get_figure()
            s1.savefig(f'heatmap_turn/seg_{self.args.special_words}_{ind}_{self.args.model_fusion}_{self.args.hier_type}.jpg',dpi=300,bbox_inches='tight')
    
            plt.show()
        
        else:
            # attention = output.attentions[-1][0][:,1:,1+input_ids.shape[1]:].mean(dim=0).cpu()
            attention = output.attentions[-1][0][:,-len(output_id):,:4].mean(dim=0).cpu()
            p1 = sns.heatmap(attention.numpy().transpose(),annot=False,yticklabels=False)
            plt.xticks(rotation=360,fontsize=8)
            p1.set_yticks([])
            s1 = p1.get_figure()
            s1.savefig(f'heatmap_turn/seg_{self.args.special_words}_{ind}_{self.args.model_fusion}_{self.args.hier_type}.jpg',dpi=300,bbox_inches='tight')
            plt.show()
    
    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma
    def simi(self,output,embeds,output_id,segments,latents = None):
        # pdb.set_trace()
        # try:
            if self.args.model_fusion == 'com_latent_seg' or  self.args.model_fusion == 'encoder_h_concat':
                # pdb.set_trace()
                b,nh,s,hd =output.past_key_values[-1][0].size()
                latent = output.past_key_values[-1][0].reshape(b,s,-1)[:,0,:]
                # sim = cosine_similarity(latent,output.hidden_states[-1].squeeze())

                sim1 = max(cosine_similarity(latent.cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu()))[0]
                sim2 = max(cosine_similarity(embeds[0].cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu()))[0]
                
                eu1 = 1/math.exp(np.linalg.norm(latent.cpu() - torch.mean(output.hidden_states[-1],dim=1).cpu()))
                eu2 = 1/math.exp(np.linalg.norm(embeds[0].cpu() - torch.mean(output.hidden_states[-1],dim=1).cpu()))
                
                cor1 = np.corrcoef(latent.cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu())[0,-1]
                cor2 = np.corrcoef(embeds[0].cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu())[0,-1]

                # if self.args.hier_type == 'jra_seg':
                if self.args.hier_type == 'block_seg_aggregate'
                    att1 = torch.mean(output.cross_attentions[-1][0][:,-len(output_id):,:].mean(dim=0).mean(dim=-1),axis = 0).cpu().numpy() #15,1
                else:
                    att1 = torch.mean(output.cross_attentions[-1][0][:,-len(output_id):,:].mean(dim=0),axis = 0) #15,3
                    try:
                        if len(att1)>1:
                            att1 = [(len(att1)-i +1)/(len(att1)+1) * (att1[i])for i in range(1, len(att1))]
                            att1 = sum(att1).cpu().numpy()
                        else:
                            att1 = 1
                    except:
                        print('no att1')
                        att1 = max(att1).cpu().numpy()
                if isinstance(att1, torch.Tensor):
                    att1 = 0
                
            elif self.args.model_fusion == 'layer_concat':
                
                b,nh,s,hd =output.past_key_values[-1][0].size()
                latent = output.past_key_values[-1][0].reshape(b,s,-1)[:,0,:]
                # sim = cosine_similarity(latent,output.hidden_states[-1].squeeze())
                sim1 = max(cosine_similarity(latent.cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu()))[0]
                sim2 = 0
                
                eu1 = 1/math.exp(np.linalg.norm(latent.cpu()-torch.mean(output.hidden_states[-1],dim=1).cpu()))
                eu2 = 0
                # if len(segments[0])>1:
                    # pdb.set_trace()
                cor1 = np.corrcoef(latent.cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu())[0,-1]
                cor2 = 0
                # pdb.set_trace()
                # att1 = mean(output.attentions[-1][0][:,-len(output_id):,:4].mean(dim=0).cpu())
                att1 = max(torch.mean(output.attentions[-1][0][:,-len(output_id):,:4].mean(dim=0),axis = 0)).cpu().numpy()

            if isinstance(segments,list):
                segments = torch.cat(segments,1)
            # pdb.set_trace()

            sim3 = max(cosine_similarity(segments[0].cpu().numpy(),torch.mean(output.hidden_states[-1],dim=1).cpu().numpy()))[0]
            eu3 = 1/math.exp(np.linalg.norm(segments[0].cpu() - torch.mean(output.hidden_states[-1],dim=1).cpu()))

            # cor3 = np.corrcoef(segments[0].cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu())[0,-1]
            temp= np.corrcoef(segments[0].cpu(),torch.mean(output.hidden_states[-1],dim=1).cpu())
            cor3 = np.max(temp[temp<0.99999999])
            # pdb.set_trace()
            att2 = torch.mean(torch.matmul(output.hidden_states[-1],segments.transpose(-1, -2))[0,-len(output_id):,:],axis = 0)
            try:
                if len(att2)>1:
                    att2 = [(len(att2)-i +1)/(len(att2)+1) * (att2[i])for i in range(1, len(att2))]
                    att2 = sum(att2)
            except:
                print('no att2')
                att2 = max(att2)
            return [sim1,sim2,sim3,eu1,eu2,eu3,cor1,cor2,cor3,att1,att2.cpu().numpy()]

    def test_attn(self):
        from collections import defaultdict
        print("Testing processing...")
        self.model.eval()
        test_losses = []
        test_ppls = defaultdict(list)
        sim_list = []
        met_list = []
        num_t = {}
        tt = []
        # output_list = {}
        # target_list = {}
        # samples_file = open('generate_test.txt', 'w', encoding='utf8')
        output_length = {}
        target_length = {}
        for i in range(1,32):
            output_length[i] = {}
        for i in range(1,32):
            target_length[i] = {}
        
        with torch.no_grad():
            
                for i, batch in enumerate(tqdm(self.test_loader)):
                    # try:
                    if i < 1248:
                        continue
    
                    output_list = {}
                    target_list = {}
    
                    input_ids, token_type_ids, labels = batch
                    input_ids, token_type_ids, labels = \
                        input_ids.to(self.args.device), token_type_ids.to(self.args.device), labels.to(self.args.device)
                    input_len = len(input_ids)
                    # output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)
                    num_turn = list(input_ids.squeeze()).count(self.args.sp1_id)+list(input_ids.squeeze()).count(self.args.sp2_id)
                    # if num_turn != 8:
                    #     continue
    
                    # if i>2000 and i<8400 or i>9500:
                    #     if num_turn <= 22:
                    #         continue
                    # if i>2000 and num_turn <15:
                    #     continue
                    # if i>3000 and num_turn <22:
                        # continue
                        
                    num_t[num_turn] = num_t.get(num_turn,0)+1
    
                    # logits and loss
                    
                    model_output = self.model(
                        input_ids=input_ids,
                        token_type_ids = token_type_ids,
                        labels = labels,
                        # from_prior = True
                        # with_latent = self.args.with_latent,
                    )
                    if self.args.model_fusion == 'com_latent_seg':
                        outputs,kl_loss,latent_embeds,hier_embeds,inputs_embeds,boundary,latents,segments= model_output
                    else:
                        outputs,kl_loss,latent_embeds,inputs_embeds ,boundary,latents,segments= model_output
                    
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    
                    ce_loss = outputs.loss
                    # self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    # logits = outputs.logits
                    loss = ce_loss + min(10000/self.args.full_kl_step,1) * kl_loss
                    test_losses.append(loss.detach())
                    ppl = torch.exp(loss.detach())
                    tt.append(ppl)


                    #generating
                    last_type = token_type_ids[0][-1]
                    index = np.argwhere(input_ids.cpu() == last_type.cpu())[1][-1]
                    token_type_id = token_type_ids[:,:index]
                    input_id = input_ids[:,:index]
                    label = labels[:,index:]
                    
                    input_clone = input_id.clone()
                    input_clone[input_clone.eq(50258)]=-100
                    input_clone[input_clone.eq(50259)]=-100
                    num_turn = int(sum(input_clone.eq(-100).squeeze()))
    
    
                    if self.args.model_fusion == 'com_latent_seg':
                        outputs ,kl_loss,latent_embeds,hier_embeds,inputs_embeds ,boundary,latents,segments= self.model(
                            input_id,
                            token_type_id,
                            from_prior = True
                        )
                    else:
                        outputs ,kl_loss,latent_embeds,inputs_embeds,boundary,latents,segments = self.model(
                            input_id,
                            token_type_id,
                            from_prior = True
                        )
                        hier_embeds = None
                    if boundary is not None:
                        if boundary.count(1)<2:
                            continue
                    else:
                        if len(segments[0])<2:
                            continue
                    
                    try:
                        # output_id = [int(id) for id in output_id.squeeze()]
                        output_id ,output= self.nucleus_sampling(input_id, token_type_id, last_type,latent_embeds,inputs_embeds,hier_embeds=hier_embeds)
                        res = self.tokenizer.decode(output_id, skip_special_tokens=True)
                    except:
                        output_id = []
                        res = None
                        print('no res')
                        # continue
                    # pdb.set_trace()
                    
    
                    test_ppls[num_turn].append(ppl)
                    input_id = [int(id) for id in input_id.squeeze()]
                    context = self.tokenizer.decode(input_id, skip_special_tokens=True)
                    # if 'shoe' not in context:
                    #     continue
                    
                    # pdb.set_trace()
                    output_list[i] = [res]
                    target = labels[labels!=-100].squeeze()
                    target = self.tokenizer.decode(target, skip_special_tokens=True)
                    # pdb.set_trace()
                    target_list[i] = [target]
                    
                    if i == 716:
                        
                        self.heatmap(output,res,output_id,boundary,input_id,i,segments)
                        print(i,context,boundary)
                        print(res)
                        print(target)
                        break

        self.model.train()
        
    def test(self,global_t):
        print("Testing processing...")
        self.model.eval()
        test_losses = []
        test_ppls = []
        output_list = {}
        target_list = {}
        # samples_file = open('generate_test.txt', 'w', encoding='utf8')
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                # if i == len(self.test_loader)-1:
                #     break
                # if i <52:
                #     continue
                if i == 201:
                    break
                    
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), labels.to(self.args.device)
                input_len = len(input_ids)
            
                # output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)
                
                #logits and loss
                
                model_output = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = labels,
                    # from_prior = True
                    # with_latent = self.args.with_latent,
                )
                if self.args.model_fusion == 'com_latent_seg':
                    outputs,kl_loss,latent_embeds,hier_embeds,inputs_embeds ,boundary= model_output
                else:
                    outputs,kl_loss,latent_embeds,inputs_embeds,boundary = model_output
                # pdb.set_trace()
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                
                ce_loss = outputs.loss
                # self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # logits = outputs.logits
                loss = ce_loss + min(global_t/self.args.full_kl_step,1) * kl_loss
                test_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                test_ppls.append(ppl)
                
                # pdb.set_trace()
                #generating
                # token_type_ids = token_type_ids[input_ids!=labels].unsqueeze(0)
                # input_ids = input_ids[input_ids!=labels].unsqueeze(0)
                last_type = token_type_ids[0][-1]
                index = np.argwhere(input_ids.cpu() == last_type.cpu())[1][-1]
                token_type_id = token_type_ids[:,:index]
                input_id = input_ids[:,:index]
                # pdb.set_trace()
                label = labels[:,index:]

                if self.args.model_fusion == 'com_latent_seg':
                    outputs ,kl_loss,latent_embeds,hier_embeds,inputs_embeds,boundary = self.model(
                        input_id,
                        token_type_id,
                        # labels = labels,
                        # with_latent = self.args.with_latent,
                        # output_embeds = True,
                        from_prior = True
                    )
                else:
                    outputs ,kl_loss,latent_embeds,inputs_embeds,boundary = self.model(
                        input_id,
                        token_type_id,
                        # labels = labels,
                        # with_latent = self.args.with_latent,
                        # output_embeds = True,
                        from_prior = True
                    )
                    hier_embeds = None
                # if boundary.count(1) <3:
                #     # print(outputs.cross_attentions[-1][0].mean(dim=0))
                #     continue
                output_id ,output= self.nucleus_sampling(input_id, token_type_id, last_type,latent_embeds,inputs_embeds,hier_embeds=hier_embeds)
                
#                 # pdb.set_trace()
#                 generation_config = GenerationConfig(
#                     pad_token_id = self.tokenizer.eos_token_id,
#                     eos_token_id = self.tokenizer.eos_token_id,
#                     bos_token_id = self.tokenizer.bos_token_id,
#                     do_sample=True,
#                     max_length=128,
#                     top_k=50,
#                     top_p=0.95,
#                     num_return_sequences=1
#                     )
#                 # pdb.set_trace()
#                 # print(inputs_embeds)
#                 output_ids = self.model.transformer.generate(
#                     inputs_embeds = inputs_embeds,
#                     generation_config = generation_config
#                     )
#                 output_id = output_ids[:,input_ids.size(1):]
                try:
                    # output_id = [int(id) for id in output_id.squeeze()]
                    res = self.tokenizer.decode(output_id, skip_special_tokens=True)
                except:
                    output_id = []
                    res = None
                if not res:
                    print('no res')
                    continue
                # pdb.set_trace()
                
                input_id = [int(id) for id in input_id.squeeze()]
                context = self.tokenizer.decode(input_id, skip_special_tokens=True)
                
                # pdb.set_trace()
                output_list[i] = [res]
                # pdb.set_trace()
                # target = labels[i+1][2]
                
                target = labels[labels!=-100].squeeze()
                target = self.tokenizer.decode(target, skip_special_tokens=True)
                # pdb.set_trace()
                target_list[i] = [target]

                if i % 40 == 0 :
                    print("context: " , context)
                    print("generation:" , res)
                    print("target:" , target)
                    print('loss:',loss.detach())
                
            test_losses = [loss.item() for loss in test_losses]
            test_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in test_ppls]
            test_loss = np.mean(test_losses)
            test_ppl = np.mean(test_ppls)
            print(f"test_loss{test_loss}")
            print(f"test_ppl{test_ppl}")
            # np.save("test_loss.npy",np.array(test_losses))
            del input_ids, token_type_ids, labels,outputs
            torch.cuda.empty_cache()
            # pdb.set_trace()
            try:
                scores = self.COCO_evaluate(output_list, target_list)
                print("Evaluation",scores)
            except:
                print('No COCO scores')
            # try:
            #     me = self.meteor_score(target_list,output_list)
            #     # nltk_meteor_score(target_list,output_list, preprocess = str.lower, stemmer = PorterStemmer(), wordnet=wordnet, alpha=0.9,beta=3, gamma=0.5)
            #     print(f"METEOR{me}")
            # except:
            #     print('No METEOR')
                
            # try:
            #     me = self.meteor_score1(target_list,output_list)
            #     # nltk_meteor_score(target_list,output_list, preprocess = str.lower, stemmer = PorterStemmer(), wordnet=wordnet, alpha=0.9,beta=3, gamma=0.5)
            #     print(f"METEOR{me}")
            # except:
            #     print('No METEOR')
            try:
                _,_,_,_ = self.distinct(output_list)
            except:
                print('No Distinct Value')
            if math.isnan(test_ppl):
                test_ppl = 1e+8
        self.model.train()

    
    def test_length(self,global_t):
        from collections import defaultdict
        print("Testing processing...")
        self.model.eval()
        test_losses = []
        test_ppls = defaultdict(list)
        num_t = {}
        tt = []
        output_list = {}
        target_list = {}
        # samples_file = open('generate_test.txt', 'w', encoding='utf8')
        output_length = {}
        target_length = {}
        accuracy_stats = {
                        "bleu1": {},
                        "bleu2": {},
                        "bleu3": {},
                        "bleu4": {},
                        # "blue": {},
                        "meteor": {},
                        "rouge": {},
                        "cider": {},
                        "ppl": {}
        }
        for i in range(1,32):
            output_length[i] = {}
        for i in range(1,32):
            target_length[i] = {}
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), labels.to(self.args.device)
                input_len = len(input_ids)
                # output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)
                num_turn = list(input_ids.squeeze()).count(self.args.sp1_id)+list(input_ids.squeeze()).count(self.args.sp2_id)
                

                # logits and loss
                model_output = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = labels,
                    # from_prior = True
                    # with_latent = self.args.with_latent,
                )
                if self.args.model_fusion == 'com_latent_seg':
                    outputs,kl_loss,latent_embeds,hier_embeds,inputs_embeds,boundary,latents,segments= model_output
                else:
                    outputs,kl_loss,latent_embeds,inputs_embeds,boundary,latents,segments= model_output
                # pdb.set_trace()
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                
                ce_loss = outputs.loss
                # self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # logits = outputs.logits
                loss = ce_loss + min(global_t/self.args.full_kl_step,1) * kl_loss
                test_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                tt.append(ppl)
                
                last_type = token_type_ids[0][-1]
                index = np.argwhere(input_ids.cpu() == last_type.cpu())[1][-1]
                token_type_id = token_type_ids[:,:index]
                input_id = input_ids[:,:index]
                label = labels[:,index:]
                
                input_clone = input_id.clone()
                input_clone[input_clone.eq(50258)]=-100
                input_clone[input_clone.eq(50259)]=-100
                num_turn = int(sum(input_clone.eq(-100).squeeze()))
		num_t[num_turn] = num_t.get(num_turn,0)+1

                if self.args.model_fusion == 'com_latent_seg':
                    outputs ,kl_loss,latent_embeds,embeds,inputs_embeds,boundary,latents,segments= self.model(
                        input_id,
                        token_type_id,
                        from_prior = True
                    )
                else:
                    outputs ,kl_loss,latent_embeds,inputs_embeds,boundary,latents,segments= self.model(
                        input_id,
                        token_type_id,
                        from_prior = True
                    )
                    hier_embeds = None
                if self.args.model_fusion == 'layer_concat':
                    latent_embeds = outputs.past_key_values
                    
                output_id ,output= self.nucleus_sampling(input_id, token_type_id, last_type,latent_embeds,inputs_embeds,hier_embeds=hier_embeds)
                try:
                    # output_id = [int(id) for id in output_id.squeeze()]
                    res = self.tokenizer.decode(output_id, skip_special_tokens=True)
                except:
                    output_id = []
                    res = None
                if not res:
                    print('no res')
                    continue
                # pdb.set_trace()
                

                test_ppls[num_turn].append(ppl)
                input_id = [int(id) for id in input_id.squeeze()]
                context = self.tokenizer.decode(input_id, skip_special_tokens=True)
                
                # pdb.set_trace()
                output_list[i] = [res]
                # pdb.set_trace()
                # target = labels[i+1][2]
                
                target = labels[labels!=-100].squeeze()
                target = self.tokenizer.decode(target, skip_special_tokens=True)
                # pdb.set_trace()
                target_list[i] = [target]
                
                if list(target_length[num_turn]):
                    # pdb.set_trace()
                    target_length[num_turn][list(target_length[num_turn])[-1]+1] = [target]
                    output_length[num_turn][list(output_length[num_turn])[-1]+1] = [res]
                else:
                    target_length[num_turn][0] = [target]
                    output_length[num_turn][0] = [res]

                if i % 40 == 0:
                    print("context: " , context)
                    print("generation:" , res)
                    print("target:" , target)
                    print('loss:',loss.detach())
                    print(num_turn)
                
            print(num_t)
            test_losses = [loss.item() for loss in test_losses]
            for key,value in test_ppls.items():
                test_pp = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in value]
                test_ppls[key] = np.mean(test_pp)
            accuracy_stats["ppl"] = test_ppls
            test_loss = np.mean(test_losses)
            test_p = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in tt]
            test_p = np.mean(test_p)
            
            print(f"test_loss{test_loss}")
            print(f"test_ppl{test_p}")
            print(f"all_test_ppl{test_ppls}")
            # np.save("test_loss.npy",np.array(test_losses))
            del input_ids, token_type_ids, labels,outputs
            torch.cuda.empty_cache()
            try:
                scores = self.COCO_evaluate(output_list, target_list)
                print("Evaluation",scores)
            except:
                print('No COCO scores')
            try:
                _,_,_,_ = self.distinct(output_list)
            except:
                print('No Total Distinct Value')
            assert len(target_length) == len(output_length)
            for i in range(len(target_length)):
                if not target_length[i+1]:
                    continue
                # if i % 2 != 0:
                #     continue
                try:

                    scores = self.COCO_evaluate(output_length[i+1], target_length[i+1])
                    print(f"The {i+2} turns evaluation",scores)
                    accuracy_stats["bleu1"][i+1] = scores['Bleu_1']
                    accuracy_stats["bleu2"][i+1] = scores['Bleu_2']
                    accuracy_stats["bleu3"][i+1] = scores['Bleu_3']
                    accuracy_stats["bleu4"][i+1] = scores['Bleu_4']
                    # accuracy_stats["blue"][i+1] = (scores['Bleu_1']+scores['Bleu_2']+scores['Bleu_3']+scores['Bleu_4'])/4
                    accuracy_stats["meteor"][i+1] = scores['METEOR']
                    accuracy_stats["rouge"][i+1] = scores['ROUGE_L']
                    accuracy_stats["cider"][i+1] = scores['CIDEr']
                    
                except:
                    print(f'{i+1} turns have no COCO scores')
                    print(output_length[i+1])
                try:
                    _,_,_,_ = self.distinct(output_length[i+1])
                except:
                    print(f'{i+1} turns have no Distinct Value')
            if math.isnan(test_p):
                test_p = 1e+8

            
            ticks = np.arange(len(accuracy_stats["blue4"]))

            group_num = len(accuracy_stats.keys())

            group_width = 0.8

            bar_span = group_width / group_num

            bar_width = bar_span
 
            baseline_x = ticks - (group_width - bar_span) / 2
            # pdb.set_trace()
            # accuracy_stats = dict(sorted(accuracy_stats.items(),key = lambda x:x[0]))
            print(accuracy_stats)
            np.save(f'acc_{self.args.special_words}_{self.args.vae_type}_{self.args.model_fusion}_{self.args.hier_type}.npy', accuracy_stats)
            # for index, (key,y) in enumerate(accuracy_stats.items()):
            #     y = dict(sorted(y.items(),key = lambda x:x[0]))
            #     if index== len(accuracy_stats)-2:
            #         # pdb.set_trace()
            #         break
            #     plt.bar(baseline_x + index*bar_span, y.values(), bar_width,label = key)
            # plt.ylabel('Metrics')
            # plt.ylim((0,0.75))
            # plt.title(f'1-Role Self-Seg in PC')

            # # pdb.set_trace()
            # plt.xticks(ticks, accuracy_stats["blue4"].keys())
            # plt.xlabel('Length of Context Turns')
            # plt.legend()
            # plt.savefig(f'length_{self.args.special_words}_{self.args.vae_type}_{self.args.model_fusion}_{self.args.hier_type}.jpg',dpi=600)
            # plt.show()
        self.model.train()
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--mode', type=str, required=True, help="The running mode: train or inference?")
    parser.add_argument('--data_dir', type=str, default="gpt2-dialogue", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument('--test_prefix', type=str, default="valid", help="The prefix of the validation data files' name.")
    
    parser.add_argument('--vae_type', type=str , default = 'avg_attn',required = True)
    parser.add_argument('--hier_type', type=str , default = '',required = True)
    parser.add_argument('--model_fusion', type=str , default = 'encoder_h_concat')
    parser.add_argument('--special_words', type=str , default = 're_label')
    parser.add_argument('--use_old', action = 'store_true')
    parser.add_argument('--with_latent', action = 'store_true')
    parser.add_argument('--with_hier', action = 'store_true')
    parser.add_argument('--low_resource', action = 'store_true')
    parser.add_argument('--model_type', type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument('--bos_token', type=str, default="<bos>", help="The BOS token.")
    parser.add_argument('--eos_token', type=str, default="<|endoftext|>", help="The EOS token.")
    parser.add_argument('--sp1_token', type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument('--sp2_token', type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    parser.add_argument('--lr', type=float, default=2e-5, help="The learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--batch_size', type=int, default=1
    , help="The batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--full_kl_step', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=30, help="The number of total epochs.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument('--max_turns', type=int, default=35, help="The maximum number of dialogue histories to include.")
    parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--top_k', type=int, default=100, help="The top-k value for nucleus sampling decoding.")
    parser.add_argument('--ckpt_dir', type=str, default="gpt2-dialogue", help="The directory name for saved checkpoints.")
    parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--end_command', type=str, default="Abort!", help="The command to stop the conversation when inferencing.")
    parser.add_argument('--loadtype', type=str , default = 'ckpt')
    parser.add_argument('--last_epoch', type=int, default=1,required= False)
    parser.add_argument('--best_loss', type=float, default=float('inf'),required=False)
                    
    args = parser.parse_args()
    
    assert args.mode in ["train", "infer","test","test_length","test_attn"]
    assert args.model_type in [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"
    ]
    
    args.data_dir = f"data"
    args.ckpt_dir = f""
              
    if args.mode == 'train':
        manager = Manager(args)
        manager.train()
        
    elif args.mode == 'infer':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
        
        manager = Manager(args)
        manager.infer()
    
    elif args.mode == 'test':
        manager = Manager(args)
        manager.test(1)
        
    elif args.mode == 'test_length':
        manager = Manager(args)
        manager.test_length(1)
        
    elif args.mode == 'test_attn':
        manager = Manager(args)
        manager.test_attn()
