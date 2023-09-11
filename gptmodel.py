import collections
import json
import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pdb
import copy

from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers.models.gpt2.modeling_gpt2 import *
# import torch.nn.functional.gelu as gelu
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.file_utils import add_start_docstrings
from segmentation import *

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations= weighted.sum(1).squeeze(1)

        return representations, scores

class MultAttn(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim


        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        self.q_attn = Conv1D(self.embed_dim, self.embed_dim)

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)


        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
        
class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_k = int(args.n_embd / args.n_head)
        self.d_v = int(args.n_embd / args.n_head)
        self.n_head = args.n_head
        self.W_Q = nn.Linear(args.n_embd, self.d_k * args.n_head)  # init (512 x 64 * 8)
        self.W_K = nn.Linear(args.n_embd, self.d_k * args.n_head)
        self.W_V = nn.Linear(args.n_embd, self.d_v * args.n_head)
        self.li1 = nn.Linear(args.n_head * self.d_v, args.n_embd)
        self.layer_norm = nn.LayerNorm(args.n_embd)
        
        torch.nn.init.kaiming_normal_(self.W_Q.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.W_K.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.W_V.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.li1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
          
    def ScaledDotProductAttention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)
        if Q.size(2) == K.size(2):
            # mask = mask.unsqueeze(1)
            # scores = scores.masked_fill(mask == 0, -1e9)
            if mask is not None:
                if mask.any() == -1e10:
                    pass
                else:
                    mask = mask.float().masked_fill(mask.unsqueeze(0).unsqueeze(1) == 0, -1e10).masked_fill(mask.unsqueeze(0).unsqueeze(1) == 1, float(0.0))
                scores = scores+mask
        scores = F.softmax(scores, dim=-1)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, Q, K, V, attn_mask=None):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # q_s:[batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # k_s:[batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)  # v_s:[batch_size x n_heads x len_q x d_v]

        context, attn = self.ScaledDotProductAttention(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        output = self.li1(context)

        return self.layer_norm(output + residual)
        # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.n_embd, out_channels=args.n_embd*2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.n_embd*2, out_channels=args.n_embd, kernel_size=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.layer_norm = nn.LayerNorm(args.n_embd)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
        
class model_vae(GPT2LMHeadModel):
    def __init__(self, config , gpt2_model,with_latent = False,with_hier = False,hier_type = None, vae_fusion = 'proj',vae_type = None,gpt2_vae_prior=None,gpt2_vae_post=None):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = gpt2_model
        self.vae_type = vae_type
        self.vae_fusion = vae_fusion
        self.with_latent = with_latent
        self.hier_type = hier_type
        self.config = config
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.with_latent:
            if vae_type == 'gpt2model':
                self.gpt2_vae_prior = gpt2_vae_prior
                self.gpt2_vae_post = gpt2_vae_post
            else:
                self.multi_vae_prior = MultiheadAttention(config)
                self.multi_vae_post = MultiheadAttention(config)
                self.multi_vae_prior_inter = MultiheadAttention(config)
                self.multi_vae_post_inter = MultiheadAttention(config)
                
                self.multi_vae_prior.W_Q.weight = self.multi_vae_post.W_Q.weight 
                self.multi_vae_prior.W_K.weight = self.multi_vae_post.W_K.weight 
                self.multi_vae_prior.W_V.weight = self.multi_vae_post.W_V.weight 
                self.multi_vae_prior.li1_weight = self.multi_vae_post.li1.weight
                self.multi_vae_prior.W_Q.bias = self.multi_vae_post.W_Q.bias
                self.multi_vae_prior.W_K.bias = self.multi_vae_post.W_K.bias
                self.multi_vae_prior.W_V.bias = self.multi_vae_post.W_V.bias
                self.multi_vae_prior.li1.bias = self.multi_vae_post.li1.bias
                
                self.multi_vae_prior_inter.W_Q.weight = self.multi_vae_post_inter.W_Q.weight 
                self.multi_vae_prior_inter.W_K.weight = self.multi_vae_post_inter.W_K.weight 
                self.multi_vae_prior_inter.W_V.weight = self.multi_vae_post_inter.W_V.weight 
                self.multi_vae_prior_inter.li1_weight = self.multi_vae_post_inter.li1.weight
                self.multi_vae_prior_inter.W_Q.bias = self.multi_vae_post_inter.W_Q.bias
                self.multi_vae_prior_inter.W_K.bias = self.multi_vae_post_inter.W_K.bias
                self.multi_vae_prior_inter.W_V.bias = self.multi_vae_post_inter.W_V.bias
                self.multi_vae_prior_inter.li1.bias = self.multi_vae_post_inter.li1.bias
                
                self.feed1 = PoswiseFeedForwardNet(config)
                self.feed2 = PoswiseFeedForwardNet(config)
                self.feed1.conv1.weight = self.feed2.conv1.weight 
                self.feed1.conv2.weight = self.feed2.conv2.weight
                self.feed1.conv1.bias = self.feed2.conv1.bias
                self.feed1.conv2.bias = self.feed2.conv2.bias
                self.feed1.layer_norm.weight = self.feed2.layer_norm.weight
                self.feed1.layer_norm.bias = self.feed2.layer_norm.bias
                
            self.prior_atten = AverageSelfAttention(config.n_embd)
            self.post_atten = AverageSelfAttention(config.n_embd)
            self.prior_atten.attention_weights = self.post_atten.attention_weights
            self.prior_linear1 = nn.Linear(config.n_embd , config.n_embd)
            self.prior_linear2 = nn.Linear(config.n_embd , 2*config.n_embd)
            self.post_linear1 = nn.Linear(config.n_embd , config.n_embd)
            self.post_linear2 = nn.Linear(config.n_embd , 2*config.n_embd)
            self.latent_norm1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
            self.latent_norm2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
            
            torch.nn.init.kaiming_normal_(self.prior_linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(self.prior_linear2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            self.post_linear1.weight = self.prior_linear1.weight
            self.post_linear2.weight = self.prior_linear2.weight

            
        # self.drop = nn.Dropout(config.embd_pdrop)
        if self.vae_fusion == 'layer_concat' or self.vae_fusion == 'com_latent_seg':
            self.c_z = Conv1D(config.n_embd * 2, config.n_embd)
            self.latent_proj = nn.Linear(config.n_embd , config.n_layer *config.n_embd)
            self.latent_norm3 = nn.LayerNorm(config.n_layer *config.n_embd, eps=config.layer_norm_epsilon)
        else:
            self.latent_proj = nn.Linear(config.n_embd , config.n_embd)
            self.latent_norm3 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        torch.nn.init.kaiming_normal_(self.latent_proj.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.with_hier  = with_hier 
        if self.with_hier:
            self.word_level = nn.GRU(input_size=config.n_embd, hidden_size=config.n_embd, batch_first=True,bidirectional = True)
            self.uttr_level = nn.GRU(input_size=config.n_embd, hidden_size=config.n_embd, batch_first=True,bidirectional = True)
            # self.init_h1 = nn.Parameter(torch.zeros([1,2,config.n_embd]))
            # self.init_h2 = nn.Parameter(torch.zeros([1,2,config.n_embd]))
            self.seqs_linear = nn.Linear(2 * config.n_embd , config.n_embd)
            torch.nn.init.kaiming_normal_(self.seqs_linear.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            self.segs_linear = nn.Linear(2 * config.n_embd , config.n_embd)
            torch.nn.init.kaiming_normal_(self.segs_linear.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            self.cluster = C99(window = 4, std_coeff = 1)
            self.inter_attn = MultiheadAttention(config)
            self.feed1 = PoswiseFeedForwardNet(config)
            # self.feed2 = PoswiseFeedForwardNet(config)
            self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
            # self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
            if self.hier_type == 'jra_seg':
                self.anchor_attn = MultiheadAttention(config)
            elif self.hier_type == 'block_seg_aggregate':
                self.aggregate = nn.GRU(input_size=config.n_embd, hidden_size=config.n_embd, batch_first=True,bidirectional = True)
                self.agg_linear = nn.Linear(2 * config.n_embd , config.n_embd)
                torch.nn.init.kaiming_normal_(self.agg_linear.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        
    def reparameterize(self, mean, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.zeros_like(std).normal_()
        return mean + torch.mul(eps, std)
    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()
    
    def hie_layer(self,input_i,index,inputs_em):
        
        input_ids = input_i.clone()[:,1:index]
        inputs_embeds = inputs_em.clone()[:,1:index,:]
        input_ids[input_ids.eq(50258)]=-100
        input_ids[input_ids.eq(50259)]=-100
        num_sen = sum(input_ids.eq(-100).squeeze())
        index = np.argwhere(input_ids.cpu()==-100)
        seqs = torch.zeros([index.size(1),2*inputs_embeds.size(-1)],device = inputs_embeds.device)
        #get utterance
        pos = 0
        
        for i,ind in enumerate(index[1]):
            if i == 0:
                continue
            sent = inputs_embeds[:,pos:ind]
            _, seq = self.word_level(sent)
            seqs[i-1] = seq.view(1,1,-1)
            pos = ind
        
        seqs[-1] = self.word_level(inputs_embeds[:,pos:])[1].view(1,1,-1)
        seqs = seqs.unsqueeze(0)
        seqs = self.seqs_linear(seqs)
        
        #get segment
        boundary = self.cluster.segment(seqs.squeeze(0).detach().cpu().numpy())
        pos= 0
        output = []
        segments = []
        # pdb.set_trace()
        for b in range(1 , len(boundary)):
            if boundary[b] == 1:
                s = seqs[:,pos:b]
                _,segment_state = self.uttr_level(s.to(self.device))
                # print(segment_state.view(1,1,-1))
                if segment_state.view(1,1,-1).size == (1, 1, 0):
                    pdb.set_trace()
                segments.append(segment_state.view(1,1,-1))
                
                del s
                torch.cuda.empty_cache()
                # output.append(interact_embed)
                pos = b
                if pos == len(boundary):
                    break
        # pdb.set_trace()
        _,segment_state = self.uttr_level(seqs[:,pos:])
        if pos == 0 or (pos >0 and pos <len(boundary)):
            segments.append(segment_state.view(1,1,-1))
            # output.append(interact_embed)
        
        try:
            segments = torch.cat(segments,1)
        except:
            pdb.set_trace()
        segments = self.ln_1(self.segs_linear(segments))
        
        #interact
        #reverse
        # output = self.inter_attn(inputs_em ,segments,segments)
        #right
        output = self.inter_attn(segments,inputs_em ,inputs_em)
        
        # interact_embed = self.inter_attn(inputs_embeds[:,index[1][pos:b]] ,self.ln_1(segment_state),self.ln_1(segment_state))
        
        del segment_state,seqs
        torch.cuda.empty_cache()
        # output = torch.LongTensor(output)
        # pdb.set_trace()
        output = self.feed1(output)
        # output = torch.concat([inputs_embeds[:,0].unsqueeze(1),output, inputs_embeds[:,-1].unsqueeze(1)],1)
        
        return output,boundary,segments
    
    def hie_layer_selfseg(self,input_i,index,inputs_em):
        
        input_ids = input_i.clone()[:,1:index]
        inputs_embeds = inputs_em.clone()[:,1:index,:]
        input_ids[input_ids.eq(50258)]=-100
        input_ids[input_ids.eq(50259)]=-100
        num_sen = sum(input_ids.eq(-100).squeeze())
        index = np.argwhere(input_ids.cpu()==-100)
        seqs = torch.zeros([index.size(1),2*inputs_embeds.size(-1)],device = inputs_embeds.device)
        #get utterance
        pos = 0
        
        for i,ind in enumerate(index[1]):
            if i == 0:
                continue
            sent = inputs_embeds[:,pos:ind]
            _, seq = self.word_level(sent)
            seqs[i-1] = seq.view(1,1,-1)
            pos = ind
        
        seqs[-1] = self.word_level(inputs_embeds[:,pos:])[1].view(1,1,-1)
        seqs = seqs.unsqueeze(0)
        seqs = self.seqs_linear(seqs)
        
        #get segment
        boundary = self.cluster.segment(seqs.squeeze(0).detach().cpu().numpy())
        pos= 0
        output = []
        segments = []
        # pdb.set_trace()
        for b in range(1 , len(boundary)):
            if boundary[b] == 1:
                s = seqs[:,pos:b]
                _,segment_state = self.uttr_level(s.to(self.device))
                # print(segment_state.view(1,1,-1))
                if segment_state.view(1,1,-1).size == (1, 1, 0):
                    pdb.set_trace()
                segments.append(segment_state.view(1,1,-1))
                
                del s
                torch.cuda.empty_cache()
                # output.append(interact_embed)
                pos = b
                if pos == len(boundary):
                    break
        # pdb.set_trace()
        _,segment_state = self.uttr_level(seqs[:,pos:])
        if pos == 0 or (pos >0 and pos <len(boundary)):
            segments.append(segment_state.view(1,1,-1))
            # output.append(interact_embed)
        
        try:
            segments = torch.cat(segments,1)
        except:
            pdb.set_trace()
        segments = self.ln_1(self.segs_linear(segments))
        
        #interact
        output = self.inter_attn(segments ,segments,segments)
        
        del segment_state,seqs
        torch.cuda.empty_cache()
        output = self.feed1(output)
        return output,boundary,segments
        
    def hie_layer_blockseg(self,input_i,index,inputs_em):
        
        input_ids = input_i.clone()[:,1:index]
        inputs_embeds = inputs_em.clone()[:,1:index,:]
        input_ids[input_ids.eq(50258)]=-100
        input_ids[input_ids.eq(50259)]=-100
        num_sen = sum(input_ids.eq(-100).squeeze())
        index = np.argwhere(input_ids.cpu()==-100)
        seqs = torch.zeros([index.size(1),2*inputs_embeds.size(-1)],device = inputs_embeds.device)
        #get utterance
        pos = 0
        
        for i,ind in enumerate(index[1]):
            if i == 0:
                continue
            sent = inputs_embeds[:,pos:ind]
            _, seq = self.word_level(sent)
            seqs[i-1] = seq.view(1,1,-1)
            pos = ind
        
        seqs[-1] = self.word_level(inputs_embeds[:,pos:])[1].view(1,1,-1)
        seqs = seqs.unsqueeze(0)
        seqs = self.seqs_linear(seqs)
        
        #get segment
        boundary = self.cluster.segment(seqs.squeeze(0).detach().cpu().numpy())
        pos= 0
        output = []
        segments = []
        # pdb.set_trace()
        for b in range(1 , len(boundary)):
            if boundary[b] == 1:
                s = seqs[:,pos:b] 
                _,segment_state = self.uttr_level(inputs_embeds[:,index[1][pos]:index[1][b]])
                segments.append(segment_state.view(1,1,-1))
                
                del s
                torch.cuda.empty_cache()
                # output.append(interact_embed)
                pos = b
                if pos == len(boundary):
                    break
        # pdb.set_trace()
        _,segment_state = self.uttr_level(inputs_embeds[:,index[1][pos]:])
        if pos == 0 or (pos >0 and pos <len(boundary)):
            segments.append(segment_state.view(1,1,-1))
        try:
            segments = torch.cat(segments,1)
        except:
            pdb.set_trace()
        segments = self.ln_1(self.segs_linear(segments))
        
        if self.hier_type == 'block_seg_inter':
            #interact
            output = self.inter_attn(segments,inputs_em,inputs_em)
        elif self.hier_type == 'block_seg':
            output = self.inter_attn(segments ,segments,segments)
        elif self.hier_type == 'block_seg_aggregate':
            output = self.aggregate(segments)[1].view(1,1,-1)
            output = self.agg_linear(output)
        
        del segment_state,seqs
        torch.cuda.empty_cache()
        output = self.feed1(output)
        
        return output,boundary,segments
    
    def hie_layer_sep(self,input_i,index,inputs_em):
        
        input_ids = input_i.clone()[:,1:index]
        inputs_embeds = inputs_em.clone()[:,1:index,:]
        input_ids[input_ids.eq(50258)]=-100
        input_ids[input_ids.eq(50259)]=-100
        num_sen = sum(input_ids.eq(-100).squeeze())
        index = np.argwhere(input_ids.cpu()==-100)
        seqs = torch.zeros([index.size(1),2*inputs_embeds.size(-1)],device = inputs_embeds.device)
        #get utterance
        pos = 0
        
        for i,ind in enumerate(index[1]):
            if i == 0:
                continue
            sent = inputs_embeds[:,pos:ind]
            _, seq = self.word_level(sent)
            seqs[i-1] = seq.view(1,1,-1)
            pos = ind
        
        seqs[-1] = self.word_level(inputs_embeds[:,pos:])[1].view(1,1,-1)
        seqs = seqs.unsqueeze(0)
        seqs = self.seqs_linear(seqs)
        
        #get segment
        boundary = self.cluster.segment(seqs.squeeze(0).detach().cpu().numpy())
        pos= 0
        output = []
        segments = []
        
        for b in range(1 , len(boundary)):
            if boundary[b] == 1:
                
                s = seqs[:,pos:b]
                _,segment_state = self.uttr_level(s.to(self.device))
                # print(segment_state.view(1,1,-1))
                segment_state = self.ln_1(self.segs_linear(segment_state.view(1,1,-1))) 
                #reverse
                # segments.append(self.inter_attn(inputs_embeds[:,index[1][pos]:index[1][b]] ,segment_state,segment_state))
                #right
                segments.append(self.inter_attn(segment_state,inputs_embeds[:,index[1][pos]:index[1][b]],inputs_embeds[:,index[1][pos]:index[1][b]]))
                
                
                del s
                torch.cuda.empty_cache()
                # output.append(interact_embed)
                pos = b
                if pos == len(boundary):
                    break
        # pdb.set_trace()
        _,segment_state = self.uttr_level(seqs[:,pos:])
        if pos == 0 or (pos >0 and pos <len(boundary)):
            segment_state = self.ln_1(self.segs_linear(segment_state.view(1,1,-1))) 
            #reverse
            # segments.append(self.inter_attn(inputs_embeds[:,index[1][pos]:],segment_state,segment_state))
            #right
            segments.append(self.inter_attn(segment_state,inputs_embeds[:,index[1][pos]:],inputs_embeds[:,index[1][pos]:]))
            # segments.append(segment_state.view(1,1,-1))
            # output.append(interact_embed)
        try:
            segments = torch.cat(segments,1)
        except:
            pdb.set_trace()
        
        #interact
        # pdb.set_trace()
        # output = torch.cat([inputs_em[:,0].unsqueeze(1),output],1)
        # assert output.size() == inputs_embeds.size() , 'Different size of seg_embedding'
        del segment_state,seqs
        torch.cuda.empty_cache()
        # output = torch.LongTensor(output)
        # pdb.set_trace()
        output = self.feed1(segments)
        # output = torch.concat([inputs_embeds[:,0].unsqueeze(1),output, inputs_embeds[:,-1].unsqueeze(1)],1)
        
        return output,boundary,segments
        
    def hie_layer_JRA(self,input_i,index,index_anchor,inputs_em):
        if index_anchor == 1:
            # pdb.set_trace()
            anchor_ids = input_i.clone()[:,index_anchor:index]
            anchor_embeds = inputs_em.clone()[:,index_anchor:index]
            #anchor attention layer
            anchor_seq = self.word_level(anchor_embeds)[0]
            anchor_seq = self.seqs_linear(anchor_seq)
            anchor_attn = self.anchor_attn(anchor_seq,anchor_seq,anchor_seq)
            # anchor_attn = self.ln_1(self.segs_linear(self.uttr_level(anchor_attn)[1].view(1,1,-1)))
            output = self.inter_attn(anchor_attn,inputs_em.clone()[:,:index_anchor],inputs_em.clone()[:,:index_anchor])
            # pdb.set_trace()
            return self.feed1(output),[1],output
            
        input_ids = input_i.clone()[:,1:index_anchor] #previous context of last uttr
        anchor_ids = input_i.clone()[:,index_anchor:index] 
        inputs_embeds = inputs_em.clone()[:,1:index_anchor,:] 
        anchor_embeds = inputs_em.clone()[:,index_anchor:index] 
        input_ids[input_ids.eq(50258)]=-100
        input_ids[input_ids.eq(50259)]=-100
        num_sen = sum(input_ids.eq(-100).squeeze())
        index = np.argwhere(input_ids.cpu()==-100)
        seqs = torch.zeros([index.size(1),2*inputs_embeds.size(-1)],device = inputs_embeds.device)
        #get utterance
        pos = 0
        
        for i,ind in enumerate(index[1]):
            if i == 0:
                continue
            sent = inputs_embeds[:,pos:ind]
            _, seq = self.word_level(sent)
            seqs[i-1] = seq.view(1,1,-1)
            pos = ind
        
        seqs[-1] = self.word_level(inputs_embeds[:,pos:])[1].view(1,1,-1)
        seqs = seqs.unsqueeze(0)
        seqs = self.seqs_linear(seqs)
        
        #anchor attention layer
        anchor_seq = self.word_level(anchor_embeds)[0]
        anchor_seq = self.seqs_linear(anchor_seq)
        anchor_attn = self.anchor_attn(anchor_seq,anchor_seq,anchor_seq)
        
        
        #get segment
        boundary = self.cluster.segment(seqs.squeeze(0).detach().cpu().numpy())
        pos= 0
        output = []
        segments = []
        
        for b in range(1 , len(boundary)):
            if boundary[b] == 1:
                
                s = seqs[:,pos:b]
                _,segment_state = self.uttr_level(s.to(self.device))
                # print(segment_state.view(1,1,-1))
                 
                segments.append(segment_state.view(1,1,-1))
                
                del s
                torch.cuda.empty_cache()
                # output.append(interact_embed)
                pos = b
                if pos == len(boundary):
                    break
        # pdb.set_trace()
        _,segment_state = self.uttr_level(seqs[:,pos:])
        if pos == 0 or (pos >0 and pos <len(boundary)):
            segments.append(segment_state.view(1,1,-1))

        try:
            segments = torch.cat(segments,1)
        except:
            pdb.set_trace()
        
        segments = self.ln_1(self.segs_linear(segments)) 
        output = self.inter_attn(segments,anchor_attn,anchor_attn)

        assert output.size() == anchor_embeds.size() , 'Different size of seg_embedding'
        del segment_state,seqs
        torch.cuda.empty_cache()

        output = self.feed1(output)
        segments = torch.cat((segments,output),1)
        boundary.append(1)
        return output,boundary,segments
        
    def hier_layer_turn(self,input_i,index,inputs_em):
        input_ids = input_i.clone()[:,1:index]

        inputs_embeds = inputs_em.clone()[:,1:index,:]

        input_ids[input_ids.eq(50258)]=-100

        input_ids[input_ids.eq(50259)]=-100

        num_sen = sum(input_ids.eq(-100).squeeze())

        index = np.argwhere(input_ids.cpu()==-100)

        seqs = torch.zeros([index.size(1),2*inputs_embeds.size(-1)],device = inputs_embeds.device)

        #get utterance

        pos = 0

        for i,ind in enumerate(index[1]):

            if i == 0:

                continue

            sent = inputs_embeds[:,pos:ind]

            _, seq = self.word_level(sent)

            seqs[i-1] = seq.view(1,1,-1)

            pos = ind

        

        seqs[-1] = self.word_level(inputs_embeds[:,pos:])[1].view(1,1,-1)

        seqs = seqs.unsqueeze(0)

        seqs = self.seqs_linear(seqs)

        boundary = None

        return seqs,boundary,seqs
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)
        
        
    def forward(self,input_ids,token_type_ids,labels = None,output_embeds = False,from_prior = False):
        
        token_embeds = self.transformer.transformer.wte(input_ids)
        type_embeds = self.transformer.transformer.wte(token_type_ids)
        inputs_embeds = token_embeds + type_embeds #1,seq_len,768
        segments = None
        # pdb.set_trace()
        if self.with_hier:
            if labels is not None:
                last_type = token_type_ids[0][-1]
                index = np.argwhere(input_ids.cpu() == last_type.cpu())[1][-1]
                if last_type == 50258:
                    anchor_type = 50259
                else:
                    anchor_type = 50258
                    
            elif from_prior:
                # pdb.set_trace()
                index = None
                anchor_type = token_type_ids[0][-1].cpu()
            # pdb.set_trace()
            index_anchor = np.argwhere(input_ids.cpu() == anchor_type)[1][-1]
            if self.hier_type == 'sep_seg':
                latent_embeds,boundary,segments = self.hie_layer_sep(input_ids ,index, inputs_embeds)
            elif self.hier_type == 'all_seg':
                latent_embeds ,boundary,segments= self.hie_layer(input_ids ,index, inputs_embeds)
            elif self.hier_type == 'self_seg':
                latent_embeds ,boundary,segments= self.hie_layer_selfseg(input_ids ,index, inputs_embeds)
            elif self.hier_type == 'jra_seg':
                latent_embeds,boundary,segments= self.hie_layer_JRA(input_ids ,index,index_anchor, inputs_embeds)
            elif self.hier_type in ['block_seg_inter','block_seg','block_seg_aggregate']:
                # pdb.set_trace()
                latent_embeds ,boundary,segments= self.hie_layer_blockseg(input_ids ,index, inputs_embeds)
            elif self.hier_type == 'turn':
                latent_embeds,boundary ,segments = self.hier_layer_turn(input_ids ,index, inputs_embeds)
                
            # inputs_embeds = inputs_embeds + hie_embeds
            if self.vae_fusion == 'com_latent_seg':
                hier_embeds = latent_embeds
            
        # pdb.set_trace()
        if self.with_latent:

            if labels is not None:

                #full label
                # pdb.set_trace()
                last_type = token_type_ids[0][-1]
                index = np.argwhere(input_ids.cpu() == last_type.cpu())[1][-1]
                x_embeds = inputs_embeds[:,:index,:]
            elif from_prior:
                # pdb.set_trace()
                x_embeds = inputs_embeds
                
            #extract latent encoding
            if self.vae_type == 'avg_attn':
                prior_embed = self.multi_vae_prior(x_embeds,x_embeds,x_embeds)
                prior_embed = self.feed1(self.latent_norm1(x_embeds + prior_embed))
                prior_embed = self.multi_vae_prior_inter(x_embeds, prior_embed,prior_embed)
                # prior_embed = self.multi_vae_prior_inter(prior_embed,x_embeds,x_embeds)
                prior_embed,_ = self.prior_atten(prior_embed)
                
                if not from_prior:
                    post_embed = self.multi_vae_post(inputs_embeds,inputs_embeds,inputs_embeds)
                    post_embed = self.feed2(self.latent_norm2(inputs_embeds + post_embed))
                    post_embed = self.multi_vae_post_inter(x_embeds, post_embed,post_embed)
                    # post_embed = self.multi_vae_post_inter(post_embed,x_embeds, x_embeds)
                    post_embed,_ = self.post_atten(post_embed) #1,768
                    
            elif self.vae_type == 'gpt2model':
                
                prior_embed = self.gpt2_vae_prior.transformer.h[-1](hidden_states = x_embeds)[0]
                prior_embed,_ = self.prior_atten(prior_embed)
                if not from_prior:
                    post_embed = self.gpt2_vae_post.transformer.h[-1](hidden_states = inputs_embeds)[0]
                    post_embed,_ = self.post_atten(post_embed)
                    

            # pdb.set_trace()
            #sample z
            if not from_prior:

                prior_embed = self.prior_linear2(torch.tanh(self.prior_linear1(prior_embed)))
                post_embed = self.post_linear2(torch.tanh(self.post_linear1(post_embed)))            
    
                prior_mean, prior_logvar = prior_embed.chunk(2,-1)
                posterior_mean, posterior_logvar = post_embed.chunk(2, -1)
                
                kld = self.kl_loss(posterior_mean, posterior_logvar,prior_mean, prior_logvar)
                latent_embeds = self.reparameterize(posterior_mean, posterior_logvar).unsqueeze(1)
            else:
                prior_embed = self.prior_linear2(torch.tanh(self.prior_linear1(prior_embed)))
                prior_mean, prior_logvar = prior_embed.chunk(2,-1)   
                kld = 0
                latent_embeds = self.reparameterize(prior_mean, prior_logvar).unsqueeze(1)
            
        else:
            kld = 0
            
        if not self.with_hier:
            segments = None
        
        if not self.with_latent:
            latents = None
        else:
            latents = latent_embeds
        # print(type(latent_embeds))
        # print(latent_embeds)
        
        latent_embeds = self.latent_proj(latent_embeds)
        latent_embeds = self.latent_norm3(latent_embeds)
        if self.vae_fusion == 'proj':
            inputs_embeds = inputs_embeds + latent_embeds
            outputs = self.transformer(
                labels = labels,
                inputs_embeds = inputs_embeds
                )
        elif self.vae_fusion == 'concat':
            # pdb.set_trace()
            
            inputs_embeds = torch.concat([latent_embeds,inputs_embeds],1)
            if labels is not None:
                first = torch.LongTensor([-100]).unsqueeze(0).to(inputs_embeds.device)
                labels = torch.concat([first,labels],1)
            outputs = self.transformer(
                labels = labels,
                inputs_embeds = inputs_embeds
                )
        elif self.vae_fusion == 'layer_concat':
            # pdb.set_trace()
            present = ()
            latent_embeds = latent_embeds.split(self.embed_dim,-1) #n,seq,768
            for i in range(len(latent_embeds)):
                key_z,value_z = self.c_z(latent_embeds[i]).split(self.embed_dim,-1) #1,1,768
                key_z = self._split_heads(key_z, self.num_heads, self.head_dim) #1,head,1,headdim
                value_z = self._split_heads(value_z, self.num_heads, self.head_dim) #1,head,1,headdim
                present = present + ((key_z,value_z),) 
            # pdb.set_trace()
            
            outputs = self.transformer(
                # input_ids = input_ids,
                past_key_values = present,
                labels = labels,
                inputs_embeds = inputs_embeds
                )
            # print(present)
            return outputs,kld,present,inputs_embeds,boundary,latents,segments
        elif self.vae_fusion == 'encoder_h_concat':
            # pdb.set_trace()
            outputs = self.transformer(
                labels = labels,
                inputs_embeds = inputs_embeds,
                encoder_hidden_states = latent_embeds,
                # is_cross_attention = True
                )
            # pdb.set_trace()
        elif self.vae_fusion == 'com_latent_seg':
            present = ()
            latents = latent_embeds
            latent_embeds = latent_embeds.split(self.embed_dim,-1) #n,seq,768
            for i in range(len(latent_embeds)):
                key_z,value_z = self.c_z(latent_embeds[i]).split(self.embed_dim,-1) #1,1,768
                key_z = self._split_heads(key_z, self.num_heads, self.head_dim) #1,head,1,headdim
                value_z = self._split_heads(value_z, self.num_heads, self.head_dim) #1,head,1,headdim
                present = present + ((key_z,value_z),) 
            # pdb.set_trace()
            outputs = self.transformer(
                # input_ids = input_ids,
                past_key_values = present,
                labels = labels,
                inputs_embeds = inputs_embeds,
                encoder_hidden_states = hier_embeds
                )
            
            return outputs,kld,present,hier_embeds,inputs_embeds,boundary,latents,segments
            


        return outputs,kld,latent_embeds,inputs_embeds,boundary,latents,segments
         