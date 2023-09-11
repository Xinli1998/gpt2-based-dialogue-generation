# gpt2-based-dialogue-generation

This is my master's thesis about GPT2-based Dialogue Generation.

## Setup

Start by preparing the required Python environment and packages and getting into the repository
```bash
git clone git@github.com:Xinli1998/gpt2-based-dialogue-generation.git
cd gpt2-based-dialogue-generation
pip install -r requirements.txt
```
Then make sure you have GPU in your environment. This project works better in CUDA.

## Training

To train various models, use corresponding parameters and codes. There are two types of models, Hierarchical and VAE, which are specified in parameters [with_hier & hier_type] and [with_latent & vae_type] respectively. Here are two examples of these two types.

First, the VAE model is incorporated through Past Units in GPT2.
```bash
python3 main_model.py --mode train \
--with_latent --vae_type 'avg_attn' \
--model_fusion 'layer_concat'
--test_prefix 'valid_daily_dialog' \
--valid_prefix 'valid_daily_dialog' \
--train_prefix 'train_daily_dialog' \
--max_turns 35
```

Second, the Hierarchical model (Self-seg) is incorporated through encoder_hidden_state in GPT2.
```bash
python3 main_model.py --mode train \
--with_hier --hier_type 'self_seg' 
--model_fusion 'encoder_h_concat' \
--test_prefix 'valid_daily_dialog' \
--valid_prefix 'valid_daily_dialog' \
--train_prefix 'train_daily_dialog' \
--ckpt_name 'BEST_PC_allturns_avg_attn_encoder_h_concat_self_seg_best_ckpt_epoch=15_valid_loss=1.7599'\
--max_turns 35
```

There are different options for those parameters specified in []:
hier_type [token_seg / block_token_seg / self_seg / last_turn_seg / block_seg_aggregate / turn]
model_fusion [proj / concat / layer_concat / encoder_h_concat / com_latent_seg] where the last one refers to the combination of VAE and Hierarchical models.
train_prefix [train_daily_dialog / train_persona_chat] 

## Length Statistics

To analyze how models perform in different numbers of context turns, the trained model could be tested in the following way:

```bash
python3 main_model.py --mode test_length \
--with_hier --hier_type 'self_seg' 
--model_fusion 'encoder_h_concat' \
--special_words 'PC_allturns' \
--test_prefix 'valid_persona_chat' \
--valid_prefix 'valid_persona_chat' \
--train_prefix 'train_persona_chat' \
--ckpt_name 'XXXtrained_model.pt'\
--max_turns 35
```

## KL plot

To evaluate VAE models, KL divergence is plotted.
```bash
python3 KL.py
```
You should modify the name of each model's document 'kl_data.npy' in KL.py



