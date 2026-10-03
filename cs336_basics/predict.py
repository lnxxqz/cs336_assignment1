import transformer_lm
import decoding
import tokenizer
import torch

datapath       = "D:/code/cs336/cs336_assignment1/TinyStoriesV2-GPT4-valid.txt"
PATH           = "D:/code/cs336/cs336_assignment1/train_model/lr_1e-3/"

valid_datapath = "D:/code/cs336/cs336_assignment1/TinyStoriesV2-GPT4-valid.txt"
valid_PATH     = PATH + "valid/"
valid_tokenpath= valid_PATH + "tokens.npy"

rulepath       = PATH + "rule"
tokenpath      = PATH + "tokens.npy"
modelpath      = PATH + "model"
rulepath       = PATH + "rule"
tokenpath      = PATH + "tokens.npy"
logpath        = PATH + "log.txt"

vocab_size     = 10000
special_tokens = ['<|endoftext|>']
context_length   = 256
d_model        = 512
d_ff           = 1344
num_layers     = 4
num_heads      = 16
rope_theta     = 10000
lr             = 1e-3
weight_decay   = 0.1
betas          = (0.9, 0.95)
eps            = 1e-8
batch_size     = 50
max_norm       = 1.0

epoch          = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
tokeniz = tokenizer.tokenizer.from_files(
            rulepath + "_vocab.pkl",
            rulepath + "_merges.pkl",
            special_tokens=special_tokens,
        )
model = transformer_lm.transformer(
    vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device
)
obj = torch.load(modelpath + "99", map_location=device)
model.load_state_dict(obj["model"])
model = model.to(device)

ini = input()
out = decoding.decoding(model, ini, tokeniz, context_length, 128)
print(tokeniz.decode(out))