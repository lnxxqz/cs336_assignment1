import os
import regex as re
from collections import defaultdict
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import pickle

class tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text)-> list[int]:
        mp = {}
        for b,id in self.vocab.items():
            mp[id] = b

        special_tokens_dic = {}

        for i in self.special_tokens or []:
            u = i.encode('utf-8')
            special_tokens_dic[u] = mp[u]

        tokens = self.special_tokens or []
        tokens = sorted(tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(t) for t in tokens) + ")"
        if self.special_tokens is None or self.special_tokens is []:
            chunks = [text]
        else :chunks = re.split(pattern, text)

        ls = []
        for chunk in chunks:
            if special_tokens_dic.get(chunk.encode('utf-8'),None) is not None:
                ls.append([special_tokens_dic[chunk.encode('utf-8')]])
            else:
                for m in re.finditer(PAT,chunk):
                    t = m.group().encode("utf-8")
                    u = [mp[i.to_bytes()] for i in t]
                    ls.append(u)

        LEN = len(self.merges)
        cnt = 0
        for merge in self.merges:
            cnt+=1
            print(f'merge: {cnt}/{LEN}')
            L = len(ls)
            id = mp[merge[0]+merge[1]]
            for j in range(L):
                i = 0
                newls = []
                le = len(ls[j])
                while i < le:
                    if i<le-1 and (self.vocab[ls[j][i]], self.vocab[ls[j][i+1]]) == merge:
                        newls.append(id)
                        i+=2
                    else :
                        newls.append(ls[j][i])
                        i+=1
                ls[j] = newls
        lis = []
        for i in ls:
            for j in i:
                lis.append(j)
        return lis

    def encode_iterable(self, iterable):
        for piece in iterable:
            yield from self.encode(piece)

    def decode(self, ids)-> str:
        txt = b''.join([self.vocab[id] for id in ids])
        return txt.decode('utf-8',errors='replace')
        


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    return tokenizer(vocab, merges, special_tokens)