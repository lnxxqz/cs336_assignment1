import os
import regex as re
from collections import defaultdict
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = [f.read()]

    for special in special_tokens:
        chunks = [piece for s in chunks for piece in s.split(special)]

    pre_token = {}
    for chunk in chunks:
        for m in re.finditer(PAT,chunk):
            t = m.group().encode("utf-8")
            pre_token[t] = pre_token.get(t, 0) + 1
    print('pretoken ok')
    dic = {}
    cnt = 0
    for i in range(256):
        dic[cnt]=i.to_bytes()
        cnt+=1

    for tk in special_tokens:
        dic[cnt] = tk.encode("utf-8")
        cnt+=1

    chunks_int = [([s for s in chunk],count,{}) for chunk,count in pre_token.items()]
    index = defaultdict(set)
    pair_cnt = defaultdict(lambda: (0, (0, 0)))
    for i in range(len(chunks_int)):
        for j in range(len(chunks_int[i][0])-1):
            pair = (chunks_int[i][0][j],chunks_int[i][0][j+1])
            chunks_int[i][2][pair] = chunks_int[i][2].get(pair,0)+1
        for a,b in chunks_int[i][2].items():
            index[a].add(i)
            pair_cnt[a] = (pair_cnt[a][0]+b*chunks_int[i][1],(dic[a[0]],dic[a[1]]))

    def _meg(chunk, count, d, pair, ct, id):
        L = len(chunk)
        d.pop(pair,None)
        i = 0
        new_chunk = []
        las = chunk[0]
        while i+1 < L:
            pr = (chunk[i], chunk[i+1])
            if pr == pair:
                if i-1 >=0 :
                    new_pair = (las,ct)
                    old_pair = (las,chunk[i])
                    d[old_pair] = d.get(old_pair,0)-1
                    d[new_pair] = d.get(new_pair,0)+1
                    pair_cnt[new_pair] = (pair_cnt[new_pair][0]+count,(dic[las],dic[ct]))
                    index[new_pair].add(id)
                    pair_cnt[old_pair] = (pair_cnt[old_pair][0]-count, pair_cnt[old_pair][1])
                    if pair_cnt[old_pair][0] <= 0:
                        pair_cnt.pop(old_pair)
                        index[old_pair].remove(id)
                if i+2 <L :
                    new_pair = (ct,chunk[i+2])
                    old_pair = (chunk[i+1],chunk[i+2])
                    d.pop(old_pair,None)
                    d[new_pair] = d.get(new_pair,0)+1
                    pair_cnt[new_pair] = (pair_cnt[new_pair][0]+count,(dic[ct],dic[chunk[i+2]]))
                    index[new_pair].add(id)
                    pair_cnt[old_pair] = (pair_cnt[old_pair][0]-count, pair_cnt[old_pair][1])
                    if pair_cnt[old_pair][0] <= 0:
                        pair_cnt.pop(old_pair)
                        index[old_pair].remove(id)
                i=i+2
                new_chunk.append(ct)
                las = ct
            else: 
                new_chunk.append(chunk[i])
                las = chunk[i]
                i=i+1
        if i<L:new_chunk.append(chunk[L-1])
        return (new_chunk, count, d)
                                
    lst = []
    while cnt < vocab_size:
        print(cnt)
        if not pair_cnt:break
        merge = max(pair_cnt ,key=pair_cnt.get)
        # print(merge)
        dic[cnt] = dic[merge[0]] + dic[merge[1]]
        cnt+=1

        lst.append((dic[merge[0]], dic[merge[1]]))

        t = list(index[merge])
        for i in t:
            chunks_int[i] = _meg(chunks_int[i][0], chunks_int[i][1], chunks_int[i][2], merge, cnt-1, i)
        pair_cnt.pop(merge)

    return (dic, lst)
    