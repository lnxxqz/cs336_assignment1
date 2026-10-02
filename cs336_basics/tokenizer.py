import pickle
import heapq
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # vocab: id -> bytes
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 预计算：bytes -> id
        self.bytes_to_id = {b: i for i, b in vocab.items()}

        # 预计算：单字节 id 表，用于快速把 UTF-8 bytes 转成初始 id
        self.byte_to_id = [self.bytes_to_id[bytes([i])] for i in range(256)]

        # 预计算特殊 token 的 id
        self.special_token_to_id = {
            tok: self.bytes_to_id[tok.encode("utf-8")]
            for tok in self.special_tokens
        }

        # 特殊 token 分割正则，长 token 优先
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
            self.special_split_re = re.compile(pattern)
        else:
            self.special_split_re = None

        # 预编译 PAT
        self.pat = re.compile(PAT)

        # 预计算 merge: (left_id, right_id) -> (rank, new_id)
        self.merge_ranks = {}
        for rank, (a, b) in enumerate(merges):
            a_id = self.bytes_to_id[a]
            b_id = self.bytes_to_id[b]
            new_id = self.bytes_to_id[a + b]
            self.merge_ranks[(a_id, b_id)] = (rank, new_id)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, ids):
        """对单个 PAT 片段做 BPE 合并，按 merge rank 从小到大合并。"""
        n = len(ids)
        if n < 2:
            return ids

        # 双向链表 + 最小堆
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[n - 1] = -1
        alive = [True] * n
        heap = []

        for i in range(n - 1):
            info = self.merge_ranks.get((ids[i], ids[i + 1]))
            if info is not None:
                rank, new_id = info
                heapq.heappush(heap, (rank, i, ids[i], ids[i + 1], new_id))

        while heap:
            rank, i, left_id, right_id, new_id = heapq.heappop(heap)

            if not alive[i]:
                continue

            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if ids[i] != left_id or ids[j] != right_id:
                continue

            # 合并 i 和 j
            ids[i] = new_id
            alive[j] = False

            nj = nxt[j]
            nxt[i] = nj
            if nj != -1:
                prev[nj] = i

            # 新产生的相邻 pair 入堆
            pi = prev[i]
            if pi != -1:
                info = self.merge_ranks.get((ids[pi], ids[i]))
                if info is not None:
                    heapq.heappush(heap, (info[0], pi, ids[pi], ids[i], info[1]))

            if nj != -1:
                info = self.merge_ranks.get((ids[i], ids[nj]))
                if info is not None:
                    heapq.heappush(heap, (info[0], i, ids[i], ids[nj], info[1]))

        # 收集链表结果
        result = []
        i = 0
        while i != -1:
            result.append(ids[i])
            i = nxt[i]
        return result

    def encode(self, text) -> list[int]:
        out = []

        chunks = self.special_split_re.split(text) if self.special_split_re else [text]

        for chunk in chunks:
            if not chunk:
                continue

            # 特殊 token 直接输出 id
            if chunk in self.special_token_to_id:
                out.append(self.special_token_to_id[chunk])
                continue

            # 普通文本按 PAT 切分，再分别做 BPE
            for m in self.pat.finditer(chunk):
                bs = m.group().encode("utf-8")
                ids = [self.byte_to_id[b] for b in bs]
                out.extend(self._apply_merges(ids))

        return out

    def encode_iterable(self, iterable):
        for piece in iterable:
            yield from self.encode(piece)

    def decode(self, ids) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    return tokenizer(vocab, merges, special_tokens)