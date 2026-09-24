import cs336_basics.train_bpe
import cs336_basics.tokenizer
import time
import psutil
import pickle



if __name__ == '__main__':
    path = 'TinyStoriesV2-GPT4-train.txt'
    t0 = time.perf_counter()
    vocab,merges = cs336_basics.train_bpe.run_train_bpe(path, vocab_size=10000, special_tokens=["<|endoftext|>"])
    t1 = time.perf_counter()
    print(t1 - t0, "seconds")
    print(f'{psutil.Process().memory_info().rss/(1024)/1024}MB')

    with open("tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    
    print(max(vocab.values(), key=len).decode("utf-8", errors="replace"))