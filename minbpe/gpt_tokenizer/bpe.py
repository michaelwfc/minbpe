"""Byte Pair encoding
The original algorithm operates by iteratively replacing the most common contiguous sequences of characters in a target text with unused 'placeholder' bytes. The iteration ends when no sequences can be found, leaving the target text effectively compressed.   
Decompression can be performed by reversing this process, querying known placeholder terms against their corresponding denoted sequence, using a lookup table. In the original paper, this lookup table is encoded and stored alongside the compressed text.

"""
import os
import json
from typing import List, ByteString, Dict, Union, Text, Tuple
from gpt_tokenizer.corpus import CORPUS_TEXT


class BPE():

    def __init__(self, vocab_size=276, max_integer_of_one_byte=256) -> None:
        self.vocab_size = vocab_size
        self.max_integer_of_one_byte = max_integer_of_one_byte
        self.num_merges = self.vocab_size - self.max_integer_of_one_byte
        self.bytepair_to_placehold = {}

        self.vacob = {idx: bytes([idx]) for idx in range(self.max_integer_of_one_byte)}

    def build_vocab(self, corpus=List[Text]) -> Dict[int, ByteString]:
        vocab = self.vacob
        bytepair_to_placehold = self._get_bytepair_to_placehold(corpus=corpus)
        for (p0, p1), idx in bytepair_to_placehold.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.bytepair_to_placehold = bytepair_to_placehold
        self.vocab = vocab

        return vocab

    def _get_bytepair_to_placehold(self, corpus=List[Text]) -> Dict[Tuple[int], int]:  # byte_ls_corpus: List[List[ByteString]]):
        placeholder_init_no = self.max_integer_of_one_byte

        string = " ".join(corpus)
        # byte_ls = b" ".join(byte_ls_corpus)

        ids = self._to_token_ids(string)

        bytepair_to_placehold = {}
        # while len(ids) > 2:
        for num_merge in range(self.num_merges):
            counts = self._get_stats(ids)
            # get the most frequect byte pair
            # sorted_counts = sorted([(k, v) for k, v in counts.items()], key=lambda x: x[1], reverse=True)
            # top_pair, top_pair_frequency = sorted_counts[0]
            top_pair = max(counts, key=counts.get)
            # if top_pair_frequency > 1:
            # update bytepair_to_placehold
            placeholde_increet_no = len(bytepair_to_placehold)
            placehold_no = placeholder_init_no + placeholde_increet_no

            # replace the top_pair to placehold
            print(f"{num_merge}th merging {top_pair} to a new token {placehold_no}")
            ids = self._merge(ids, top_pair, placehold_no)

            bytepair_to_placehold.update({top_pair: placehold_no})

        return bytepair_to_placehold

    def encode(self, text):
        """encode by first find the minimum pair then merge"""

        ids = self._to_token_ids(text)

        while len(ids) > 1:
            counts = self._get_stats(ids)
            pair = min(counts, key=lambda p: self.bytepair_to_placehold.get(p, float("inf")))
            if pair not in self.bytepair_to_placehold:
                break

            placehold = self.bytepair_to_placehold.get(pair)
            ids = self._merge(ids, pair, placehold)
        return ids

    def _encode_by_bytepair_to_placehold(self, text: Text):  # byte_ls: List[ByteString]):
        """encode the text by iterate the pair and merge the original ids"""
        ids = self._to_token_ids(text)
        for pair, placehold in self.bytepair_to_placehold.items():
            ids = self._merge(ids, pair, placehold)
        return ids

    def decode(self, ids: List[int]) -> Text:
        token_bytes = b"".join(self.vacob[idx] for idx in ids)
        text = token_bytes.decode("utf-8", errors='replace')
        return text

    def _decode_by_bytepair_to_placehold(self, ids: List[int]) -> Text:
        """deencode the text by bytepair_to_placehold and decompress to original ids"""
        for pair, placehold in list(self.bytepair_to_placehold.items())[::-1]:
            ids = self._decompress(ids, pair, placehold)
        # from ids to text
        text = self._ids_to_text(ids)
        return text

    def _to_token_ids(self, text: Text) -> List[int]:
        """encode the string to bytes, iterate each byte, mapping one byte(8 bits) to the corresponding integer(from 0 to 256)"""
        text_bytes = text.encode("utf-8")
        int_ls = list(map(int, text_bytes))
        return int_ls

    def _ids_to_text(self, token_ids: List[int]):
        """conert the token_id(from 0 to 256) to one byte, then join the bytes and decode the bytes to string"""
        bytes_ = b"".join([token_id.to_bytes(1, byteorder='big') for token_id in token_ids])
        # bytes_ = b"".join([hex(id)  for id in ids])
        text = bytes_.decode("utf-8")
        return text

    def _get_stats(self, byte_ls: List[Union[ByteString, int]]):
        counts = {}
        for index in range(len(byte_ls)-1):
            pair = tuple(byte_ls[index:index+2])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids: List[int], pair: [int, int], placehold: int):
        index = 0
        new_ids = []
        while index < len(ids):
            if index < len(ids)-1 and ids[index] == pair[0] and ids[index+1] == pair[1]:
                new_ids.append(placehold)
                index += 2
            else:
                new_ids.append(ids[index])
                index += 1
        return new_ids

    def _decompress(self, ids: List[int], pair: [int, int], placehold: int) -> List[int]:
        index = 0
        new_ids = []
        for index in range(len(ids)):
            if ids[index] == placehold:
                new_ids.extend(pair)
            else:
                new_ids.append(ids[index])
        return new_ids


class GPT2Tokenizer():
    """
    gpt-2 :https://github.com/openai/gpt-2/blob/master/src/encoder.py

    !wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
    !wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json
    """

    def __init__(self) -> None:
        self.data_dir = "./data/gpt2-tokenizer-data"
        self.vocab_bpe_file = f"{self.data_dir}/vocab.bpe"
        self.encoder_file = f"{self.data_dir}/encoder.json"

    def _load_data(self,):
        with open(self.encoder_file, 'r') as f:
            encoder = json.load(f)  # <--- ~equivalent to our "vocab"

        with open(self.vocab_bpe_file, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        # ^---- ~equivalent to our "merges"
        return encoder, bpe_merges


if __name__ == "__main__":
    string_01 = "aaabdaaabac"
    string_02 = "abc"
    # corpus = [string_01, string_02]
    # input_text = string_01
    # vocab_size = 258

    corpus = [CORPUS_TEXT]
    input_text = CORPUS_TEXT
    vocab_size = 276

    bpe = BPE(vocab_size=vocab_size)

    vocab = bpe.build_vocab(corpus)
    print(f"vocab={vocab}")
    vocab[0].decode()

    # counts = bpe._get_stats(byte_ls)
    # print(f"counts={counts}")
    encoded_tokens = bpe.encode(text=input_text)
    ori_tokens = bpe._to_token_ids(text=input_text)
    ratio = len(ori_tokens) / len(encoded_tokens)
    print(f"ori_tokens.len:    {ori_tokens.__len__()}\nencoded_tokens.len:{encoded_tokens.__len__()}\nratio:{ratio}")

    decoded_text = bpe.decode(encoded_tokens)
    assert input_text == decoded_text
    print("Done")
