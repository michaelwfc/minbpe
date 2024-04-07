import os
import sentencepiece as spm

# write a toy.txt file with some random text
with open("toy.txt", "w", encoding="utf-8") as f:
    f.write(
        "SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")


class SentencePieceTokenizer():
    def __init__(self) -> None:
        self.options = dict(
            # input spec
            input="toy.txt",
            input_format="text",
            # output spec
            model_prefix="tok400",  # output filename prefix
            # algorithm spec
            # BPE alg
            model_type="bpe",
            vocab_size=400,
            # normalization
            normalization_rule_name="identity",  # ew, turn off normalization
            remove_extra_whitespaces=False,
            input_sentence_size=200000000,  # max number of training sentences
            max_sentence_length=4192,  # max number of bytes per sentence
            seed_sentencepiece_size=1000000,
            shuffle_input_sentence=True,
            # rare word treatment
            character_coverage=0.99995,
            byte_fallback=True,
            # merge rules
            split_digits=True,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            max_sentencepiece_length=16,
            add_dummy_prefix=True,
            allow_whitespace_only_pieces=True,
            # special tokens
            unk_id=0,  # the UNK token MUST exist
            bos_id=1,  # the others are optional, set to -1 to turn off
            eos_id=2,
            pad_id=-1,
            # systems
            num_threads=os.cpu_count(),  # use ~all system resources
        )

    def train(self):
        spm.SentencePieceTrainer.train(**self.options)

    def _load_model(self):
        sp = spm.SentencePieceProcessor()
        sp.load('tok400.model')
        vocab = [[sp.id_to_piece(idx), idx]
                 for idx in range(sp.get_piece_size())]
        print(f"vocab=\n{vocab}")
        return sp

    def encode(self, text):
        sp = self._load_model()
        ids = sp.encode(text)
        print(ids)
        return ids


if __name__ == "__main__":
    sp_tokenizer = SentencePieceTokenizer()
    sp_tokenizer.train()
    text =  "hello 안녕하세요"
    token_ids = sp_tokenizer.encode(text=text)
