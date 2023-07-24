import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model"  # The llama sentencepiece tokenizer model

class Tokenizer:
    def __init__(self):
        model_path = TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        # Ensure vocab_size and piece_size are the same
        assert self.n_words == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encode the input string 's' using SentencePiece tokenizer.

        Parameters:
            s (str): The input string to be tokenized.
            bos (bool): Whether to add the BOS (Beginning of Sentence) token.
            eos (bool): Whether to add the EOS (End of Sentence) token.

        Returns:
            List[int]: A list of token IDs corresponding to the input string.
        """
        # Ensure 's' is a string
        assert isinstance(s, str)

        # Encode the input string
        t = self.sp_model.encode(s)

        # Add BOS and EOS tokens if required
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int]) -> str:
        """
        Decode a list of token IDs into the original string.

        Parameters:
            t (List[int]): A list of token IDs.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)
