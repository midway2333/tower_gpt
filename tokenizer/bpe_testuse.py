import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('work\\tokenizer\\spm_dict.model')   # type: ignore

padding_id = sp.pad_id()
print(padding_id)
