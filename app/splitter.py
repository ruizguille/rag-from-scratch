# Inspired by LlamaIndex's Sentence Splitter
# https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/sentence.py
import nltk
import tiktoken
from functools import partial

tiktoken_tokenizer = tiktoken.get_encoding('cl100k_base')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def token_size(text):
    return len(tiktoken_tokenizer.encode(text))

def split_by_separator(text, sep):
    splits = text.split(sep)
    res = [s + sep for s in splits[:-1]]
    if splits[-1]:
        res.append(splits[-1])
    return res

def split_sentences(text):
    spans = [s[0] for s in sentence_tokenizer.span_tokenize(text)] + [len(text)]
    return [text[spans[i]:spans[i+1]] for i in range(len(spans) - 1)]


class TextSplitter:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.splitters = [
            partial(split_by_separator, sep='\n\n'),
            partial(split_by_separator, sep='\n'),
            split_sentences,
            partial(split_by_separator, sep=' ')
        ]
    
    def _split_recursive(self, text, level=0):
        if token_size(text) <= self.chunk_size or level == len(self.splitters):
            return [text]
        
        splits = []
        for s in self.splitters[level](text):
            if token_size(s) <= self.chunk_size:
                splits.append(s)
            else:
                splits.extend(self._split_recursive(s, level + 1))
        return splits

    def _merge_splits(self, splits):
        chunks = []
        current_chunk = ''

        for s in splits:
            if current_chunk and (token_size(current_chunk + s) > self.chunk_size):
                trimmed_chunk = current_chunk.strip()
                if trimmed_chunk:
                    chunks.append(trimmed_chunk)
                current_chunk = ''
            current_chunk += s
        
        trimmed_chunk = current_chunk.strip()
        if trimmed_chunk:
            chunks.append(trimmed_chunk)
        return chunks
    
    def split(self, text):
        splits = self._split_recursive(text)
        chunks = self._merge_splits(splits)
        return chunks
    
    def __call__(self, text):
        return self.split(text)
