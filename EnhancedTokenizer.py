class EnhancedTokenizer:
    """Simple word-based tokenizer for vision-language tasks"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
    
    def build_vocab(self, texts: list[str]):
        """Build vocabulary from a list of texts"""
        # Count word frequencies
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort words by frequency and take top vocab_size - reserved tokens
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - len(self.word2idx)]:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
    
    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        """Encode text into input_ids and attention_mask"""
        words = text.lower().split()
        input_ids = [self.word2idx['<BOS>']]
        for word in words[:max_length - 2]:  # Reserve space for BOS and EOS
            input_ids.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        input_ids.append(self.word2idx['<EOS>'])
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(self.word2idx['<PAD>'])
            attention_mask.append(0)
        
        return input_ids, attention_mask