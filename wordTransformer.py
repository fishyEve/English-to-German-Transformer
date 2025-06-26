import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np

# Eve Collier
# CPSC 372 - Spring 2025
# Project 3: Transformers 

# Globals:
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# WordDataset
# Class basically handles data preparation for our transformer
# Parameters:
# src_file: Path to file with English words
# trg_file: Path to file with German words
# Returns: (src_tensor, trg_tensor) for each word pair - also builds src and trg vocabularies from characters
class WordDataset(Dataset):
    def __init__(self, src_file, trg_file):
        # Open English and German word files
        with open(src_file) as f:
            self.src_words = [line.strip() for line in f] # English words
        
        with open(trg_file) as f:
            self.trg_words = [line.strip() for line in f] # German words

        # Each English word needs to have a corresponding German word
        assert len(self.src_words) == len(self.trg_words), "Files must have same number of lines"
        
        # Build vocabulary by each character
        self.src_vocab = self.build_vocab(self.src_words) # English
        self.trg_vocab = self.build_vocab(self.trg_words) # German
        
    def build_vocab(self, words):
        # Initialize with special tokens
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}  # Padding, start, end, unknown
        counter = Counter() #  Count character frequencies
        for char in words: # For every character in the word...
            counter.update(list(char)) # Add it to our list

        # Add characters to vocab in order of frequency
        for char, _ in counter.most_common():
            if char not in vocab:
                vocab[char] = len(vocab) # Assign next available index
                
        return vocab
    
    def __len__(self):
        return len(self.src_words) # Total number of word pairs
    
    def __getitem__(self, idx):
        # Add start/end tokens and convert words to character IDs
        src_word = ['<sos>'] + list(self.src_words[idx]) + ['<eos>'] # English words w/ token
        trg_word = ['<sos>'] + list(self.trg_words[idx]) + ['<eos>'] # German words w/ token
        
        # Convert characters to their corresponding IDs
        src_ids = [self.src_vocab.get(c, self.src_vocab['<unk>']) for c in src_word]
        trg_ids = [self.trg_vocab.get(c, self.trg_vocab['<unk>']) for c in trg_word]
        
        return torch.tensor(src_ids), torch.tensor(trg_ids) # Return as PyTorch tensors

# collate_fn() - overridden function 
# Function to pad and combine multiple samples into a single batch
# Parameters:
# batch: A list of (src_tensor, trg_tensor) pairs
# Returns: src_batch - padded source tensor of shape (batch_size, src_seq_len) and
# trg_batch (Tensor) - padded target tensor of shape (batch_size, trg_seq_len)
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch) # Separate source and target
    
    # Pad sequences to same length within batch
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    
    return src_batch, trg_batch

# WordTransformer
# Our model definition for our English to German language transformer
# Parameters:
# src_vocab_size: Size of the English (source) vocabulary
# trg_vocab_size: Size of the German (target) vocabulary
# Returns: predictions of shape (batch_size, trg_seq_len, trg_vocab_size)
class WordTransformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.embed_size = 128 # Dimension of character embeddings
        self.src_embed = nn.Embedding(src_vocab_size, 128, padding_idx=0)
        self.trg_embed = nn.Embedding(trg_vocab_size, 128, padding_idx=0)

        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # Use structure of transformer from https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1/ 
        self.transformer = nn.Transformer(
            d_model=128,  # Dimension of model - each char in the word is represented by a size 128 vector
            nhead=8,      # Number of attention heads - 8 different ways to 'figure out' how to piece the characters together
            num_encoder_layers=3, # Depth of encoder - English is read 3 times (once by each layer)
            #num_encoder_layers=2,
            num_decoder_layers=3, # Depth of decoder - German output is written 3 times (once by each layer)
            #num_decoder_layers=2,
            dim_feedforward=512,  # Size of the feedforward network in each layer. 
            #dim_feedforward=256,
            dropout=0.1,     # I have this because almost every example I saw has it- this prevents overfitting which supposedly helps 
                             # to make our transformer 'more robust'- update I learned in AI II that this is gonna kill 10% of neurons to make 
                             # the remaining ones better
            batch_first=True # Input format (batch, seq, feature) - allows us to process our words in batches (and one word per batch)
        )
        # Final layer to predict output characters
        self.fc_out = nn.Linear(128, trg_vocab_size)
        #self.src_mask = None
        #self.trg_mask = None      
    # forward()
    # Defines the forward pass of our Transformer model
    # Parameters:
    # src: Padded source batch (batch_size, src_seq_len)
    # trg: Padded target batch up to current decoding step (batch_size, trg_seq_len)
    # tgt_mask: Mask to prevent decoder from looking ahead
    # 
    # Returns: output - logits for each token (batch_size, trg_seq_len, trg_vocab_size)
    def forward(self, src, trg, tgt_mask=None):
        # Convert character IDs to embeddings
        src_emb = self.src_embed(src)  # English embeddings
        trg_emb = self.trg_embed(trg)  # German embeddings
        
        # Create padding masks
        #src_padding_mask = (src == 0)
        #trg_padding_mask = (trg == 0)
        
        # Transformer processing
        output = self.transformer(
            src_emb, # Source sequence
            trg_emb, # Target sequence
            tgt_mask=tgt_mask, # DON'T look ahead
            src_key_padding_mask=(src == 0),   # Ignore padding - ENCODER
            tgt_key_padding_mask=(trg == 0),   # Ignore padding - DECODER
            memory_key_padding_mask=(src == 0) # Ignore padding - DECODER'S ATTENTION ON ENCODER'S OUTPUT
        )
        # Final char predictions
        output = self.fc_out(output)
        return output

# train_model()- Training method
# Helper function to handle training the Transformer model
# Parameters: None 
# Returns:
# model: The trained transformer model
# src_vocab: Vocabulary used for source (English) characters
# trg_vocab: Vocabulary used for target (German) characters
def train_model():
    # Load and prep our dataset
    dataset = WordDataset('englishWords.txt', 'germanWords.txt') # Load the files with the words
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = WordTransformer(
        src_vocab_size=len(dataset.src_vocab),
        trg_vocab_size=len(dataset.trg_vocab)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training 
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for src, trg in train_loader:
            # Move data to GPU if I even have one
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            # Shift by one so <SOS> can predict token at index 1 - https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1/ 
            trg_input = trg[:, :-1]  # Remove last token for input
            trg_output = trg[:, 1:]  # Remove first token for target

            # Create target mask for training 
            trg_mask = nn.Transformer.generate_square_subsequent_mask(
                trg_input.size(1)).to(DEVICE)
            
            # Reset gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(src, trg_input, tgt_mask=trg_mask) 
            
            # Reshape for loss calculation
            #output = output.transpose(0, 1).transpose(1, 2)
            #loss = criterion(output, trg_output)
            loss = criterion(output.reshape(-1, output.size(-1)), trg_output.reshape(-1)) # Calc loss between predictions and targets

            # Backprop
            loss.backward() # Compute gradients
            optimizer.step()# Update Adam

            # Track loss
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}') # Show progress
    return model, dataset.src_vocab, dataset.trg_vocab

# translate_word()
# Translates a single English word into German using inferencing
# Parameters:
# model: Trained Transformer model
# word: Input English word to be translated
# src_vocab: Source vocabulary mapping characters to IDs
# trg_vocab: Target vocabulary mapping characters to IDs
# max_len: Max allowed length of translation output 
# Returns: translation - the final, generated translated word in German  
def translate_word(model, word, src_vocab, trg_vocab, max_len=20):
    model.eval() # Inference mode

    
    # Convert source word to tensor with proper start/end tokens
    src_tokens = ['<sos>'] + list(word.lower()) + ['<eos>']
    src_ids = [src_vocab.get(c, src_vocab['<unk>']) for c in src_tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Initialize target with <sos> (start) token
    trg_ids = [trg_vocab['<sos>']]

    # Create mapping from IDs to characters
    idx_to_char = {v: k for k, v in trg_vocab.items()}

    # Generate chars one at a time
    #while True:
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_ids).unsqueeze(0).to(DEVICE)
        
        # Create target mask
        trg_mask = nn.Transformer.generate_square_subsequent_mask(
            len(trg_ids)
        ).to(DEVICE)
        
        with torch.no_grad():
            output = model(src_tensor, trg_tensor, tgt_mask=trg_mask)
            # Get the last predicted token
            logits = output[0, -1, :]
            # Convert token to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from that probability distribution
            pred_token = torch.multinomial(probs, num_samples=1).item()
            
            trg_ids.append(pred_token) # Add to generated sequence
            #pred_token = output.argmax(2)[-1, 0].item()
            #trg_ids.append(pred_token)
            
            # Debug print
            #print(f"Step {i}: Generated char '{idx_to_char.get(pred_token, '?')}' (prob: {probs[pred_token]:.2f})")

            # We are done if end token generated
            if pred_token == trg_vocab['<eos>']:
                break
    # Convert IDs to characters
    translation = ''.join([idx_to_char.get(i, '') for i in trg_ids[1:-1]]) # Remove tokens 
    return translation
    
    #for _ in range(max_len):
        #trg_tensor = torch.LongTensor(trg_ids).unsqueeze(0).to(DEVICE)
        
        # Create target mask
        #trg_mask = nn.Transformer.generate_square_subsequent_mask(
            #len(trg_ids)
        #).to(DEVICE)
        
        #with torch.no_grad():
            #output = model(
                #rc_tensor,
                #trg_tensor,
                #tgt_mask=trg_mask
            #)
        
        # Get the most likely next token
        #pred_token = output.argmax(2)[-1, 0].item()
        #trg_ids.append(pred_token)
        
        # Stop if we predict <eos>
        #if pred_token == trg_vocab['<eos>']:
            #break
    
    # Convert token IDs to characters
    #trg_tokens = []
    #for token_id in trg_ids[1:]:  # Skip <sos>
        #if token_id == trg_vocab['<eos>']:
            #break
        #trg_tokens.append(trg_vocab.get(token_id, ''))
    
    #return ''.join(trg_tokens)
    
    # Convert source word to tensor
    #src_tokens = ['<sos>'] + list(word) + ['<eos>']
    #src_ids = [src_vocab.get(c, src_vocab['<unk>']) for c in src_tokens]
    #src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(DEVICE)  # Add batch dim
    
    # Initialize target with <sos>
    #trg_ids = [trg_vocab['<sos>']]
    
    #for _ in range(max_len):
        #trg_tensor = torch.LongTensor(trg_ids).unsqueeze(0).to(DEVICE)
        
        # Create target mask
        #trg_mask = nn.Transformer.generate_square_subsequent_mask(
            #len(trg_ids)
        #).to(DEVICE)
        
        #with torch.no_grad():
            #output = model(src_tensor, trg_tensor, tgt_mask=trg_mask)
        
        # Get the most likely next token
        #pred_token = output.argmax(2)[-1, 0].item()
        #trg_ids.append(pred_token)
        
        # Stop if we predict <eos>
        #if pred_token == trg_vocab['<eos>']:
            #break
    
    # Convert ids to characters
    #trg_tokens = [trg_vocab.get(i, '') for i in trg_ids[1:-1]]  # Remove <sos> and <eos>
    #return ''.join(trg_tokens)
