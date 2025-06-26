# ENGLISH TO GERMAN TRANSLATION USING A TRANSFORMER
Eve Collier


## FILE MANIFEST
**wordTransformer.py - This file is the implementation of a Transformer for English to German language translation. Here is a summary of the key components:**

1. WordDataset class
    - Initalization loads the English (src_file) and German (trg_file) words from text files
    - build_vocab(self, words):
      - Builds vocabularies for English and German based on character frequencies
      - Returns the vocabulary dictionary (vocab) that maps each unique character (from the input words) to a unique index. This dictionary also 
      includes tokens for padding (to avoid shape mismatch error), start-of-sequence (beginning of word), end-of-sequence (end of word), and
      unknown characters (used for characters that are not found in the vocabulary).
    - __len__(self):
      - Overridden Dataset function to determine how many items are in our WordDataset
      - Returns the total number of word pairs in the dataset 
    - __getitem__(self, idx)
      - Overridden Dataset function to retrieve a sample (source and target word pair) from our WordDataset.
      - Converts words to sequences of token IDs, adding special tokens (<sos>, <eos>, <unk>). This is used to is used to process our source (English) 
      word and our target (German) word for use in training. This is important because the transformer needs to know where the sequence starts and ends
      in order to piece together each word. If a character is not found in the vocabulary, it defaults to the ID for the <unk> (unknown) token.
      - Returns the two sequences (English and German words) as PyTorch tensors- torch.tensor(src_ids), torch.tensor(trg_ids). These tensors are the 
      integer representations of the characters in each word, where each element corresponds to a character's ID in the vocabulary.

2. collate_fn Helper Function:
    - Overridden DataLoader function to achieve custom batching for our WordDataset.
    - This helper function exists to combine multiple samples into a SINGLE batch for training with our PyTorch Transformer. This is needed when our 
    words differ in length...which will apply to most English/German word pairs. Doing this project, I  learned that PyTorch's DataLoader will automatically
    stack samples into a batch (single tensor)- which is cool when the words are the same length, but from experience, I know it won't work if the words 
    differ in length.
    src_batch, trg_batch = zip(*batch) allows us to unzip the batch into two lists- one for the source sequences (src_batch) 
    and one for the target sequences (trg_batch). We use pad_sequence to pad each list so all sequences (words) are the same length (using <pad>).
    - Returns the padded source and target batches as two tensors: src_batch and trg_batch. Both src_batch and trg_batch will be tensors where 
    the shorter sequences are padded to match the length of the longest sequence in their batches.


3. WordTransformer class
   - Defines the architecture of our ransformer model for English to German translation
     - Initalization establishes the model with embedding layers for both the source and target vocabularies. The Transformer architecture is also defined
     here:
        - Dimension of Transformer (d_model): 128 dimensional embedding layer (each word is turned into a vector of numbers- each number represents each character)
        - Attention Heads (nhead): 8 attention heads- 8 different ways for our transformer to 'look at' different parts of the input (word) when processing- these 
        different views are all used in parallel
        - Encoder (num_encoder_layers): 3 stacked layers - each refines the transformer's understanding of the input (word). My thinking was:
        First layer -> letters, Second layer -> patterns between letters ('qu' for example), Third layer -> (hopefully) understand the full word 
        - Decoder (num_decoder_layers): 3 stacked layers- each attempts to generate the corresponding German word, one character at a time. This is done using the 
        encoder's output (the processed, understood English word) and what it's generated this far. The way I thought about the layers was:
        First layer -> rough draft of next character, Second layer -> refine next character based on context of previously generated characters, Third layer -> finalize
        the predicted next character
            - ALSO, the decoder uses a mask so we don't look ahead at future characters, because then this would be pointless
        - dropout: 0.1, during training, 0.1 of the nodes in the network are dropped, which prevents Overfitting and helps generalization. The transformer is able to 
        learn patterns and not just copy the training set.
        - batch_first: True to override PyTorch's default (sequence,batch,feature) format of reading input to be, instead, (batch, sequence, embedding). This is how the 
        DataLoader is established.
        - and finally, self.fc_out: the final layer that turns the output embedding into character predictions of the corresponding German word to the English input. The 
        transformer outputs a vector of size 128 for each step, so this layer converts each of those vectors into a vector of size trg_vocab_size, which represents the probabilities for each possible output character.
     - forward ()
        - This function runs a forward pass through the Transformer model given a English word (src) and the corresponding generated German word so far (trg). We also have
        a target mask to prevent looking ahead at the next characters. We turn each character into a 128 dimensional vector (learned embeddings) such that it can interface with the Transformer. The output is the final result from the transformer- the predictions for the next German character. The encoder first takes the
        English word embeddings (src_emb) and learns a deep representation of the word. The decoder takes the German prefix embeddings (src_emb) and looks at whats been generated so far as well as the encoder's output regarding the English word, then it predicts the next character in the word in German. The attention heads in the
        encoder and decoder help the transformer do this. We also have to ignore <pad> tokens so they don't affect the transformer's learning. Each output character goes 
        through a Linear layer to turn it into a 128D vector of probability scores for each possible character. This is used for training (to compare to the actual German word to compute loss) and the final character generation. 

4. train_model() Helper Function
    - This helper function exists to train the WordTransformer model to translate given English words to German. It loads the dataset of English/German word pairs 
    (englishWords.txt & germanWords.txt) and preps tbe DataLoader and WordTransformer model. We also establish a vocabulary for English (src_vocab) and German words
    (trg_vocab). We train the transformer on word pairs and use masking to prevent it from seeing future characters in a word. We use the cross-entropy loss to predict character positions in a word.  We use PyTorch's Adam optimizer for step and gradient updates. 
    During training, we loop over batches of training data- not all at once. Because of that, the loss is calculated per batch- how wrong the transformer was per batch, We backpropogate to compute our gradients and use optimizer.step() to then apply the updates.
    - Returns...
      - model (nn.Module): the trained WordTransformer model
      - src_vocab (dict): Vocab mapping for English characters
      - trg_vocab(dict): Vocab mapping for German characters

5. translate_word(...) Helper Function
    - This helper function exists to take a trained WordTransformer model and a single English word and generate the corresponding German word, one character at a time.
    - Parameters:
      - model: the trained WordTransformer model
      - word: the English word (as a string) to be translated to German
      - src_vocab: the character vocabulary for English
      - trg_vocab: the character vocabulary for German
      - max_len: max number of characters allowed to generate in the translation (20)
    - We first set the model to inference mode before beginning converting the source word to a PyTorch tensor- first adding the start of sequence <sos> and end of sequence <eos> tokens to the word. Then, we map the English characters to its corresponding ID from the src_vocab and finally convert the list into a tensor, adding  a batch dimension so that we only have one word translated per batch. 
    At this point, its time to start the output sequence. The first character is always the <sos> token, so we add that before creating a dictonary to map token ID from the trg_vocab back to characters for the final German word. Then we loop by max_len to generate up to 20 characters. In the loop, we conver the German output to a tensor and then create a mask to ensure the model and only see previously generated characters and not future characters in the word it's trying to generate. We do a 
    call to the transformer to generate predictions and then convert that to a probability distribution over the trg_vocab. We sample one character bnased on the probabilities and then append the predicted token to the output sequence. If an <eos> token is detected, we break out of the loop because we know that means we're at the end of a word. Once out of the loop, we convert the list of token IDs (minus <sos> and <eos>) back into characters- the final, translated German word. translate_word then returns that inferenced word. 
    Returns: translation - our final, translated, German word



**main.py - Main script for running English to German translation using a WordTransformer. Calls the 
appropriate functions to handle training the network, having the network translate the file of words, and then 
printing the results. Here is a summary of the key components:**

1. main()
   - Backbone of the program- calls all the appropriate functions and ultimately runs the entire program



**germanWords.txt & englishWords.txt**

Two text files containing corresponding English and German words. The data we will ultimately be training our network with.


## PROJECT DEPENDENCIES 
* numpy            - math
* torch            - PyTorch 
* englishWords.txt - dataset
* germanWords.txt  - dataset



## PROTOCOL
1. Data Prep
   - Call train_model() to create WordDataset to train our network.
2. Training
   - Initalization
     - Create WordDataset 
     - Set up DataLoader with a batch size of 32 and shuffle set to 'True' so the characters are in a different order
   - Actual training
     - 100 epochs
     - source (src) and target (trg) sequences are loaded from the dataset (our English and German word pairs)
     - trg_input = trg[:, :-1]: takes the target sequence and removes the last token so the model can predict the next token in the sequenc
     - trg_output = trg[:, 1:]: This takes the target sequence and removes the first token to use it as the ground truth for comparison 
     - trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_input.size(1)).to(DEVICE): This creates a mask for the target sequence to prevent the model from seeing characters ahead when generating each token
     - output = model(src, trg_input, tgt_mask=trg_mask): This performs the forward pass through the transformer. The model takes the source (English) sequence (src), the target (German) input sequence (trg_input), and the target mask (trg_mask). It gives us a prediction for the next character in the sequence.
     - The model’s output is reshaped and compared with the actual target output using loss = criterion(output.reshape(-1, output.size(-1)), trg_output.reshape(-1)). This calculates the cross-entropy loss between the predicted and actual token sequences, where the model is trained to minimize the difference between the predicted token IDs and the ground truth token IDs.
     - loss.backward(): Computes the gradients of the loss with respect to the model parameters. This step is necessary to update the model's weights.
     - optimizer.step(): Updates the model's parameters using the Adam optimizer based on the computed gradients. The optimizer adjusts the weights in the direction that minimizes the loss.
     - total_loss += loss.item(): We accumulate the loss for each batch to track the overall performance of the model during the training process.
     - At the end of each epoch, we print the average loss: print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}').
     - After training, the model and vocabularies (dataset.src_vocab, dataset.trg_vocab) are returned.
3. Word Translation
   - For each word:
     - Initialize the model in evaluation mode using model.eval() 
     - Convert source word to tokens:
        - Add a start-of-sequence token (<sos>) at the beginning and an end-of-sequence token (<eos>) at the end of the word.
        - Map each character of the word to its corresponding ID using the source vocabulary (src_vocab).
        - Convert the list of character IDs into a PyTorch tensor and add a batch dimension using unsqueeze(0) to prepare it for model input.
     - Initalize the target sequence: begin with the start-of-sequence token (<sos>) for the target sequence (trg_ids).
     - Generate the target sequence, token by token:
        - For each step, pass the source tensor (src_tensor) and the target tensor (trg_tensor (which has all previously generated tokens)) to the model.
        - Use the target mask to prevent the model from looking ahead during generation
     - Model prediction:
        - For the most recent token in the target sequence, extract the logits (character probabilities) from the model’s output.
        - Apply softmax to convert the logits into a probability distribution across the vocabulary.
        - Sample a token based on these probabilities (using torch.multinomial), and append the predicted token to the target sequence.
     - Break out of the loop after 20 iterations or if a <eos> token is read!!!
     - Convert token IDs to characters:
        - Once the generation process finishes, convert the predicted token IDs back into characters using the target vocabulary (trg_vocab).
        - Remove the start-of-sequence (<sos>) and end-of-sequence (<eos>) tokens from the generated sequence.
     - Return the final translation as a string (the German word and its corresponding English word).
4. Visualization
   - Print the results on the screen via:
   print(f"{word} -> {translation}"). Boom, we are done. 




## DEVELOPMENT PROCESS
The first step I took toward this project was refrencing Attention Is All You Need.

An issue I ran into was the outputs of my DataLoader (WordDataset) causing shape mismatch errors. It took me a couple of days to figure this out:
https://pytorch.org/docs/stable/data.html Here is a quote from the docs, " you run into a situation where the outputs of DataLoader have dimensions 
or type that is different from your expectation, you may want to check your collate_fn". Cool.

For my training method, I didn't originally have a max_len variable. I had a situation where I was falling into infinite loops because of missing <eos>
token (I only had <sos> at the beginning of each word). Before I fixed the tokens, I added a max_len. Then, later down the line, I came to the realization of adding the <eos> tokens to the words such that it'd be easier to tell when words actually end.

At one point I had a dimension mismatch errors between source and target embeddings- oops. At one point, the d_model and embedding sizes were different from one another, which caused problems. I made both of them of size 128 in the end.

In my forward() method in the WordTransformer Class, I did not ignore paddings. At first, I was getting (incorrect) translations with tokens in them, which is definetly not what we want. After looking around online I decided to ignore the padding by using padding masks:
src_key_padding_mask=(src == 0)
gt_key_padding_mask=(trg == 0)  
memory_key_padding_mask=(src == 0) 
This mask is passed into the Transformer layers so that the model can focus on actual, valid tokens and ignore the <pad> tokens.

During training at one point, my loss values were not decreasing. This is because the source/target sequences were not aligned- I needed to shift the target input and output by one token (trg_input = trg[:, :-1], trg_output = trg[:, 1:]). Once this was corrected, the loss value finally began to decrease between epochs.

My model definetly wasn't working at first. One thing I did during the process of trying to get it work was the fun game of changing values for different things over 
and over. I changed the number of attention heads the Transformer uses quite a bit, but in the end, decided on 8. I chose this after learning more
about attention heads and realizing the more heads there are, the more diverse patterns the transformer will be able to pick up on- https://medium.com/@weidagang/demystifying-transformers-multi-head-attention-43b3173de391 


## BUILD INSTRUCTION
Ensure you have wordTransformer.py, main.py, germanWords.txt, and englishWords.txt in a repository. On the command line, simply run:

**python3 main.py**

From there, you can watch the model go through all 100 of its epochs and watch as it's loss decreases as the epochs continue. Once training is done, the
screen will display all of the words in englishWords.txt, followed by the transformer's translated corresponding German word.

## KNOWN ISSUES
- Sometimes, if a word has two of the same character in a row, the transformer will generate an extra one. For example, in the output:
all -> allle
Allle has one extra 'l' and should be alle.
On the other hand, sometimes the transformer will only generate one of the two characters. For example, in the output:
stupid -> dum
Dum is missing a 'm' and should be dumm.
And other times, the transformer does it perfectly fine. For example, in the output:
sea -> meer
Meer is spelt as meer and is correct.


## WORKS CITED/UTILIZED RESOURCES
https://arxiv.org/pdf/1706.03762

https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853/ 

https://www.udacity.com/blog/2025/04/understanding-transformer-architecture-the-backbone-of-modern-ai.html 

https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html 

https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1/ 

https://sanjayasubedi.com.np/deeplearning/masking-in-attention/ 

https://iifx.dev/en/articles/355122507 

https://medium.com/@weidagang/demystifying-transformers-multi-head-attention-43b3173de391 

https://medium.com/harness-transformers/english-to-spanish-translation-with-transformer-452ad43d101f 

https://discuss.pytorch.org/t/how-to-use-collate-fn/27181/4

https://pytorch.org/docs/stable/data.html 

https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9/ 
