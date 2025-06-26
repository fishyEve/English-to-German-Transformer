from wordTransformer import train_model, translate_word

# Eve Collier
# CPSC 372 - Spring 2025
# Project 3: Transformers 

def main():
    print("Training English to German word TRANSFORMER...")
    model, src_vocab, trg_vocab = train_model()
    
    # Test translations
    #test_words = ["cat", "boat", "key", "leg", "queen", "stupid"]
    #print("\nTest Translations:")
    #for word in test_words:
        #translation = translate_word(model, word, src_vocab, trg_vocab)
        #print(f"{word} -> {translation}")
    # Load English words from file
    with open("englishWords.txt") as f:
        english_words = [line.strip().lower() for line in f if line.strip()]

    print("\nTranslations from englishWords.txt:")
    for word in english_words:
        translation = translate_word(model, word, src_vocab, trg_vocab)
        print(f"{word} -> {translation}")

    #while True:
        #word = input("\nEnter an English word to be translated into German (or 'quit' to exit): ").strip().lower()
        #if word == 'quit':
            #break
        #translation = translate_word(model, word, src_vocab, trg_vocab)
        #print(f"Translation: {translation}")

if __name__ == "__main__":
    main()