file = open("src/utils/test/data/raw/data.txt", "r")

text = file.read()

file.close()
text = text.lower()
print(text)
word = ""
words = [ ]
for ch in text:
    # print(ch)
    if ch != " " and ch != "\n":
        word =word+ch
    else:
        words.append(word)
        word = ""

print(words)
vocab={}
next_id = 0
for t in words:
    if t not in vocab:
        
        vocab[t] = next_id
        next_id += 1

print(vocab)
with open("src/utils/test/data/vocab/encode_vocab.txt", "w") as f:
    for word, id in vocab.items():
        f.write(f"{word}\t{id}\n")


with open("src/utils/test/data/vocab/decode_vocab.txt", "w") as f:
    for word, id in vocab.items():
        f.write(f"{id}\t{word}\n")






