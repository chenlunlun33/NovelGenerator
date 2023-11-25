## Pin Post
This project has only trained and tested from the "original" folder. However, due to insufficient personal GPU computing resources, the training of the novel generator could not be completed. As a result, there may be some errors in the code, especially in the construction and usage of the tgt_mask. Another aspect that can be modified is the embedding method. Consider using BPE or SentencePiece, which may yield better performance. The contents in the "model" folder below are those that I have reorganized and partially modified from the "original" folder.

# NovelGenerator
transformer type novel generator with stochastic decoder layer

## 1. Word Embedding
The word embedding part utilizes Chinese characters-to-vector conversion, applying nn.Embedding for implementation.
Consider using BPE or SentencePiece to tokenize by breaking down tokens into smaller subwords. This approach can enhance the model's understanding of the meanings of individual characters and words, thereby improving overall comprehension.

## 2. Layers
The encoder layer here doesn't have anything special. The significant modification lies in the decoder layer, where a stochastic decoder layer has been added. This involves applying small differences to the matrix by assigning values from a standard normal distribution. The aim is to introduce variation in the later outputs, with the hope of generating more diverse content in the novel.
