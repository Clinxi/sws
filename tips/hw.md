# lab1

|group9|姓名|
|---|---|
|Team Member 1|贾世安(JIA SHIAN)|
|Team Member 2|陶毅诚(TAO YICHENG)|

## question1
1. what an embedding layer does?
+ An embedding layer is a neural network layer that maps each integer to a high-dimensional vector. The purpose of the embedding layer is to learn a representation of the words that captures their semantic meaning. This is important because it allows the LSTM to better understand the relationships between words in the text.

2. why we cannot just feed the integers from the tokenizer direct to the LSTM.
+ **Lack of semantic meaning**: Integers produced by a tokenizer do not inherently carry any semantic meaning. They are simply numerical representations assigned to different words based on their position in the vocabulary. LSTM or recurrent layers are designed to capture sequential dependencies and patterns in data, but they are not equipped to interpret the semantic relationships between words.
+ **High-dimensional input**: Tokenizers typically assign a unique integer to each word in the vocabulary, resulting in a large range of integers. Directly feeding these high-dimensional integer sequences to an LSTM would require the network to handle a high number of unique inputs, which can be computationally expensive and lead to overfitting due to the large number of parameters.
+ **Loss of word relationships**: When tokenizing text, words are often split into their individual units (tokens) based on punctuation and other rules. By feeding only the integers to an LSTM, we lose the information about the relationships between the tokens within a word. This can lead to a loss of important semantic information and hinder the model's ability to understand the context and meaning of the text.

## question2
1. Explain why we don't need to chop up our tokens into groups of 5 tokens to predict the 6th for transformers, but must do so for LSTMs.

+ The difference in handling token sequences between transformers and LSTMs is primarily due to their architectural design and the way they process information.

+ Transformers are based on the self-attention mechanism and operate on the entire sequence of tokens at once. They have a parallel processing capability that allows them to capture dependencies and relationships between tokens across long distances. This means that transformers can consider the context of any token within the entire sequence when making predictions. Therefore, there is no need to chop up tokens into groups to predict the next token in a transformer model.

+ On the other hand, LSTMs are sequential models that process token sequences one element at a time in a sequential manner. They maintain an internal state that carries information from previous tokens to inform the prediction of the next token. LSTMs have a fixed memory size, and long sequences can cause memory limitations and computational inefficiency.

+ To overcome these limitations, when using LSTMs or other recurrent neural networks (RNNs), token sequences are often divided into smaller chunks or groups. By breaking the sequence into chunks, the memory constraints are reduced, and the model can process the tokens in smaller, manageable parts. Each chunk is processed sequentially, and the hidden state from one chunk is passed to the next as a form of context.

## question3
1. In our network we have used a one-hot approach; our network will have over 50,000 outputs, where one of them will be set to "1" and the rest to "0" when training. Why can't we just have one output, where the target value is the index of the next word?

+ Dimensionality: By having a single output, you are essentially trying to represent over 50,000 different classes with a single value, which would be challenging for a neural network to learn. Each word would need to be represented by a unique value within a limited range, making it difficult for the network to capture the nuances and relationships between different words.
+ Loss Function: In order to train a neural network, you need to define a loss function that quantifies the difference between the predicted output and the true target. With a single output, you would need to devise a loss function that can handle such a high-dimensional target space. This would likely be complex and less intuitive compared to using a one-hot encoding, which allows for a straightforward comparison between the predicted and true values.
+ Model Interpretability: Using a one-hot encoding provides a clear interpretation of the network's output. Each output neuron corresponds to a specific word, and the neuron with the highest value indicates the predicted word. This makes it easier to understand and analyze the model's predictions. With a single output, it would be more challenging to interpret and analyze the network's behavior.
+ Generalization: One-hot encoding allows for better generalization. Each output neuron is responsible for predicting a specific word, enabling the model to learn distinct patterns and relationships between words. This allows the model to generalize its knowledge to unseen data. A single output may struggle to capture the complexity of word relationships and may result in poorer generalization performance.
+ Overall, using a one-hot approach with multiple outputs is a more effective and commonly used technique for word prediction tasks. It provides a more expressive representation, facilitates training, and allows for better interpretation and generalization capabilities compared to using a single output with the target value as the index of the next word.

2. Why do we use softmax and categorical cross entropy for the activation function and loss function?
+ for the activation function: The softmax activation function is used to convert the output of a neural network into a probability distribution over multiple classes. It ensures that the predicted probabilities sum up to 1 and are in the range [0, 1]. in this lab, we need to use a probability model to choose which output with the maximum probability based on the current series of words, so softmax function is the best choice for us to get the most suitable word. Another advatage for softmax function is that it is differential and it is easier for model to optimize the parameter together with cross-entrop loss function through gradient-based optimization algorithms like stochastic gradient descent (SGD).
+ for the loss function: the categorical cross-entropy measures the dissimilarity between the predicted probabilities (y_pred) and the true labels (y_true). It penalizes large errors and encourages the model to assign high probabilities to the correct classes which is suitable here. And categorical cross-entropy is a differentiable function, enabling backpropagation and gradient-based optimization to update the model's parameters. Another important thing is that the categorical cross-entropy loss aligns with the probabilistic interpretation of softmax. It measures the dissimilarity between the predicted probabilities and the true class probabilities, driving the model to improve its predictions towards the true distribution.(cross-entropy loss function and softmax activation function have a good cooperation)

# lab2

|group9|姓名|
|---|---|
|Team Member 1|贾世安(JIA SHIAN)|
|Team Member 2|陶毅诚(TAO YICHENG)|

## question1
Explain why transformers train(speculate i think) using entire sentences instead of short "lookback" sentences like LSTM.
(in fact, we use "batch_size" for training except "lookback")
+ Transformers rely on self-attention mechanisms, where each word in a sentence attends to all other words within that sentence. This mechanism allows the model to capture the contextual relationships between all words simultaneously, regardless of their positional distance. In contrast, LSTM models process sequences sequentially, which restricts their ability to capture long-range dependencies effectively.
+ Transformers can process sentences in parallel, making them highly efficient for training on modern hardware accelerators like GPUs and TPUs. Each word in a sentence can be processed independently, allowing for parallel matrix multiplications and speeding up training. This parallelism is not possible with LSTM models, as they rely on sequential processing and can only process one word at a time.
+ Transformers use positional encodings to incorporate positional information into the model. This enables the model to learn the relative positions of words within the sentence. As a result, the model can handle sentences of varying lengths without losing positional information. LSTM models, on the other hand, do not inherently capture positional information and require additional mechanisms like padding or truncation to handle variable-length sequences.
+ Transformers are particularly effective at capturing long-term dependencies in text. The self-attention mechanism allows information to propagate directly across all positions in the input, enabling the model to consider distant context when making predictions. LSTM models, although capable of capturing some long-term dependencies, can struggle with maintaining and propagating information over longer sequences due to the vanishing or exploding gradient problem.

## question2
Explain why the sentences used to train the transformers must be of fixed length.
+ **Batch Processing**: Transformers are commonly trained on mini-batches of sentences for efficient parallel processing. To form a batch, all input sequences within the batch must have the same length. By fixing the length of sentences, it allows the model to process multiple sentences simultaneously in a batch, which can significantly speed up training by leveraging parallel computing.
+ **Memory Efficiency**: Transformers require a fixed-size attention matrix for each token in the sequence. The attention matrix captures the relationships between tokens. By using fixed-length sentences, the model can allocate memory efficiently for the attention matrices, regardless of the actual sentence length. This enables more efficient memory usage during training and inference.
+ **Optimization Stability**: During training, the model's parameters are updated based on gradients computed through backpropagation. When using variable-length sequences, the gradients computed for different sentence lengths can vary, leading to unstable optimization. By fixing the sentence length, the gradients remain consistent across examples, which promotes stable and consistent updates to the model's parameters.
+ **Hardware and Software Constraints**: Many deep learning frameworks and hardware accelerators have limitations on variable-length sequences. Using fixed-length sentences simplifies the implementation and utilization of these frameworks and accelerators, making training and inference more efficient and compatible with the available infrastructure.

## question3
Using the Hugging Face website or otherwise, explain the parameters in AutoConfig.from_pretrained that we have used.
1. **model_name**: This parameter specifies the name or identifier of the pre-trained model. It is used to retrieve the corresponding configuration file from the Hugging Face model hub.

2. **vocab_size**: The vocab_size parameter is used to specify the size of the vocabulary for the model. It should match the number of unique tokens in the tokenizer associated with the model.

3. **n_ctx**: The n_ctx parameter sets the maximum length of the input sequences for the model. It determines the size of the attention matrices and positional encodings.

4. **bos_token_id**: The bos_token_id parameter specifies the token ID for the beginning of sentence (BOS) token. It represents the start of a sequence and is used for tasks like text generation.

5. **eos_token_id**: The eos_token_id parameter specifies the token ID for the end of sentence (EOS) token. It represents the end of a sequence and is used for tasks like text generation

## question4
Compare the texts generated from the transformer that was trained from scratch versus the transformer that used the pretrained GPT2 weights. Do you see a difference in quality, e.g. fewer "non-English" words?

1. the texts generated from the transformer that was trained used the pretrained GPT2 has fewer "non-English" words for example, the thransformer that was trained from scratch will outputs something like "know\'t", ""."" , which is "non-English", but the pretrained is better.
2. pretrained transformer has less grammar errors




