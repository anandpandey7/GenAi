import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization, Dropout
import numpy as np

# # load the harry potter book as the dataset -> url = https://www.kaggle.com/datasets/shubhammindola/harry-potter-books
# def load_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read()
#     return text

# file_path = "hp_1.txt"
# text = load_data(file_path).lower()

def load_data(path):
    # Use utf-8 encoding which handles a wider range of characters
    # Add error handling strategy (errors='replace' will replace problematic characters)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = f.read()
    return data    

path = "Book1.txt"  
text = load_data(path).lower()
text

# Tokenize the text
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
input_sequences = []
tokens = tokenizer.texts_to_sequences([text])[0]
seq_length = 50

# First seq_length tokens (input): Used for training the model.
# Last token (target): Used as the label the model tries to predict.
# so total of (50 + 1) in one input_sequence index

for i in range(seq_length, len(tokens)):
    input_sequences.append(tokens[i - seq_length:i + 1])

#print(input_sequences[0])

# Pad sequences and split inputs/targets
# after this X will have inputs and y will have label for those inputs
input_sequences = np.array(pad_sequences(input_sequences, maxlen=seq_length + 1, padding='pre'))
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# One-hot encode the labels , note- there are other ways for
# encoding like pre-trained word2vec encoding and so on
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

#Transformer Model

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads

        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        # Split the embedding into multiple heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(self.projection_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, value)

    def call(self, inputs):
        query, key, value = inputs # unpacking the inputs shape - (batch_size, seq_length, embed_dim)
        batch_size = tf.shape(query)[0]

        query = self.split_heads(self.query_dense(query))
        key = self.split_heads(self.key_dense(key))
        value = self.split_heads(self.value_dense(value))

        # Scaled dot-product attention
        attention_output = self.attention(query, key, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concatenated = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concatenated)
    
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim , activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = self.position_embedding(positions)
        return self.token_embedding(inputs) + positions
    

# Model Parameters
embed_dim = 128  # Embedding size
num_heads = 4    # Number of attention heads
ff_dim = 512     # Feed-forward layer size
maxlen = seq_length  # max it is 50 defined above

# below total words = 6662 (see above - basically all tokens in the text)

# Build the model
inputs = tf.keras.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, total_words, embed_dim)
x = embedding_layer(inputs)
print(x.shape)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x, training=True)
print(x.shape)
x = x[:, -1, :]
print(x.shape)
x = Dense(total_words, activation="softmax")(x)
print(x.shape)
model = tf.keras.Model(inputs=inputs, outputs=x)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()