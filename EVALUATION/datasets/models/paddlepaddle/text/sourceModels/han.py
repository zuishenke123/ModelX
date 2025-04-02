import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class AttentionLayer(nn.Layer):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(hidden_dim, hidden_dim)
        context_vector_np = np.random.randn(hidden_dim)
        self.context_vector = paddle.to_tensor(context_vector_np)

    def forward(self, x):
        u = paddle.tanh(self.attention_fc(x))  # Shape: [batch_size, seq_len, hidden_dim]
        alpha = F.softmax(paddle.matmul(u, self.context_vector), axis=1)  # Shape: [batch_size, seq_len]
        tmp = paddle.unsqueeze(alpha, -1)
        attended_output = paddle.sum(x * tmp, axis=1)  # Shape: [batch_size, hidden_dim]
        return attended_output, alpha


class HAN(nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, word_dropout=0.5, sentence_dropout=0.5):
        super(HAN, self).__init__()
        # Word-level embedding and LSTM
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_lstm = nn.LSTM(embedding_dim, hidden_dim, direction='bidirect')
        self.word_attention = AttentionLayer(hidden_dim * 2)
        self.word_dropout = nn.Dropout(word_dropout)

        # Sentence-level LSTM and attention
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, direction='bidirect')
        self.sentence_attention = AttentionLayer(hidden_dim * 2)
        self.sentence_dropout = nn.Dropout(sentence_dropout)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: [batch_size, num_sentences, num_words]
        batch_size, num_sentences, num_words = x.shape

        # Reshape to process words within each sentence
        x = x.reshape([-1, num_words])  # Shape: [batch_size * num_sentences, num_words]
        embedded_words = self.embedding(x)  # Shape: [batch_size * num_sentences, num_words, embedding_dim]
        word_lstm_output, _ = self.word_lstm(embedded_words)  # Shape: [batch_size * num_sentences, num_words, hidden_dim*2]
        word_attended, _ = self.word_attention(word_lstm_output)  # Shape: [batch_size * num_sentences, hidden_dim*2]
        word_output = self.word_dropout(word_attended)  # Apply dropout at word level

        # Reshape to process sentences within the document
        sentence_input = word_output.reshape([batch_size, num_sentences, -1])  # Shape: [batch_size, num_sentences, hidden_dim*2]
        sentence_lstm_output, _ = self.sentence_lstm(sentence_input)  # Shape: [batch_size, num_sentences, hidden_dim*2]
        sentence_attended, _ = self.sentence_attention(sentence_lstm_output)  # Shape: [batch_size, hidden_dim*2]
        sentence_output = self.sentence_dropout(sentence_attended)  # Apply dropout at sentence level

        # Classification
        logits = self.fc(sentence_output)  # Shape: [batch_size, output_dim]
        return logits