import torch
import numpy as np


class AttentionLayer(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = torch.nn.Linear(in_features=hidden_dim,
            out_features=hidden_dim)
        context_vector_np = np.random.randn(hidden_dim)
        self.context_vector = torch.tensor(data=context_vector_np)

    def forward(self, x):
        u = torch.tanh(input=self.attention_fc(x))
        alpha = torch.nn.functional.softmax(input=torch.matmul(input=u,
            other=self.context_vector), dim=1)
        tmp = torch.unsqueeze(input=alpha, dim=-1)
        attended_output = torch.sum(input=x * tmp, axis=1)
        return attended_output, alpha


class HAN(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
        word_dropout=0.5, sentence_dropout=0.5):
        super(HAN, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self.word_lstm = torch.nn.LSTM(input_size=hidden_dim)
        self.word_attention = AttentionLayer(hidden_dim * 2)
        self.word_dropout = torch.nn.Dropout(p=word_dropout)
        self.sentence_lstm = torch.nn.LSTM(input_size=hidden_dim)
        self.sentence_attention = AttentionLayer(hidden_dim * 2)
        self.sentence_dropout = torch.nn.Dropout(p=sentence_dropout)
        self.fc = torch.nn.Linear(in_features=hidden_dim * 2, out_features=
            output_dim)

    def forward(self, x):
        batch_size, num_sentences, num_words = x.shape
        x = x.reshape([-1, num_words])
        embedded_words = self.embedding(x)
        word_lstm_output, _ = self.word_lstm(embedded_words)
        word_attended, _ = self.word_attention(word_lstm_output)
        word_output = self.word_dropout(word_attended)
        sentence_input = word_output.reshape([batch_size, num_sentences, -1])
        sentence_lstm_output, _ = self.sentence_lstm(sentence_input)
        sentence_attended, _ = self.sentence_attention(sentence_lstm_output)
        sentence_output = self.sentence_dropout(sentence_attended)
        logits = self.fc(sentence_output)
        return logits
