#I have added coments in python notebook, therefore not adding here, since logic is almost same
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
from datetime import datetime
import wandb
import pandas
import argparse
import requests, zipfile, io

url = "https://drive.google.com/u/0/uc?id=1uRKU4as2NlS9i8sdLRS1e326vQRdhvfw&export=download"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()
# print(device)
wandb.login(key="75723b6e8716d094c31d6b7e25cc865ac5907e1f")

csvFile = pandas.read_csv('aksharantar_sampled/hin/hin_train.csv', names = ['English', 'Hindi'])
train_input = csvFile['English']
train_output = csvFile['Hindi']
csvFile = pandas.read_csv('aksharantar_sampled/hin/hin_valid.csv', names = ['English', 'Hindi'])
valid_input = csvFile['English']
valid_output = csvFile['Hindi']
csvFile = pandas.read_csv('aksharantar_sampled/hin/hin_test.csv', names = ['English', 'Hindi'])
test_input = csvFile['English']
test_output = csvFile['Hindi']

SOW_token = 0
EOW_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.index2char = {0: "#", 1: "$"}
        self.n_chars = 2  # Count SOS and EOS

    def addAllWords(self, words):
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        for c in word:
            if c not in self.char2index:
                self.char2index[c] = self.n_chars
                self.index2char[self.n_chars] = c
                self.n_chars += 1
lang_input = Lang("English")
lang_input.addAllWords(train_input)
lang_output = Lang("Hindi")
lang_output.addAllWords(train_output)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p, batch_size, embedding_size, cell_type = "LSTM", bidirection = False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.bidirection = bidirection
        if(cell_type == "GRU"):
            self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout_p, bidirectional = bidirection)
        elif(cell_type == "LSTM"):
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout_p, bidirectional = bidirection)
        elif(cell_type == "RNN"):
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout = dropout_p, bidirectional = bidirection)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1,self.batch_size, self.embedding_size)
        output = self.dropout(embedded)
        if(self.cell_type == "GRU"):
            output, hidden = self.gru(output, hidden)
        elif(self.cell_type == "LSTM"):
            output, (hidden, cell) = self.lstm(output)
        elif(self.cell_type == "RNN"):
            output, hidden = self.rnn(output, hidden)
        if self.bidirection:
            hidden = hidden.reshape(2, hidden.size(0)//2, hidden.size(1), hidden.size(2))
            hidden = torch.add(hidden[0]*0.5, hidden[1]*0.5)
            if(self.cell_type == "LSTM"):
                cell = cell.reshape(2, cell.size(0)//2, cell.size(1), cell.size(2))
                cell = torch.add(cell[0]*0.5, cell[1]*0.5)
            output = output.permute(2, 1, 0)
            output = torch.split(output, output.shape[0]//2)
            
            output = torch.add(output[0].permute(2, 1, 0)*0.5, output[1].permute(2, 1, 0)*0.5)
        if self.cell_type == "LSTM":
            return output, hidden, cell
        else:
            return output, hidden

    def initHidden(self):
        if self.bidirection:
            return torch.zeros(2*self.num_layers, self.batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)
max_length = 0
for i in train_input:
    max_length = max(max_length, len(i))
for i in valid_input:
    max_length = max(max_length, len(i))
for i in test_input:
    max_length = max(max_length, len(i))
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p, batch_size, embedding_size, cell_type = "LSTM"):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.cell_type = cell_type
        #(hidden size +embedding) * max_length
        self.attn = nn.Linear(self.hidden_size + self.embedding_size, max_length+1)
        # hidden + embedding , hidden
        self.attn_combine = nn.Linear(self.hidden_size + self.embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if(self.cell_type == "GRU"):
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout = self.dropout_p)
        elif(self.cell_type == "LSTM"):
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout = self.dropout_p)
        elif(self.cell_type == "RNN"):
            self.rnn = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, dropout = dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        # input -> 1*batch_size
        input = input.unsqueeze(0)
        output = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
        output = self.dropout(output)
        #batch_size * seq_length
        if self.cell_type == "LSTM":
            attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0][0]), 1)), dim=1)
        else:
            attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute(1, 0, 2))
        attn_applied = attn_applied.squeeze(1)
        output = torch.cat((output[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        if(self.cell_type == "GRU"):
            output, hidden = self.gru(output, hidden)
        elif(self.cell_type == "LSTM"):
            output, (hidden, cell) = self.lstm(output, (hidden[0], hidden[1]))
        elif(self.cell_type == "RNN"):
            output, hidden = self.rnn(output, hidden)
        if self.cell_type == "LSTM":
            return self.out(output[0]), hidden, cell, attn_weights
        return self.out(output[0]), hidden, attn_weights
    
def indexesFromWord(lang, word):
    return [lang.char2index[c] for c in word]

def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOW_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromWord(lang_input, pair[0])
    target_tensor = tensorFromWord(lang_output, pair[1])
    return (input_tensor, target_tensor)
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, cell_type):
    teacher_forcing_ratio = 0.5
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    if cell_type == "LSTM":
        encoder_output, encoder_hidden, encoder_cell = encoder(input_tensor, encoder_hidden)
    else:
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_input = torch.tensor([SOW_token]*batch_size, device=device)
    decoder_hidden = encoder_hidden
    
    if cell_type == "LSTM":
        decoder_cell = encoder_cell
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if cell_type == "LSTM":
                decoder_output, decoder_hidden, decoder_cell, attn_weights = decoder(decoder_input, (decoder_hidden, decoder_cell), encoder_output)
            else:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if cell_type == "LSTM":
                decoder_output, decoder_hidden, decoder_cell, attn_weights = decoder(decoder_input, (decoder_hidden, decoder_cell), encoder_output)
            else:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
#             decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length,  attn_weights

def getBatchedTensorFromWords(words, batch_size, lang):
    input_tensor = [tensorFromWord(lang, word) for word in words]
    batched_tensors = ((nn.utils.rnn.pad_sequence(input_tensor).squeeze(2)).to(device))
    batchWise = []
    for i in range(0, batched_tensors.shape[1], batch_size):
        currBatch = (batched_tensors[0:batched_tensors.shape[0], i:(i+batch_size)])
        if currBatch.shape[0] < max_length+1:
            padding_tensor = torch.zeros(max_length+1-currBatch.shape[0], currBatch.shape[1], dtype = currBatch.dtype, device=device)
            currBatch = torch.cat((currBatch, padding_tensor), dim = 0)
        batchWise.append(currBatch)
    
    return batchWise

def trainIters(encoder, decoder, n_datapoints, epochs, learning_rate, batch_size, embedding_size, cell_type, num_layers_encoder, num_layers_decoder, hidden_size, bidirectional, dropout_encoder, dropout_decoder, wandb_project, wandb_entity):
    wandb.init(
        project=wandb_project,
        entity=wandb_entity
    )
    run_name = "embS_{}_nlEnc_{}_nlDec_{}_hl_{}_cellType_{}_biDir_{}_dropEnc_{}_dropDec_{}_ep_{}_bs_{}".format(embedding_size, num_layers_encoder, num_layers_decoder, hidden_size, cell_type, bidirectional, dropout_encoder, dropout_decoder, epochs, batch_size)
    print_loss_total = 0 

    encoder_optimizer = optim.NAdam(encoder.parameters(), lr=learning_rate, weight_decay = 0.0005)
    decoder_optimizer = optim.NAdam(decoder.parameters(), lr=learning_rate, weight_decay = 0.0005)
    criterion = nn.CrossEntropyLoss()
    
    train_batch_input = getBatchedTensorFromWords(train_input, batch_size, lang_input)
    train_batch_target = getBatchedTensorFromWords(train_output, batch_size, lang_output)
    
    valid_batch_input = getBatchedTensorFromWords(valid_input, batch_size, lang_input)
    
    for epochNum in range(epochs):
        for i in range(len(train_batch_input)):
            loss, attn = train(train_batch_input[i], train_batch_target[i], encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, cell_type)
            print_loss_total += loss*batch_size

        print_loss_avg = print_loss_total / len(train_input)
        print_loss_total = 0
        print("Average loss after ", epochNum+1, "epochs is ", print_loss_avg)
        valid_accuracy = findAccuracy(encoder, decoder, valid_batch_input, valid_output, cell_type, len(valid_input), batch_size)
        print("Valid accuracy is ", valid_accuracy)
        wandb.log({"validation_accuracy": valid_accuracy, "training_loss": print_loss_avg, 'epoch': epochNum})

    train_accuracy = findAccuracy(encoder, decoder, train_batch_input, train_output, cell_type, len(train_input), batch_size)
    print("Train accuracy is ", train_accuracy)
    
    wandb.log({"training_accuracy": train_accuracy})
    wandb.run.name = run_name
    wandb.run.save()
    wandb.run.finish()
    return attn

def evaluate(encoder, decoder, input_tensors, cell_type, batch_size):
    with torch.no_grad():
        
        input_length = input_tensors.size(0)
        encoder_hidden = encoder.initHidden()

        if cell_type == "LSTM":
            encoder_output, encoder_hidden, encoder_cell = encoder(input_tensors, encoder_hidden)
        else:
            encoder_output, encoder_hidden = encoder(input_tensors, encoder_hidden)

        decoder_input = torch.tensor([SOW_token]*batch_size, device=device)  # SOW

        decoder_hidden = encoder_hidden

        if cell_type == "LSTM":
            decoder_cell = encoder_cell

        decoded_words = [""]*batch_size

        for di in range(input_length):
            
            if cell_type == "LSTM":
                decoder_output, decoder_hidden, decoder_cell, attn_weights = decoder(decoder_input, (decoder_hidden, decoder_cell), encoder_output)
            else:
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            topv, topi = decoder_output.data.topk(1)
            for i in range(batch_size):
                if topi[i].item() == EOW_token or topi[i] == 0:
                    continue
                else:
                    decoded_words[i] += lang_output.index2char[topi[i].item()]

            decoder_input = topi.squeeze().detach()

        return decoded_words
def findAccuracy(encoder1, decoder1, input, actual_output, cell_type, n, batch_size):
    correct = 0
    for i in range(len(input)):
        output_word = evaluate(encoder1, decoder1, input[i], cell_type, batch_size)
        for j in range(i*batch_size, i*batch_size+batch_size):
            if(actual_output[j] == output_word[j-i*batch_size]):
                correct += 1
    return 100*correct/n

def runSweep(params):
    
    embedding_size = params.emb_size
    num_layers_encoder = params.num_layers_encoder
    num_layers_decoder = params.num_layers_decoder
    hidden_size = params.hidden_size
    batch_size = params.batch_size
    epochs = params.epochs
    cell_type = params.cell_type
    bidirectional = params.bidirectional
    dropout_encoder = params.dropout_encoder
    dropout_decoder = params.dropout_decoder
    learning_rate = 0.001
    encoder1 = EncoderRNN(lang_input.n_chars, hidden_size, num_layers_encoder, dropout_encoder, batch_size, embedding_size, cell_type, bidirectional).to(device)
    decoder1 = AttnDecoderRNN(hidden_size, lang_output.n_chars, num_layers_decoder, dropout_decoder, batch_size, embedding_size, cell_type).to(device)
    attn = trainIters(encoder1, decoder1, len(train_input), epochs, learning_rate, batch_size, embedding_size, cell_type, num_layers_encoder, num_layers_decoder, hidden_size, bidirectional, dropout_encoder, dropout_decoder, params.wandb_project, params.wandb_entity)

parser = argparse.ArgumentParser(description='calculate accuracy and loss for given hyperparameters')
parser.add_argument('-wp', '--wandb_project', type=str, help='wandb project name', default='Assignment 3')
parser.add_argument('-we', '--wandb_entity', type=str, help='wandb entity', default='cs22m006')
parser.add_argument('-es', '--emb_size', type=int, help='embedding size', default=256)
parser.add_argument('-nle', '--num_layers_encoder', type=int, help='number of layers in encoder', default=3)
parser.add_argument('-nld', '--num_layers_decoder', type=int, help='number of layers in decoder', default=3)
parser.add_argument('-hs', '--hidden_size', type=int, help='hidden size', default=512)
parser.add_argument('-bs', '--batch_size', type=int, help='batch size', default=256)
parser.add_argument('-ep', '--epochs', type=int, help='epochs', default=20)
parser.add_argument('-ct', '--cell_type', type=str, help='Cell type', default="LSTM")
parser.add_argument('-bdir', '--bidirectional', type=bool, help='bidirectional', default=True)
parser.add_argument('-de', '--dropout_encoder', type=float, help='dropout encoder', default=0.4)
parser.add_argument('-dd', '--dropout_decoder', type=float, help='dropout decoder', default=0.2)
params = parser.parse_args()
if __name__ == '__main__':
    runSweep(params)
