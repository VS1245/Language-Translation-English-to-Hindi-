import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TextDataset(Dataset):

    def __init__(self, english_sentences, hindi_sentences):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.hindi_sentences[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, L):
        super().__init__()
        self.d_model = d_model
        self.L = L
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        # odd_i = torch.arange(1, d_model, 2).float()
        even_denominator = torch.pow(10000, even_i/self.d_model)
        # odd_denominator = torch.pow(10000, odd_i/self.d_model)
        position = torch.arange(self.L).reshape(self.L, 1)
        even_PE = torch.sin(position/even_denominator)
        odd_PE = torch.cos(position/even_denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2).reshape(self.L, self.d_model)
        # print("Positional Encoding")
        return stacked

class SentenceEmbedding(nn.Module):
    def __init__(self, max_sq_len, d_model, language_to_index, START, END, PADDING):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sq_len = max_sq_len
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sq_len)
        self.dropout = nn.Dropout(p=0.1)
        self.START = START
        self.END = END
        self.PADDING = PADDING
    
    def batch_tokenisation(self, batch, start_token = True, end_token= True):
        def tokenize(sentence, start_token=True, end_token= True):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[START])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END])
            for _ in range(len(sentence_word_indicies), self.max_sq_len):
                sentence_word_indicies.append(self.language_to_index[PADDING])
            return torch.tensor(sentence_word_indicies)
        tokenized = []
        for i in batch:
            tokenized.append( tokenize(i, start_token=start_token, end_token = end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized

    def forward(self, x, start_token, end_token): # sentence
        x  = self.batch_tokenisation(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        # print("SentenceEmbedding")
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2*d_model)
        self.qlayer = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim = -1)
        self.linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self,q,k,v,mask=None):
        d_k = q.shape[-1]
        k_t = k.transpose(-2,-1)
        scaled1 = torch.matmul(q, k_t) / math.sqrt(d_k)
        if mask is not None:
            scaled1 = (scaled1.permute(1,0,2,3) + mask).permute(1,0,2,3)
        attention = self.softmax(scaled1)
        out = torch.matmul(attention, v)
        return out, attention
    
    def forward(self, x,y, mask = None):
        batch_size, L, d_model = x.size()
        kv = self.kv_layer(x) #1024
        q = self.qlayer(y) #512
        kv = kv.reshape(batch_size, L, self.num_heads, 2*self.head_dim)
        q = q.reshape(batch_size, L, self.num_heads, self.head_dim)
        kv = kv.permute(0,2,1,3)
        q = q.permute(0,2,1,3)
        k, v = kv.chunk(2, dim=-1)
        out, attention = self.scaled_dot_product_attention(q,k,v,mask)
        out = out.permute(0,2,1,3)
        out = out.reshape(batch_size, L, self.d_model)
        out = self.linear(out)
        # print("MultiHeadCrossAttention")
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        self.softmax = nn.Softmax(dim = -1)
        self.linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self,q,k,v,mask=None):
        d_k = q.shape[-1]
        k_t = k.transpose(-2,-1)
        scaled = torch.matmul(q, k_t) / math.sqrt(d_k)
        if mask is not None:
            scaled = (scaled.permute(1,0,2,3) + mask).permute(1,0,2,3)
        attention = self.softmax(scaled)
        out = torch.matmul(attention, v)
        return out, attention
    
    def forward(self, x, mask = None):
        batch_size, L, d_model = x.size()
        qkv = self.qkv_layer(x) #1536
        qkv = qkv.reshape(batch_size, L, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        q, k, v = qkv.chunk(3, dim=-1)
        out, attention = self.scaled_dot_product_attention(q,k,v,mask)
        out = out.permute(0,2,1,3)
        out = out.reshape(batch_size, L, self.d_model)
        out = self.linear(out)
        # print("MultiHeadAttention")
        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden,drop_prob):
        super(PositionWiseFeedForward, self).__init__()
        self.layer1 = nn.Linear(d_model, hidden)
        self.layer2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        # print("PositionWiseFeedForward")
        return x


class LayerNormalisation(nn.Module):
    def __init__(self,parameters_shape,eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean)**2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean)/std
        out = self.gamma * y + self.beta
        # print("LayerNormalisation")
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model,ffn_hidden, num_heads,  drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm1 = LayerNormalisation(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalisation([d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm3 = LayerNormalisation([d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self,x,y,self_attention_mask, cross_attention_mask):
        residual_y = y
        y = self.attention(y, self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + residual_y)
        residual_y = y
        y = self.cross_attention(x,y, cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y + residual_y)
        residual_y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + residual_y)
        return y

class SequentialDecoder(nn.Sequential):
    # def forward(self, *inputs):
    def forward(self, x,y, self_mask, cross_mask):
        # x,y,self_mask, cross_mask = inputs
        for module in self._modules.values():
            y = module(x,y,self_mask, cross_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sq_len, language_to_index, START, END, PADDING):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sq_len, d_model, language_to_index, START, END, PADDING)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self,x,y,self_attention_mask, cross_attention_mask, START, END):
        y = self.sentence_embedding(y,START, END)
        # try:
        #     y = self.layers(x,y,self_attention_mask, cross_attention_mask)
        # except:
        #     print("Error is in decoder")
        y = self.layers(x,y,self_attention_mask, cross_attention_mask)
        return y

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalisation(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalisation([d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self,x, self_attention_mask):
        residual_x =x
        x = self.attention(x, mask= self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x+residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+residual_x)
        return x

class SequentialEncoder(nn.Sequential):
    # def forward(self, *inputs):
    def forward(self, x, self_mask):
        for module in self._modules.values():
            x = module(x, self_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sq_len, language_to_index, START, END, PADDING):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sq_len, d_model, language_to_index, START, END, PADDING)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        
    def forward(self,x, self_attention_mask, START, END):
        x = self.sentence_embedding(x, START, END)
        # try:
        #     x = self.layers(x, self_attention_mask)
        # except:
        #     print("Error is in encoder")
        # return x
        x = self.layers(x, self_attention_mask)
        return x

class  Transformer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sq_len, hindi_vocab_size, english_to_index, hindi_to_index, START, END, PADDING):
        super().__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sq_len, english_to_index, START, END, PADDING)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sq_len, hindi_to_index, START, END, PADDING)
        self.linear = nn.Linear(d_model, hindi_vocab_size)
    
    def forward(self, x, y,encoder_self_attention_mask = None, decoder_self_attention_mask = None, decoder_cross_attention_mask = None, enc_start = False, dec_start= False, dec_end = False, enc_end = False):
        x = self.encoder(x, encoder_self_attention_mask, enc_start, enc_end)
        out = self.decoder(x, y , decoder_self_attention_mask, decoder_cross_attention_mask, dec_start, dec_end)
        out = self.linear(out)
        return out

def create_masks(eng_batch, hin_batch):
    NEG_INFTY = -1e9
    num_sen = len(eng_batch)
    look_ahead_mask = torch.full([max_sq_len, max_sq_len],True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sen, max_sq_len, max_sq_len], False)
    decoder_padding_mask_self_attention = torch.full([num_sen, max_sq_len, max_sq_len], False)
    decoder_padding_mask_cross_attention = torch.full([num_sen, max_sq_len, max_sq_len], False)

    for i in range(num_sen):
        eng_sen_len, hin_sen_len = len(eng_batch[i]), len(hin_batch[i])
        eng_padding_char = np.arange(eng_sen_len, max_sq_len)
        hin_padding_char = np.arange(hin_sen_len, max_sq_len)
        encoder_padding_mask[i,:,eng_padding_char] = True
        encoder_padding_mask[i,eng_padding_char, :] = True
        decoder_padding_mask_self_attention[i, :, hin_padding_char] = True
        decoder_padding_mask_self_attention[i, hin_padding_char, :] = True
        decoder_padding_mask_cross_attention[i, :, eng_padding_char] = True
        decoder_padding_mask_cross_attention[i, hin_padding_char, :] = True

    
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY,0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


START = '<START>'
PADDING = '<PADDING>'
END = '<END>'

characters_hindi = [START,'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 
                    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 
                    'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 
                    'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह','ॉ',
                    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 
                    'ः', '्', 'ॐ', '।', '॥', 'ँ', '़', 'ऽ', '०', '१', '२', 
                    '३', '४', '५', '६', '७', '८', '९', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', ':', '!',
                    '?', '(', ')', '‘', '’', '“', '”', '-', '_',' ', PADDING, END]


characters_english = [START,'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
              '.', ',', ':', ';', '!', '?', "'", '"', '(', ')', '[', ']', '{', '}', 
              '-', '_', '/', '\\', '@', '#', '$', '%', '^', '&','|', '~', '`', 
              '0','1', '2', '3', '4', '5', '6', '7', '8', '9',' ', PADDING, END]


hindi_vocab_size = len(characters_hindi)
enlish_vocab_size = len(characters_english)

with open('en-hi\\train.en') as enfile:
    english = [next(enfile) for _ in range(10000)]
with open('en-hi\\train.hi', encoding = 'utf-8') as hifile:
    hindi =  [next(hifile) for _ in range(10000)]

index_to_hindi = {k:v for k,v in enumerate(characters_hindi)}
hindi_to_index = {v:k for k,v in enumerate(characters_hindi)}
index_to_english = {k:v for k,v in enumerate(characters_english)}
english_to_index = {v:k for k,v in enumerate(characters_english)}

max_sq_len = 200

def is_valid_token(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def isvalidlength(sentence, max_sq_len):
    return len(list(sentence))< (max_sq_len-1)

valid_index = []
for i in range(10000):
    if hindi[i][-1]=='\n':
            hindi[i] = hindi[i][:-1]
    if english[i][-1]=='\n':
        english[i] = english[i][:-1]
    hsen, esen = hindi[i], english[i]
    if(is_valid_token(hsen,characters_hindi) and is_valid_token(esen, characters_english) and isvalidlength(esen,max_sq_len) and isvalidlength(hsen,max_sq_len)):
        valid_index.append(i)
# print(len(valid_index))


d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob =0.1
num_layers = 1
max_sq_len = 200
hindi_vocab_size = len(characters_hindi)
transformer = Transformer(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sq_len, hindi_vocab_size, english_to_index, hindi_to_index, START, END, PADDING)

english_sentences = [english[i] for i in valid_index]
hindi_sentences = [hindi[i] for i in valid_index]

dataset = TextDataset(english_sentences, hindi_sentences)
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

criterian = nn.CrossEntropyLoss(ignore_index=hindi_to_index[PADDING],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in tqdm(enumerate(iterator)):
        transformer.train()
        eng_batch, hn_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, hn_batch)
        optim.zero_grad()
        hn_predictions = transformer(eng_batch, hn_batch, encoder_self_attention_mask.to(device), decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device),False,False,True,True)
        labels = transformer.decoder.sentence_embedding.batch_tokenisation(hn_batch, start_token=False, end_token=True)
        loss = criterian(hn_predictions.view(-1, hindi_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == hindi_to_index[PADDING], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Hindi Translation: {hn_batch[0]}")
            hn_sentence_predicted = torch.argmax(hn_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in hn_sentence_predicted:
              if idx == hindi_to_index[END]:
                break
              predicted_sentence += index_to_hindi[idx.item()]
            print(f"Hindi Prediction: {predicted_sentence}")


            transformer.eval()
            hn_sentence = ("",)
            eng_sentence = ("What is your name?",)
            for word_counter in range(max_sq_len):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, hn_sentence)
                predictions = transformer(eng_sentence,hn_sentence,encoder_self_attention_mask.to(device), decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device),False,False,True,False)
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_hindi[next_token_index]
                hn_sentence = (hn_sentence[0] + next_token, )
                if next_token == END:
                  break
            
            print(f"Evaluation translation (What is your name?) : {hn_sentence}")