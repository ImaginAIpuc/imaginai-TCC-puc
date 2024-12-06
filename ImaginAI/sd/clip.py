import torch
from torch import nn
from torch.nn import functional as F
from attention import AutoAtencao  # Certifique-se de manter o nome correspondente ao arquivo traduzido

class EmbeddingCLIP(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        self.embedding_token = nn.Embedding(n_vocab, n_embd)
        self.embedding_posicao = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):

        x = self.embedding_token(tokens)
        x += self.embedding_posicao
        
        return x

class CamadaCLIP(nn.Module):
    def __init__(self, n_cabeca: int, n_embd: int):
        super().__init__()
        
        self.normalizacao_1 = nn.LayerNorm(n_embd)
        self.atencao = AutoAtencao(n_cabeca, n_embd)
        self.normalizacao_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):

        resíduo = x

        x = self.normalizacao_1(x)

        x = self.atencao(x, mascara_causal=True)

        x += resíduo

        resíduo = x

        x = self.normalizacao_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)
        
        x += resíduo

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingCLIP(49408, 768, 77)

        self.camadas = nn.ModuleList([
            CamadaCLIP(12, 768) for i in range(12)
        ])

        self.normalizacao_final = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        estado = self.embedding(tokens)

        for camada in self.camadas: 

            estado = camada(estado)

        saida = self.normalizacao_final(estado)
        
        return saida
