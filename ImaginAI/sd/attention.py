import torch
from torch import nn
from torch.nn import functional as F
import math

class AutoAtencao(nn.Module):
    def __init__(self, n_cabecas, d_emb, proj_entrada_bias=True, proj_saida_bias=True):
        super().__init__()
        self.proj_entrada = nn.Linear(d_emb, 3 * d_emb, bias=proj_entrada_bias)
        self.proj_saida = nn.Linear(d_emb, d_emb, bias=proj_saida_bias)
        self.n_cabecas = n_cabecas
        self.d_cabeca = d_emb // n_cabecas

    def forward(self, x, mascara_causal=False):
        formato_entrada = x.shape 
        tamanho_lote, comprimento_sequencia, d_emb = formato_entrada
        formato_intermediario = (tamanho_lote, comprimento_sequencia, self.n_cabecas, self.d_cabeca)

        q, k, v = self.proj_entrada(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(formato_intermediario).transpose(1, 2), (q, k, v))

        pesos = q @ k.transpose(-1, -2)
        
        if mascara_causal:
            mascara = torch.ones_like(pesos, dtype=torch.bool).triu(1)
            pesos.masked_fill_(mascara, -torch.inf)
        
        pesos /= math.sqrt(self.d_cabeca)
        pesos = F.softmax(pesos, dim=-1)

        saida = pesos @ v
        saida = saida.transpose(1, 2).reshape(formato_entrada)
        saida = self.proj_saida(saida)
        
        return saida

class AtencaoCruzada(nn.Module):
    def __init__(self, n_cabecas, d_emb, d_cruzada,  proj_entrada_bias=True, proj_saida_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_emb, d_emb, bias=proj_entrada_bias)
        self.k_proj = nn.Linear(d_cruzada, d_emb, bias=proj_entrada_bias)
        self.v_proj = nn.Linear(d_cruzada, d_emb, bias=proj_entrada_bias)
        self.proj_saida = nn.Linear(d_emb, d_emb, bias=proj_saida_bias)
        self.n_cabecas = n_cabecas
        self.d_cabeca = d_emb // n_cabecas
    
    def forward(self, x, y):
        formato_entrada = x.shape
        tamanho_lote, comprimento_sequencia, d_emb = formato_entrada
        formato_intermediario = (tamanho_lote, -1, self.n_cabecas, self.d_cabeca)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q, k, v = map(lambda t: t.view(formato_intermediario).transpose(1, 2), (q, k, v))

        pesos = q @ k.transpose(-1, -2)
        pesos /= math.sqrt(self.d_cabeca)
        pesos = F.softmax(pesos, dim=-1)
        saida = pesos @ v
        
        saida = saida.transpose(1, 2).contiguous().view(formato_entrada)
        saida = self.proj_saida(saida)

        return saida
