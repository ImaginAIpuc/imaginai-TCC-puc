import torch
from torch import nn
from torch.nn import functional as F
from modulos import attention as attention
from modulos.attention import AutoAtencao

class VAE_BlocoAtencao(nn.Module):
    def __init__(self, canais):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, canais)
        self.atencao = AutoAtencao(1, canais)
    
    def forward(self, x):
        residuo = x 
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.atencao(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residuo
        return x 

class VAE_BlocoResidual(nn.Module):
    def __init__(self, canais_entrada, canais_saida):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, canais_entrada)
        self.conv_1 = nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, canais_saida)
        self.conv_2 = nn.Conv2d(canais_saida, canais_saida, kernel_size=3, padding=1)
        if canais_entrada == canais_saida:
            self.camada_residual = nn.Identity()
        else:
            self.camada_residual = nn.Conv2d(canais_entrada, canais_saida, kernel_size=1, padding=0)
    
    def forward(self, x):
        residuo = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.camada_residual(residuo)

class VAE_Decodificador(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_BlocoResidual(512, 512),
            VAE_BlocoAtencao(512),
            VAE_BlocoResidual(512, 512),
            VAE_BlocoResidual(512, 512),
            VAE_BlocoResidual(512, 512),
            VAE_BlocoResidual(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_BlocoResidual(512, 512),
            VAE_BlocoResidual(512, 512),
            VAE_BlocoResidual(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_BlocoResidual(512, 256),
            VAE_BlocoResidual(256, 256),
            VAE_BlocoResidual(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_BlocoResidual(256, 128),
            VAE_BlocoResidual(128, 128),
            VAE_BlocoResidual(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215  # Remove a escala gerada pelo codificador.
        for modulo in self:
            x = modulo(x)
        return x
