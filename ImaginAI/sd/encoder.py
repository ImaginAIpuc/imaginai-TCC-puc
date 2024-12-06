import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_BlocoAtencao, VAE_BlocoResidual

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(

            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            VAE_BlocoResidual(128, 128),

            VAE_BlocoResidual(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            VAE_BlocoResidual(128, 256), 

            VAE_BlocoResidual(256, 256), 

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 

            VAE_BlocoResidual(256, 512), 

            VAE_BlocoResidual(512, 512), 

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            VAE_BlocoResidual(512, 512), 

            VAE_BlocoResidual(512, 512), 

            VAE_BlocoResidual(512, 512), 

            VAE_BlocoAtencao(512), 

            VAE_BlocoResidual(512, 512), 
            
            nn.GroupNorm(32, 512), 
            
            nn.SiLU(), 

            # "Devido ao padding=1, significa que a largura e altura aumentarão em 2.
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Como padding = 1 significa que Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Como Out_Width = In_Width + 2 (o mesmo para Out_Height), isso irá compensar o tamanho do Kernel de 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)."

            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):

        for module in self:

            if getattr(module, 'stride', None) == (2, 2): 
            
                x = F.pad(x, (0, 1, 0, 1))
            
                x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        x = mean + stdev * noise
        
        # Escalonado pela constante pega em: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x
