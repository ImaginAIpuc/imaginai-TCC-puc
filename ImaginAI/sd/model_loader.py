from clip import CLIP 
from encoder import VAE_Encoder
from decoder import VAE_Decodificador
from diffusion import Difusao

import model_converter

def carregar_modelos_pesos_padrao(caminho_ckpt, dispositivo):
    state_dict = model_converter.load_from_standard_weights(caminho_ckpt, dispositivo)

    encoder = VAE_Encoder().to(dispositivo)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decodificador().to(dispositivo)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Difusao().to(dispositivo)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(dispositivo)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }