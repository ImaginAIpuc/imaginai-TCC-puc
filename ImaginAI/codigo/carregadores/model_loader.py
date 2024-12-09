import modulos.clip as clip
from modulos.clip import CLIP 
from modulos import codificador as codificador
from modulos.codificador import VAE_Encoder
from modulos import decodificador as decodificador
from modulos.decodificador import VAE_Decodificador
from modulos import diffusion as difussion
from modulos.diffusion import Difusao
import carregadores.model_converter as model_converter

def carregar_modelos_pesos_padrao(caminho_ckpt, dispositivo):
    try:
        state_dict = model_converter.load_from_standard_weights(caminho_ckpt, dispositivo)
    except KeyError as e:
        print(f"Erro ao carregar o estado do modelo: {e}")
        raise

    codificador = VAE_Encoder().to(dispositivo)
    decodificador = VAE_Decodificador().to(dispositivo)
    diffusion = Difusao().to(dispositivo)
    clip = CLIP().to(dispositivo)

    try:
        codificador.load_state_dict(state_dict['encoder'], strict=True)
        decodificador.load_state_dict(state_dict['decoder'], strict=True)
        diffusion.load_state_dict(state_dict['diffusion'], strict=True)
        clip.load_state_dict(state_dict['clip'], strict=True)
    except KeyError as e:
        print(f"Chave '{e}' não encontrada no estado do modelo.")
        raise

    return {
        'clip': clip,
        'encoder': codificador,
        'decoder': decodificador,
        'diffusion': diffusion,
    }
