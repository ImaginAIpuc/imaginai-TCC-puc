import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMAmostra

LARGURA = 512
ALTURA = 512
LATENTES_LARGURA = LARGURA // 8
LATENTES_ALTURA = ALTURA // 8

def gerar(
    prompt,
    uncond_prompt=None,
    imagem_entrada=None,
    forca=0.8,
    fazer_cfg=True,
    escala_cfg=7.5,
    nome_denoiser="ddpm",
    n_inferencia_passos=50,
    modelos={},
    semente=None,
    dispositivo=None,
    dispositivo_ocioso=None,
    gerar_token=None,
):
    with torch.no_grad():
        if not 0 < forca <= 1:
            raise ValueError("forca deve estar entre 0 e 1")

        if dispositivo_ocioso:
            para_ocioso = lambda x: x.to(dispositivo_ocioso)
        else:
            para_ocioso = lambda x: x

        gerador = torch.Generator(device=dispositivo)
        if semente is None:
            gerador.seed()
        else:
            gerador.manual_seed(semente)

        clip = modelos["clip"]
        clip.to(dispositivo)
        
        if fazer_cfg:

            cond_tokens = gerar_token.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=dispositivo)

            cond_context = clip(cond_tokens)

            uncond_tokens = gerar_token.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids

            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=dispositivo)

            uncond_context = clip(uncond_tokens)

            contexto = torch.cat([cond_context, uncond_context])
        else:

            tokens = gerar_token.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            tokens = torch.tensor(tokens, dtype=torch.long, device=dispositivo)

            contexto = clip(tokens)
        para_ocioso(clip)

        if nome_denoiser == "ddpm":
            samplador = DDPMAmostra(gerador)
            samplador.definir_passos_inferencia(n_inferencia_passos)
        else:
            raise ValueError("Valor desconhecido para samplador %s. ")

        forma_latentes = (1, 4, LATENTES_ALTURA, LATENTES_LARGURA)

        if imagem_entrada:
            encoder = modelos["encoder"]
            encoder.to(dispositivo)

            imagem_entrada_tensor = imagem_entrada.resize((LARGURA, ALTURA))

            imagem_entrada_tensor = np.array(imagem_entrada_tensor)

            imagem_entrada_tensor = torch.tensor(imagem_entrada_tensor, dtype=torch.float32)

            imagem_entrada_tensor = rescale(imagem_entrada_tensor, (0, 255), (-1, 1))

            imagem_entrada_tensor = imagem_entrada_tensor.unsqueeze(0)

            imagem_entrada_tensor = imagem_entrada_tensor.permute(0, 3, 1, 2)

            ruido_encoder = torch.randn(forma_latentes, generator=gerador, device=dispositivo)

            latentes = encoder(imagem_entrada_tensor, ruido_encoder)

            samplador.set_strength(force=forca)
            latentes = samplador.add_noise(latentes, samplador.timesteps[0])

            para_ocioso(encoder)
        else:

            latentes = torch.randn(forma_latentes, generator=gerador, device=dispositivo)

        difusao = modelos["diffusion"]
        difusao.to(dispositivo)

        tempos = tqdm(samplador.tempos)
        for i, tempo in enumerate(tempos):

            emb_time = obter_embedding_tempo(tempo).to(dispositivo)

            entrada_modelo = latentes

            if fazer_cfg:

                entrada_modelo = entrada_modelo.repeat(2, 1, 1, 1)

            saida_modelo = difusao(entrada_modelo, contexto, emb_time)

            if fazer_cfg:
                saida_cond, saida_uncond = saida_modelo.chunk(2)
                saida_modelo = escala_cfg * (saida_cond - saida_uncond) + saida_uncond

            latentes = samplador.tempo(tempo, latentes, saida_modelo)

        para_ocioso(difusao)

        decoder = modelos["decoder"]
        decoder.to(dispositivo)
        imagens = decoder(latentes)
        para_ocioso(decoder)

        imagens = rescale(imagens, (-1, 1), (0, 255), clamp=True)
        imagens = imagens.permute(0, 2, 3, 1)
        imagens = imagens.to("cpu", torch.uint8).numpy()
        return imagens[0]
    
def rescale(x, intervalo_antigo, intervalo_novo, clamp=False):
    antigo_min, antigo_max = intervalo_antigo
    novo_min, novo_max = intervalo_novo
    x -= antigo_min
    x *= (novo_max - novo_min) / (antigo_max - antigo_min)
    x += novo_min
    if clamp:
        x = x.clamp(novo_min, novo_max)
    return x

def obter_embedding_tempo(tempo):

    frequencias = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([tempo], dtype=torch.float32)[:, None] * frequencias[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
