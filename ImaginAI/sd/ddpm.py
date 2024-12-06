import torch
import numpy as np

class DDPMAmostra:

    def __init__(self, gerador: torch.Generator, num_passos_treinamento=1000, beta_inicio: float = 0.00085, beta_fim: float = 0.0120):
        # Parâmetros "beta_inicio" e "beta_fim" obtidos de: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # Para as convenções de nomenclatura, consulte o arquivo (https://arxiv.org/pdf/2006.11239.pdf)
        self.betas = torch.linspace(beta_inicio ** 0.5, beta_fim ** 0.5, num_passos_treinamento, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_prod_cum = torch.cumprod(self.alphas, dim=0)
        self.um = torch.tensor(1.0)

        self.gerador = gerador

        self.num_passos_treinamento = num_passos_treinamento
        self.tempos = torch.from_numpy(np.arange(0, num_passos_treinamento)[::-1].copy())

    def definir_passos_inferencia(self, num_passos_inferencia=50):
        self.num_passos_inferencia = num_passos_inferencia
        proporcao_passos = self.num_passos_treinamento // self.num_passos_inferencia
        tempos = (np.arange(0, num_passos_inferencia) * proporcao_passos).round()[::-1].copy().astype(np.int64)
        self.tempos = torch.from_numpy(tempos)

    def _obter_passo_anterior(self, tempo: int) -> int:
        passo_anterior = tempo - self.num_passos_treinamento // self.num_passos_inferencia
        return passo_anterior
    
    def _obter_variacao(self, tempo: int) -> torch.Tensor:
        passo_anterior = self._obter_passo_anterior(tempo)

        alpha_prod_t = self.alphas_prod_cum[tempo]
        alpha_prod_t_anterior = self.alphas_prod_cum[passo_anterior] if passo_anterior >= 0 else self.um
        beta_atual_t = 1 - alpha_prod_t / alpha_prod_t_anterior

        # Para t > 0, calcula a variância prevista βt (ver fórmulas (6) e (7) em https://arxiv.org/pdf/2006.11239.pdf)
        # e adiciona ruído à amostra anterior
        variancia = (1 - alpha_prod_t_anterior) / (1 - alpha_prod_t) * beta_atual_t

        # Ajusta para garantir que não seja 0
        variancia = torch.clamp(variancia, min=1e-20)

        return variancia
    
    def definir_intensidade(self, intensidade=1):
        """
        Define o quanto de ruído adicionar à imagem de entrada.
        Mais ruído (intensidade ~ 1) significa que a saída estará mais distante da imagem de entrada.
        Menos ruído (intensidade ~ 0) significa que a saída estará mais próxima da imagem de entrada.
        """
        passo_inicio = self.num_passos_inferencia - int(self.num_passos_inferencia * intensidade)
        self.tempos = self.tempos[passo_inicio:]
        self.passo_inicio = passo_inicio

    def tempo(self, tempo: int, latentes: torch.Tensor, saida_modelo: torch.Tensor):
        t = tempo
        passo_anterior = self._obter_passo_anterior(t)

        # 1. Calcular alphas, betas
        alpha_prod_t = self.alphas_prod_cum[t]
        alpha_prod_t_anterior = self.alphas_prod_cum[passo_anterior] if passo_anterior >= 0 else self.um
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_anterior = 1 - alpha_prod_t_anterior
        alpha_atual_t = alpha_prod_t / alpha_prod_t_anterior
        beta_atual_t = 1 - alpha_atual_t

        # 2. Calcular amostra original prevista do ruído previsto ("x_0 previsto")
        amostra_original_prevista = (latentes - beta_prod_t ** (0.5) * saida_modelo) / alpha_prod_t ** (0.5)

        # 3. Calcular coeficientes para x_0 e x_t
        coef_amostra_original = (alpha_prod_t_anterior ** (0.5) * beta_atual_t) / beta_prod_t
        coef_amostra_atual = alpha_atual_t ** (0.5) * beta_prod_t_anterior / beta_prod_t

        # 4. Calcular a amostra anterior prevista µ_t
        amostra_prevista_anterior = coef_amostra_original * amostra_original_prevista + coef_amostra_atual * latentes

        # 5. Adicionar ruído
        variancia = 0
        if t > 0:
            dispositivo = saida_modelo.device
            ruido = torch.randn(saida_modelo.shape, generator=self.gerador, device=dispositivo, dtype=saida_modelo.dtype)
            variancia = (self._obter_variacao(t) ** 0.5) * ruido
        
        amostra_prevista_anterior += variancia

        return amostra_prevista_anterior
    
    def adicionar_ruido(
        self,
        amostras_originais: torch.FloatTensor,
        tempos: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_prod_cum = self.alphas_prod_cum.to(device=amostras_originais.device, dtype=amostras_originais.dtype)
        tempos = tempos.to(amostras_originais.device)

        sqrt_alpha_prod = alphas_prod_cum[tempos] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(amostras_originais.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_um_menos_alpha_prod = (1 - alphas_prod_cum[tempos]) ** 0.5
        sqrt_um_menos_alpha_prod = sqrt_um_menos_alpha_prod.flatten()
        while len(sqrt_um_menos_alpha_prod.shape) < len(amostras_originais.shape):
            sqrt_um_menos_alpha_prod = sqrt_um_menos_alpha_prod.unsqueeze(-1)

        # Retorna ruído adicionado
        return sqrt_alpha_prod * amostras_originais + sqrt_um_menos_alpha_prod * torch.randn_like(amostras_originais)
