## 🧩 **Resumo do funcionamento do código de segmentação**

O objetivo do código é **encontrar automaticamente os “furos” (trechos de baixa energia)** em um áudio — ou seja, **silêncios, pausas ou falhas de som**.

### ⚙️ **Etapas principais**

1. **Pré-processamento**

   * Aplica filtro de pré-ênfase e normaliza o áudio.
   * Deixa o sinal mais uniforme para análise.

2. **Cálculo da energia (RMS)**

   * Mede a intensidade sonora em janelas curtas.
   * Suaviza e normaliza a curva RMS para remover ruído.

3. **Detecção de vales**

   * Encontra pontos onde a energia cai (mínimos locais).
   * Usa o `find_peaks` invertido para detectar esses vales.

4. **Filtragem de vales**

   * Calcula a **profundidade** de cada vale (diferença entre pico e vale).
   * Mantém apenas os **vales realmente profundos**, que indicam quedas significativas.

5. **Agrupamento e fusão**

   * Agrupa vales próximos no tempo (parte do mesmo furo).
   * Remove furos curtos demais.
   * Funde furos muito próximos em um só trecho.

6. **Visualização**

   * Mostra o RMS com vales e furos destacados.
   * Plota o espectrograma com os furos sobrepostos.

---

### 🎯 **Resultado final**

O código gera uma lista de **intervalos de tempo** correspondentes a **furos de baixa energia** no áudio — bem como gráficos que permitem **ver e validar** essas detecções visualmente.

---

## ⚙️ **Parâmetros e seus efeitos**

| Parâmetro                         | Função                                                                | Impacto prático                                                                |
| --------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **`MIN_HOLE_DURATION`**           | Duração mínima de um furo (em segundos).                              | Controla o tamanho mínimo de pausa detectada. Aumentar → ignora pausas curtas. |
| **`SMOOTH_WINDOW`**               | Tamanho da janela de suavização do RMS (em frames).                   | Janela maior → curva de energia mais suave, menos sensível a ruído.            |
| **`VALLEY_WINDOW_SEC`**           | Janela (em segundos) para buscar picos antes/depois de cada vale.     | Afeta o cálculo da “profundidade” da queda local.                              |
| **`DEPTH_THRESH`**                | Profundidade mínima (0–1) do vale para ser considerado significativo. | Valor maior → detecta apenas quedas muito fortes de energia.                   |
| **`MIN_PROMINENCE_VALLEY`**       | Proeminência mínima bruta para `find_peaks`.                          | Filtra vales muito rasos ou flutuações pequenas.                               |
| **`GROUP_GAP_SEC`**               | Máximo intervalo (em s) entre vales para agrupá-los em um mesmo furo. | Ajuda a unir pausas fragmentadas próximas.                                     |
| **`MERGE_GAP_SEC`**               | Distância máxima entre furos para mesclá-los.                         | Funde pausas muito próximas em uma só.                                         |
| **`HOP_LENGTH` / `FRAME_LENGTH`** | Definem a resolução temporal e espectral da análise RMS.              | Menor hop → análise mais detalhada (mais frames, mais preciso).                |

---

## 🧩 **Principais variáveis intermediárias**

| Variável       | Conteúdo                                                            |
| -------------- | ------------------------------------------------------------------- |
| `rms_norm`     | Curva de energia normalizada ao longo do tempo.                     |
| `valleys_all`  | Todos os vales detectados (mínimos locais).                         |
| `valleys_kept` | Subconjunto de vales realmente profundos (filtrados).               |
| `depths`       | Profundidade de cada vale (intensidade da queda).                   |
| `holes`        | Lista final de furos detectados: pares `(início, fim)` em amostras. |

---

## 📊 **Saída e visualização**

* **`plot_diagnostics_with_valleys()`** → mostra o RMS, vales e furos (visão temporal).
* **`plot_spectrogram_with_holes()`** → mostra o espectrograma com furos destacados em verde.


