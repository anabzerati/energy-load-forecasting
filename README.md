# Previsão de Demanda Elétrica Urbana baseada em Dados de Sensores

Este repositório contém os códigos, experimentos e resultados do estudo sobre previsão de consumo energético utilizando modelos de Deep Learning aplicados a séries temporais multivariadas. O trabalho avalia diferentes janelas temporais (lags), horizontes de previsão e combinações de variáveis (consumo, temporais e climáticas), comparando o desempenho de LSTM e Transformer.

Dados de consumo energético de [High-Resolution Load Dataset from Smart Meters Across Various Cities in Morocco](https://archive.ics.uci.edu/dataset/1158/high-resolution+load+dataset+from+smart+meters+across+various+cities+in+morocco)

Dados climáticos obtidos pela [Open Meteo API](https://open-meteo.com/)

## Objetivos

- Comparar arquiteturas de **LSTM** e **Transformer** para predição de **curto** a **médio prazo** (1h e 6h).
- Analisar o impacto de **lags** temporais (3h, 6h, 12h, 24h).  
- Avaliar a contribuição de **variáveis externas**: histórico de consumo + variáveis temporais e climáticas

## Principais Resultados

| Features                    | Modelo | Lag  | Horizonte | MAE    | RMSE  | MAPE  | R²    |
|-----------------------------|--------|----- |-----------|--------|-------|-------|-------|
| Somente Consumo             | LSTM   | 24h  | 1h        | 25.331 | 39.989| 0.068| 0.785|
| Somente Consumo             | Transformer   | 24h       | 6h     | 56.288 | 72.374 | 0.152 |0.295 |
| Consumo + Temporal          | LSTM   | 24h   | 1h       | 25.248 | 38.673 | 0.068 | 0.799 |
| Consumo + Temporal          | Transformer   | 24h | 6h        | 59.966 | 73.365 | 0.163 | 0.275 |
| Consumo + Climático         | LSTM   | 24h | 1h        | 25.434 | 39.874  | 0.068 | 0.786|
| Consumo + Climático         | Transformer   | 24h | 6h        | 60.486 | 75.976 | 0.169 | 0.223 |
| Consumo + Temp + Climático  | LSTM   | 24h | 1h        | **23.970** | **37.996** | **0.064** | **0.805** |
| Consumo + Temp + Climático  | Transformer | 24h | 6h        | **54.721** | **67.715** | **0.152** | **0.382** |

- Combinação de fearures temporais e climáticas é benefecial, reduzindo erros e aumentando o R² em praticamente todos os cenários.
- LSTM apresenta melhor desempenho em horizontes curtos (1h) e quando utiliza lags menores - captura de dependências locais de curto prazo.
- Transformers apresentam desempenho superior em horizontes mais longos (6h) e quando utilizam lags maiores - modelam dependências de longo alcance.
