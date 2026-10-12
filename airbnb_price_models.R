# ==============================================================================
# Script R para Previsão de Preços de Quartos Airbnb em Budapeste
# Metodologia: CRISP-DM
# Dataset: listings_Budapest.csv
# ==============================================================================

# ==============================================================================
# 1. CONFIGURAÇÃO E CARREGAMENTO DE PACOTES
# ==============================================================================

# Instalação de pacotes (descomentar se necessário)
# install.packages(c("caret", "ggplot2", "dplyr", "corrplot", "randomForest", 
#                    "gbm", "e1071", "ipred", "tidyr"))

# Carregamento de pacotes
library(caret)
library(ggplot2)
library(dplyr)
library(corrplot)
library(randomForest)
library(gbm)
library(e1071)
library(ipred)
library(tidyr)
library(doParallel)
library(tidyverse)


# Configuração para reprodutibilidade
set.seed(123)

# ==============================================================================
# 2. ETAPA 1: BUSINESS UNDERSTANDING
# ==============================================================================

# Objetivo: Estimar o preço de quartos listados no Airbnb em Budapeste
# 
# Variáveis preditoras selecionadas:
# - neighbourhood: distrito (localização é crucial para preços)
# - latitude/longitude: Posição geográfica exata
# - room_type: Tipo de alojamento (casa inteira, quarto privado, etc.)
# - minimum_nights: Requisito mínimo de estadia
# - number_of_reviews: Indicador de popularidade
# - reviews_per_month: Frequência de utilização
# - calculated_host_listings_count: Profissionalismo do anfitrião
# - availability_365: Disponibilidade anual
# - number_of_reviews_ltm: Atividade recente
#
# Variáveis excluídas (irrelevantes para previsão):
# - id: Identificador único sem valor preditivo
# - name: Texto descritivo, não estruturado
# - host_id: Identificador sem valor preditivo
# - host_name: Nome do anfitrião, irrelevante
# - neighbourhood_group: Contém apenas NAs no dataset de Budapeste
# - last_review: Data da última avaliação, convertida em features mais úteis
# - license: Informação de licença, irrelevante para preço

# ==============================================================================
# 3. CARREGAMENTO E INSPEÇÃO INICIAL DOS DADOS
# ==============================================================================

# Pré-processamento do ficheiro para limpar linhas problemáticas
file_path <- "data/listings_Budapest.csv"

if (file.exists(file_path)) {
  # Leitura direta do CSV
  dados <- read.csv(file_path, stringsAsFactors = FALSE)
  
  cat("Dados carregados com sucesso!\n")
  cat("Total de observações:", nrow(dados), "\n")
} else {
  stop("Ficheiro não encontrado: ", file_path)
}

#Salvar linhas desformatadas
if (file.exists(file_path)) {
  linhas_brutas <- readLines(file_path, warn = FALSE)
  linhas_limpas <- gsub(";;$", "", linhas_brutas)
  linhas_limpas <- linhas_limpas[linhas_limpas != ""]
  linhas_limpas <- gsub('^\"|\"$', "", linhas_limpas)
  linhas_limpas <- gsub('\"\"', '\"', linhas_limpas, fixed = TRUE)
  
  dados <- read_csv(I(linhas_limpas), show_col_types = FALSE)
  message("Dados carregados com sucesso! Linhas totais: ", nrow(dados))
} else {
  stop("ERRO: Ficheiro não encontrado.")
}

# Inspeção inicial
cat("=== DIMENSÕES DO DATASET ===\n")
print(dim(dados))

cat("\n=== ESTRUTURA DOS DADOS ===\n")
str(dados)

cat("\n=== PRIMEIRAS LINHAS ===\n")
head(dados)

cat("\n=== RESUMO ESTATÍSTICO ===\n")
summary(dados)

# Remover linhas com price = 0 ou NA
dados <- dados %>%
  filter(!is.na(price) & price > 0)

dados <- dados %>%
  select(-any_of(c("neighbourhood_group")))

dados$neighbourhood <- as.factor(dados$neighbourhood)
dados$room_type <- as.factor(dados$room_type)
dados$host_id <- as.character(dados$host_id)
dados$id <- as.character(dados$id)
dados$price <- dados$price*0.0026

#número de dias desde a última review
data_referencia <- max(dados$last_review, na.rm = TRUE)
dados$days_since_last_review <- as.numeric(data_referencia - dados$last_review)
summary(dados)



#dados <- dados %>%
#  filter(price < 31475828) %>%
#  filter(minimum_nights <300)


#nao utilizados após a análise
dados_longas_estadias <- dados %>%
  filter(minimum_nights > 30)
summary(dados_longas_estadias)

dados_sem_reviews <- dados %>%
  filter(number_of_reviews == 0)
summary(dados_sem_reviews)

dados_sem_avail <- dados %>%
  filter(availability_365 == 0)
summary(dados_sem_avail)

dados_preço_alto <- dados %>%
  filter(price > 150)
summary(dados_preço_alto)

dados_anuncios_inativos <- dados %>%
  filter(days_since_last_review >365)
summary(dados_anuncios_inativos)




# ==============================================================================
# 4. ETAPA 2: LIMPEZA DE DADOS
# ==============================================================================

cat("\n=== VALORES OMISSOS ANTES DA LIMPEZA ===\n")
print(colSums(is.na(dados)))

# Remover colunas irrelevantes (usando any_of para evitar erros se coluna não existir)
dados_limpos <- dados %>%
  select(-any_of(c("id", "name", "host_id", "host_name", "neighbourhood_group", 
                   "last_review", "license")))

# Converter colunas numéricas (podem ter sido lidas como texto)
dados_limpos$price <- as.numeric(dados_limpos$price)
dados_limpos$latitude <- as.numeric(dados_limpos$latitude)
dados_limpos$longitude <- as.numeric(dados_limpos$longitude)
dados_limpos$minimum_nights <- as.numeric(dados_limpos$minimum_nights)
dados_limpos$number_of_reviews <- as.numeric(dados_limpos$number_of_reviews)
dados_limpos$reviews_per_month <- as.numeric(dados_limpos$reviews_per_month)
dados_limpos$calculated_host_listings_count <- as.numeric(dados_limpos$calculated_host_listings_count)
dados_limpos$availability_365 <- as.numeric(dados_limpos$availability_365)
dados_limpos$number_of_reviews_ltm <- as.numeric(dados_limpos$number_of_reviews_ltm)

# Tratamento de valores omissos em reviews_per_month (substituir por 0)
dados_limpos$reviews_per_month[is.na(dados_limpos$reviews_per_month)] <- 0



cat("\n=== VALORES OMISSOS APÓS TRATAMENTO ===\n")
print(colSums(is.na(dados_limpos)))

# Tratamento de outliers usando IQR para price
Q1 <- quantile(dados_limpos$price, 0.25)
Q3 <- quantile(dados_limpos$price, 0.75)
IQR <- Q3 - Q1
limite_inferior <- Q1 - 1.5 * IQR
limite_superior <- Q3 + 1.5 * IQR

print(limite_inferior)
print(limite_superior)

cat("\n=== ANÁLISE DE OUTLIERS (PRICE) ===\n")
cat("Q1:", Q1, "\n")
cat("Q3:", Q3, "\n")
cat("IQR:", IQR, "\n")
cat("Limite inferior:", limite_inferior, "\n")
cat("Limite superior:", limite_superior, "\n")

n_antes <- nrow(dados_limpos)
dados_limpos <- dados_limpos %>%                                  
  filter(price >= limite_inferior & price <= limite_superior)
n_depois <- nrow(dados_limpos)




cat("Observações antes:", n_antes, "\n")
cat("Observações após remoção de outliers:", n_depois, "\n")
cat("Outliers removidos:", n_antes - n_depois, "\n")

# ==============================================================================
# 5. ETAPA 3: ANÁLISE EXPLORATÓRIA (EDA)
# ==============================================================================

cat("\n=== ESTATÍSTICAS DESCRITIVAS APÓS LIMPEZA ===\n")
summary(dados_limpos)

# Histograma de preços
ggplot(dados_limpos, aes(x = price)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribuição de Preços - Airbnb Budapeste",
       x = "Preço (EUR)", y = "Frequência") +
  theme_minimal()


# Histograma de preços altos #############################################################################################
ggplot(dados_preço_alto, aes(x = price)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribuição de Preços altos - Airbnb Budapeste",
       x = "Preço (EUR)", y = "Frequência") +
  theme_minimal()

# Boxplot por room_type
ggplot(dados_limpos, aes(x = room_type, y = price, fill = room_type)) +
  geom_boxplot() +
  labs(title = "Preço por Tipo de Quarto",
       x = "Tipo de Quarto", y = "Preço (EUR)") +
  theme_minimal() +
  theme(legend.position = "none")

# Contagem por room_type
ggplot(dados_limpos, aes(x = room_type, fill = room_type)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(title = "Número de Listagens por Tipo de Quarto",
       x = "Tipo de Quarto", y = "Contagem") +
  theme_minimal() +
  theme(legend.position = "none")

# Boxplot por neighbourhood (top 10 distritos)
top_neighbourhoods <- dados_limpos %>%
  count(neighbourhood) %>%
  arrange(desc(n)) %>%
  head(10) %>%
  pull(neighbourhood)

dados_limpos %>%
  filter(neighbourhood %in% top_neighbourhoods) %>%
  ggplot(aes(x = reorder(neighbourhood, price, FUN = median), 
             y = price, fill = neighbourhood)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Preço por distrito (Top 10)",
       x = "distrito", y = "Preço (EUR)") +
  theme_minimal() +
  theme(legend.position = "none")



# Scatter plot: latitude vs longitude (mapa de preços)
ggplot(dados_limpos, aes(x = longitude, y = latitude, color = price)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Distribuição Geográfica de Preços",
       x = "Longitude", y = "Latitude", color = "Preço (EUR)") +
  theme_minimal()

# Distribuição de availability_365
ggplot(dados_limpos, aes(x = availability_365)) +
  geom_histogram(bins = 50, fill = "coral", color = "white") +
  labs(title = "Distribuição de Disponibilidade Anual",
       x = "Dias Disponíveis (365)", y = "Frequência") +
  theme_minimal()

# Scatter plot: reviews_per_month vs price
ggplot(dados_limpos, aes(x = reviews_per_month, y = price)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Relação entre Avaliações por Mês e Preço",
       x = "Avaliações por Mês", y = "Preço (EUR)") +
  theme_minimal()

# Histogramas das variáveis numéricas principais
dados_limpos %>%
  select(price, minimum_nights, number_of_reviews, reviews_per_month, 
         availability_365) %>%
  gather(key = "variavel", value = "valor") %>%
  ggplot(aes(x = valor)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~variavel, scales = "free") +
  labs(title = "Distribuição das Variáveis Numéricas") +
  theme_minimal()

# ==============================================================================
# 6. ETAPA 4: ANÁLISE DE CORRELAÇÃO
# ==============================================================================

# Selecionar apenas variáveis numéricas
variaveis_numericas <- dados_limpos %>%
  select(price, latitude, longitude, minimum_nights, number_of_reviews,
         reviews_per_month, calculated_host_listings_count, 
         availability_365, number_of_reviews_ltm)

# Calcular matriz de correlação
matriz_correlacao <- cor(variaveis_numericas, use = "complete.obs")

cat("\n=== MATRIZ DE CORRELAÇÃO ===\n")
print(round(matriz_correlacao, 3))

# Heatmap de correlação com visualização melhorada
corrplot(matriz_correlacao, 
         method = "color",
         type = "upper",
         addCoef.col = "black",
         number.cex = 0.6,
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 0.7,
         cl.cex = 0.7,
         title = "Matriz de Correlação",
         mar = c(0, 0, 2, 2),
         col = colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))(200),
         diag = FALSE)

# Correlações com price
cat("\n=== CORRELAÇÕES COM PRICE ===\n")
correlacoes_price <- matriz_correlacao["price", ]
correlacoes_price <- sort(correlacoes_price, decreasing = TRUE)
print(round(correlacoes_price, 3))

# ==============================================================================
# 7. ETAPA 5: PRÉ-PROCESSAMENTO
# ==============================================================================

# Feature engineering: Distância ao centro calculado a partir dos dados
# O centro é calculado como a média das coordenadas do dataset
centro_lat <- mean(dados_limpos$latitude, na.rm = TRUE)
centro_lon <- mean(dados_limpos$longitude, na.rm = TRUE)

cat("\n=== CENTRO CALCULADO A PARTIR DOS DADOS ===\n")
cat("Latitude média (centro):", centro_lat, "\n")
cat("Longitude média (centro):", centro_lon, "\n")


#centro_lat <- 47.50382
#centro_lon <- 19.0625

dados_limpos <- dados_limpos %>%
  mutate(
    dist_centro = sqrt((latitude - centro_lat)^2 + (longitude - centro_lon)^2)
  )

# Visualizar nova feature
ggplot(dados_limpos, aes(x = dist_centro, y = price)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Relação entre Distância ao Centro e Preço",
       x = "Distância ao Centro (calculado)", y = "Preço (EUR)") +
  theme_minimal()

# Codificação de variáveis categóricas usando dummies
cat("\n=== CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS ===\n")#################################################################################################

# Criar dummies para room_type
#dummies_room <- model.matrix(~ room_type - 1, data = dados_limpos)
#colnames(dummies_room) <- gsub("room_type", "room_", colnames(dummies_room))

# Criar dummies para neighbourhood (top 10, resto como "other")
#dados_limpos <- dados_limpos %>%
#  mutate(neighbourhood_grouped = ifelse(
#    neighbourhood %in% top_neighbourhoods, 
#    neighbourhood, 
#    "Other"
#  ))

#dummies_neighbourhood <- model.matrix(~ neighbourhood_grouped - 1, data = dados_limpos)
#colnames(dummies_neighbourhood) <- gsub("neighbourhood_grouped", "neigh_", 
#                                         colnames(dummies_neighbourhood))

# Preparar dataset final para modelação
dados_modelo <- dados_limpos %>%
  select(price, latitude, longitude, minimum_nights, number_of_reviews,
         reviews_per_month, calculated_host_listings_count, 
         availability_365, number_of_reviews_ltm, dist_centro, neighbourhood, room_type)###################################################################

# Adicionar dummies
#dados_modelo <- cbind(dados_modelo, dummies_room, dummies_neighbourhood)

# Limpar nomes das colunas (remover espaços e caracteres especiais)
colnames(dados_modelo) <- make.names(colnames(dados_modelo))

cat("Dimensões do dataset para modelação:", dim(dados_modelo), "\n")
cat("Variáveis incluídas:\n")
print(names(dados_modelo))

# ==============================================================================
# 8. ETAPA 6: DIVISÃO TREINO/TESTE
# ==============================================================================

set.seed(123)

# Criar índices para divisão 80/20
indices_treino <- createDataPartition(dados_modelo$price, p = 0.8, list = FALSE)

# Dividir dados
dados_treino <- dados_modelo[indices_treino, ]
dados_teste <- dados_modelo[-indices_treino, ]

# Forçar conversão para data.frame (garantia extra)
dados_treino <- as.data.frame(dados_treino)
dados_teste <- as.data.frame(dados_teste)

cat("\n=== ESTRUTURA DOS DADOS DE TREINO ===\n")
str(dados_treino)

cat("\n=== DIVISÃO TREINO/TESTE ===\n")
cat("Observações de treino:", nrow(dados_treino), "\n")
cat("Observações de teste:", nrow(dados_teste), "\n")

# Verificar distribuição de price em ambos os conjuntos
cat("\nDistribuição de price no treino:\n")
print(summary(dados_treino$price))
cat("\nDistribuição de price no teste:\n")
print(summary(dados_teste$price))

# ==============================================================================
# 9. ETAPA 7: TREINO DE MODELOS
# ==============================================================================

# Configurar processamento paralelo
cl <- makePSOCKcluster(detectCores() - 1) # Usa todos os cores menos um
registerDoParallel(cl)
cat("Processamento paralelo iniciado com", getDoParWorkers(), "nucleos.\n")

# Configuração de validação cruzada 10-fold
ctrl <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = FALSE
)

cat("\n=== TREINO DE MODELOS ===\n")

# --- Modelo 1: Regressão Linear ---
cat("\n--- Treinando Regressão Linear ---\n")
modelo_lm <- train(
  price ~ .,
  data = dados_treino,
  method = "lm",
  trControl = ctrl
)
print(modelo_lm)
summary(modelo_lm)

mtry_grid <- data.frame(

  mtry = c(1, 2, 3, 4, 6, 8, 10)
)

# --- Modelo 2: Random Forest ---
cat("\n--- Treinando Random Forest ---\n")
modelo_rf <- train(
  price ~ .,
  data = dados_treino,
  method = "rf",
  trControl = ctrl,
#  tuneLength = 3,  # Testar 3 valores de mtry
  tuneGrid = mtry_grid,
  ntree = 100
)
print(modelo_rf)

# --- Modelo 3: Bagging ---
cat("\n--- Treinando Bagging ---\n")
modelo_bag <- train(
  price ~ .,
  data = dados_treino,
  method = "treebag",
  trControl = ctrl,
  nbagg = 100
)
print(modelo_bag)

# --- Modelo 4: Gradient Boosting ---
cat("\n--- Treinando Gradient Boosting ---\n")
modelo_gbm <- train(
  price ~ .,
  data = dados_treino,
  method = "gbm",
  trControl = ctrl,
  verbose = FALSE,
  tuneLength = 3
)
print(modelo_gbm)

# Parar cluster de processamento paralelo
stopCluster(cl)
registerDoSEQ()
cat("Processamento paralelo finalizado.\n")

# ==============================================================================
# 10. ETAPA 8: VALIDAÇÃO (PREVISÕES NO CONJUNTO DE TESTE)
# ==============================================================================

cat("\n=== PREVISÕES NO CONJUNTO DE TESTE ===\n")

# Previsões para cada modelo
pred_lm <- predict(modelo_lm, newdata = dados_teste)
pred_rf <- predict(modelo_rf, newdata = dados_teste)
pred_bag <- predict(modelo_bag, newdata = dados_teste)
pred_gbm <- predict(modelo_gbm, newdata = dados_teste)

# Criar dataframe com previsões
resultados_previsoes <- data.frame(
  Real = dados_teste$price,
  LM = pred_lm,
  RF = pred_rf,
  Bagging = pred_bag,
  GBM = pred_gbm
)

cat("\nPrimeiras previsões:\n")
print(head(resultados_previsoes))

# ==============================================================================
# 11. ETAPA 9: AVALIAÇÃO DE PERFORMANCE
# ==============================================================================

# Função para calcular métricas
calcular_metricas <- function(real, previsto, nome_modelo) {
  rmse <- sqrt(mean((real - previsto)^2))
  mae <- mean(abs(real - previsto))
  ss_total <- sum((real - mean(real))^2)
  ss_residual <- sum((real - previsto)^2)
  r2 <- 1 - (ss_residual / ss_total)
  
  data.frame(
    Modelo = nome_modelo,
    RMSE = round(rmse, 2),
    MAE = round(mae, 2),
    R2 = round(r2, 4)
  )
}

# Calcular métricas para todos os modelos
metricas_lm <- calcular_metricas(dados_teste$price, pred_lm, "Regressão Linear")
metricas_rf <- calcular_metricas(dados_teste$price, pred_rf, "Random Forest")
metricas_bag <- calcular_metricas(dados_teste$price, pred_bag, "Bagging")
metricas_gbm <- calcular_metricas(dados_teste$price, pred_gbm, "Gradient Boosting")

# Tabela comparativa
tabela_metricas <- rbind(metricas_lm, metricas_rf, metricas_bag, metricas_gbm)
tabela_metricas <- tabela_metricas %>%
  arrange(desc(R2))

cat("\n=== COMPARAÇÃO DE PERFORMANCE DOS MODELOS ===\n")
print(tabela_metricas)

# Identificar melhor modelo
melhor_modelo <- tabela_metricas$Modelo[1]
cat("\n*** Melhor modelo:", melhor_modelo, "***\n")

# --- Visualizações de Performance ---

# Gráfico de comparação de métricas
tabela_metricas_long <- tabela_metricas %>%
  gather(key = "Metrica", value = "Valor", -Modelo)

ggplot(tabela_metricas_long, aes(x = Modelo, y = Valor, fill = Modelo)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Metrica, scales = "free_y") +
  labs(title = "Comparação de Métricas por Modelo") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Gráfico Previsto vs Real para cada modelo
par(mfrow = c(2, 2))

plot(dados_teste$price, pred_lm, 
     main = "Regressão Linear: Previsto vs Real",
     xlab = "Preço Real", ylab = "Preço Previsto",
     col = rgb(0.2, 0.4, 0.6, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

plot(dados_teste$price, pred_rf, 
     main = "Random Forest: Previsto vs Real",
     xlab = "Preço Real", ylab = "Preço Previsto",
     col = rgb(0.2, 0.6, 0.4, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

plot(dados_teste$price, pred_bag, 
     main = "Bagging: Previsto vs Real",
     xlab = "Preço Real", ylab = "Preço Previsto",
     col = rgb(0.6, 0.4, 0.2, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

plot(dados_teste$price, pred_gbm, 
     main = "GBM: Previsto vs Real",
     xlab = "Preço Real", ylab = "Preço Previsto",
     col = rgb(0.6, 0.2, 0.4, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

par(mfrow = c(1, 1))

# Gráfico de resíduos para o melhor modelo
if(melhor_modelo == "Random Forest") {
  residuos <- dados_teste$price - pred_rf
} else if(melhor_modelo == "Gradient Boosting") {
  residuos <- dados_teste$price - pred_gbm
} else if(melhor_modelo == "Bagging") {
  residuos <- dados_teste$price - pred_bag
} else {
  residuos <- dados_teste$price - pred_lm
}

# Histograma de resíduos
ggplot(data.frame(residuos = residuos), aes(x = residuos)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  labs(title = paste("Distribuição de Resíduos -", melhor_modelo),
       x = "Resíduos", y = "Frequência") +
  theme_minimal()

# QQ-plot de resíduos
qqnorm(residuos, main = paste("QQ-Plot de Resíduos -", melhor_modelo))
qqline(residuos, col = "red")

# --- Importância das Variáveis (Random Forest) ---
cat("\n=== IMPORTÂNCIA DAS VARIÁVEIS (RANDOM FOREST) ===\n")
importancia_rf <- varImp(modelo_rf)
print(importancia_rf)
plot(importancia_rf, top = 15, main = "Top 15 Variáveis Mais Importantes - Random Forest")

# ==============================================================================
# 12. RESUMO FINAL
# ==============================================================================

cat("\n")
cat("==============================================================================\n")
cat("                           RESUMO FINAL                                       \n")
cat("==============================================================================\n")
cat("\n")
cat("Dataset: listings_Budapest.csv\n")
cat("Observações utilizadas:", nrow(dados_modelo), "\n")
cat("Variáveis preditoras:", ncol(dados_modelo) - 1, "\n")
cat("\n")
cat("Modelos treinados:\n")
cat("  1. Regressão Linear\n")
cat("  2. Random Forest\n")
cat("  3. Bagging\n")
cat("  4. Gradient Boosting\n")
cat("\n")
cat("Melhor modelo:", melhor_modelo, "\n")
cat("  - RMSE:", tabela_metricas$RMSE[1], "\n")
cat("  - MAE:", tabela_metricas$MAE[1], "\n")
cat("  - R²:", tabela_metricas$R2[1], "\n")
cat("\n")
cat("==============================================================================\n")
