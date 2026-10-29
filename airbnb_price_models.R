# ==============================================================================
# R script for predicting Airbnb listing prices in Budapest
# Methodology: CRISP-DM
# Dataset: listings_Budapest.csv
# ==============================================================================

# ==============================================================================
# 1. SETUP AND PACKAGE LOADING
# ==============================================================================

# Package installation (uncomment if needed)
# install.packages(c("caret", "ggplot2", "dplyr", "corrplot", "randomForest", 
#                    "gbm", "e1071", "ipred", "tidyr"))

# Paths are relative to the folder that contains this script, so it runs from
# wherever the repository was downloaded. The dataset is read from data/.
local({
  dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
  if (is.null(dir)) {
    arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(arg)) dir <- dirname(normalizePath(sub("^--file=", "", arg[1])))
  }
  if (is.null(dir) && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
    dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  if (!is.null(dir) && nzchar(dir)) setwd(dir)
})

# Load packages
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


# Seed for reproducibility
set.seed(123)

# ==============================================================================
# 2. STEP 1: BUSINESS UNDERSTANDING
# ==============================================================================

# Goal: estimate the price of Airbnb listings in Budapest
# 
# Selected predictor variables:
# - neighbourhood: district (location is key to pricing)
# - latitude/longitude: exact geographic position
# - room_type: type of accommodation (entire home, private room, etc.)
# - minimum_nights: minimum stay requirement
# - number_of_reviews: popularity indicator
# - reviews_per_month: booking frequency
# - calculated_host_listings_count: host professionalism
# - availability_365: annual availability
# - number_of_reviews_ltm: recent activity
#
# Excluded variables (not relevant for prediction):
# - id: unique identifier with no predictive value
# - name: unstructured descriptive text
# - host_id: identifier with no predictive value
# - host_name: host's name, not relevant
# - neighbourhood_group: contains only NAs in the Budapest dataset
# - last_review: date of the last review, converted into more useful features
# - license: licence information, not relevant to price

# ==============================================================================
# 3. DATA LOADING AND INITIAL INSPECTION
# ==============================================================================

# Pre-process the file to clean up malformed lines
file_path <- "data/listings_Budapest.csv"

if (file.exists(file_path)) {
  # Read the CSV directly
  dados <- read.csv(file_path, stringsAsFactors = FALSE)
  
  cat("Data loaded successfully!\n")
  cat("Total observations:", nrow(dados), "\n")
} else {
  stop("File not found: ", file_path)
}

# Recover malformed lines
if (file.exists(file_path)) {
  linhas_brutas <- readLines(file_path, warn = FALSE)
  linhas_limpas <- gsub(";;$", "", linhas_brutas)
  linhas_limpas <- linhas_limpas[linhas_limpas != ""]
  linhas_limpas <- gsub('^\"|\"$', "", linhas_limpas)
  linhas_limpas <- gsub('\"\"', '\"', linhas_limpas, fixed = TRUE)
  
  dados <- read_csv(I(linhas_limpas), show_col_types = FALSE)
  message("Data loaded successfully! Total rows: ", nrow(dados))
} else {
  stop("ERROR: File not found.")
}

# Initial inspection
cat("=== DATASET DIMENSIONS ===\n")
print(dim(dados))

cat("\n=== DATA STRUCTURE ===\n")
str(dados)

cat("\n=== FIRST ROWS ===\n")
head(dados)

cat("\n=== SUMMARY STATISTICS ===\n")
summary(dados)

# Remove rows with price = 0 or NA
dados <- dados %>%
  filter(!is.na(price) & price > 0)

dados <- dados %>%
  select(-any_of(c("neighbourhood_group")))

dados$neighbourhood <- as.factor(dados$neighbourhood)
dados$room_type <- as.factor(dados$room_type)
dados$host_id <- as.character(dados$host_id)
dados$id <- as.character(dados$id)
dados$price <- dados$price*0.0026

# Number of days since the last review
data_referencia <- max(dados$last_review, na.rm = TRUE)
dados$days_since_last_review <- as.numeric(data_referencia - dados$last_review)
summary(dados)



#dados <- dados %>%
#  filter(price < 31475828) %>%
#  filter(minimum_nights <300)


# Exploratory subsets (not used after the analysis)
dados_longas_estadias <- dados %>%
  filter(minimum_nights > 30)
summary(dados_longas_estadias)

dados_sem_reviews <- dados %>%
  filter(number_of_reviews == 0)
summary(dados_sem_reviews)

dados_sem_avail <- dados %>%
  filter(availability_365 == 0)
summary(dados_sem_avail)

dados_preco_alto <- dados %>%
  filter(price > 150)
summary(dados_preco_alto)

dados_anuncios_inativos <- dados %>%
  filter(days_since_last_review >365)
summary(dados_anuncios_inativos)




# ==============================================================================
# 4. STEP 2: DATA CLEANING
# ==============================================================================

cat("\n=== MISSING VALUES BEFORE CLEANING ===\n")
print(colSums(is.na(dados)))

# Drop irrelevant columns (any_of avoids errors if a column does not exist)
dados_limpos <- dados %>%
  select(-any_of(c("id", "name", "host_id", "host_name", "neighbourhood_group", 
                   "last_review", "license")))

# Convert numeric columns (they may have been read as text)
dados_limpos$price <- as.numeric(dados_limpos$price)
dados_limpos$latitude <- as.numeric(dados_limpos$latitude)
dados_limpos$longitude <- as.numeric(dados_limpos$longitude)
dados_limpos$minimum_nights <- as.numeric(dados_limpos$minimum_nights)
dados_limpos$number_of_reviews <- as.numeric(dados_limpos$number_of_reviews)
dados_limpos$reviews_per_month <- as.numeric(dados_limpos$reviews_per_month)
dados_limpos$calculated_host_listings_count <- as.numeric(dados_limpos$calculated_host_listings_count)
dados_limpos$availability_365 <- as.numeric(dados_limpos$availability_365)
dados_limpos$number_of_reviews_ltm <- as.numeric(dados_limpos$number_of_reviews_ltm)

# Handle missing values in reviews_per_month (replace with 0)
dados_limpos$reviews_per_month[is.na(dados_limpos$reviews_per_month)] <- 0



cat("\n=== MISSING VALUES AFTER TREATMENT ===\n")
print(colSums(is.na(dados_limpos)))

# Outlier treatment for price using the IQR rule
Q1 <- quantile(dados_limpos$price, 0.25)
Q3 <- quantile(dados_limpos$price, 0.75)
IQR <- Q3 - Q1
limite_inferior <- Q1 - 1.5 * IQR
limite_superior <- Q3 + 1.5 * IQR

print(limite_inferior)
print(limite_superior)

cat("\n=== OUTLIER ANALYSIS (PRICE) ===\n")
cat("Q1:", Q1, "\n")
cat("Q3:", Q3, "\n")
cat("IQR:", IQR, "\n")
cat("Lower bound:", limite_inferior, "\n")
cat("Upper bound:", limite_superior, "\n")

n_antes <- nrow(dados_limpos)
dados_limpos <- dados_limpos %>%                                  
  filter(price >= limite_inferior & price <= limite_superior)
n_depois <- nrow(dados_limpos)




cat("Observations before:", n_antes, "\n")
cat("Observations after outlier removal:", n_depois, "\n")
cat("Outliers removed:", n_antes - n_depois, "\n")

# ==============================================================================
# 5. STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

cat("\n=== DESCRIPTIVE STATISTICS AFTER CLEANING ===\n")
summary(dados_limpos)

# Price histogram
ggplot(dados_limpos, aes(x = price)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Price Distribution - Airbnb Budapest",
       x = "Price (EUR)", y = "Frequency") +
  theme_minimal()


# High-price histogram #############################################################################################
ggplot(dados_preco_alto, aes(x = price)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "High-Price Distribution - Airbnb Budapest",
       x = "Price (EUR)", y = "Frequency") +
  theme_minimal()

# Boxplot by room_type
ggplot(dados_limpos, aes(x = room_type, y = price, fill = room_type)) +
  geom_boxplot() +
  labs(title = "Price by Room Type",
       x = "Room Type", y = "Price (EUR)") +
  theme_minimal() +
  theme(legend.position = "none")

# Count by room_type
ggplot(dados_limpos, aes(x = room_type, fill = room_type)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(title = "Number of Listings by Room Type",
       x = "Room Type", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# Boxplot by neighbourhood (top 10 districts)
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
  labs(title = "Price by District (Top 10)",
       x = "District", y = "Price (EUR)") +
  theme_minimal() +
  theme(legend.position = "none")



# Scatter plot: latitude vs longitude (price map)
ggplot(dados_limpos, aes(x = longitude, y = latitude, color = price)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Geographic Distribution of Prices",
       x = "Longitude", y = "Latitude", color = "Price (EUR)") +
  theme_minimal()

# Distribution of availability_365
ggplot(dados_limpos, aes(x = availability_365)) +
  geom_histogram(bins = 50, fill = "coral", color = "white") +
  labs(title = "Annual Availability Distribution",
       x = "Days Available (out of 365)", y = "Frequency") +
  theme_minimal()

# Scatter plot: reviews_per_month vs price
ggplot(dados_limpos, aes(x = reviews_per_month, y = price)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Reviews per Month vs Price",
       x = "Reviews per Month", y = "Price (EUR)") +
  theme_minimal()

# Histograms of the main numeric variables
dados_limpos %>%
  select(price, minimum_nights, number_of_reviews, reviews_per_month, 
         availability_365) %>%
  gather(key = "variavel", value = "valor") %>%
  ggplot(aes(x = valor)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~variavel, scales = "free") +
  labs(title = "Distribution of Numeric Variables") +
  theme_minimal()

# ==============================================================================
# 6. STEP 4: CORRELATION ANALYSIS
# ==============================================================================

# Select numeric variables only
variaveis_numericas <- dados_limpos %>%
  select(price, latitude, longitude, minimum_nights, number_of_reviews,
         reviews_per_month, calculated_host_listings_count, 
         availability_365, number_of_reviews_ltm)

# Compute the correlation matrix
matriz_correlacao <- cor(variaveis_numericas, use = "complete.obs")

cat("\n=== CORRELATION MATRIX ===\n")
print(round(matriz_correlacao, 3))

# Correlation heatmap
corrplot(matriz_correlacao, 
         method = "color",
         type = "upper",
         addCoef.col = "black",
         number.cex = 0.6,
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 0.7,
         cl.cex = 0.7,
         title = "Correlation Matrix",
         mar = c(0, 0, 2, 2),
         col = colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))(200),
         diag = FALSE)

# Correlations with price
cat("\n=== CORRELATIONS WITH PRICE ===\n")
correlacoes_price <- matriz_correlacao["price", ]
correlacoes_price <- sort(correlacoes_price, decreasing = TRUE)
print(round(correlacoes_price, 3))

# ==============================================================================
# 7. STEP 5: PRE-PROCESSING
# ==============================================================================

# Feature engineering: distance to a centre derived from the data
# The centre is computed as the mean of the dataset's coordinates
centro_lat <- mean(dados_limpos$latitude, na.rm = TRUE)
centro_lon <- mean(dados_limpos$longitude, na.rm = TRUE)

cat("\n=== CENTRE COMPUTED FROM THE DATA ===\n")
cat("Mean latitude (centre):", centro_lat, "\n")
cat("Mean longitude (centre):", centro_lon, "\n")


#centro_lat <- 47.50382
#centro_lon <- 19.0625

dados_limpos <- dados_limpos %>%
  mutate(
    dist_centro = sqrt((latitude - centro_lat)^2 + (longitude - centro_lon)^2)
  )

# Visualise the new feature
ggplot(dados_limpos, aes(x = dist_centro, y = price)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Distance to Centre vs Price",
       x = "Distance to Centre (computed)", y = "Price (EUR)") +
  theme_minimal()

# Encoding categorical variables as dummies
cat("\n=== CATEGORICAL VARIABLE ENCODING ===\n")#################################################################################################

# Create dummies for room_type
#dummies_room <- model.matrix(~ room_type - 1, data = dados_limpos)
#colnames(dummies_room) <- gsub("room_type", "room_", colnames(dummies_room))

# Create dummies for neighbourhood (top 10, the rest as "other")
#dados_limpos <- dados_limpos %>%
#  mutate(neighbourhood_grouped = ifelse(
#    neighbourhood %in% top_neighbourhoods, 
#    neighbourhood, 
#    "Other"
#  ))

#dummies_neighbourhood <- model.matrix(~ neighbourhood_grouped - 1, data = dados_limpos)
#colnames(dummies_neighbourhood) <- gsub("neighbourhood_grouped", "neigh_", 
#                                         colnames(dummies_neighbourhood))

# Prepare the final modelling dataset
dados_modelo <- dados_limpos %>%
  select(price, latitude, longitude, minimum_nights, number_of_reviews,
         reviews_per_month, calculated_host_listings_count, 
         availability_365, number_of_reviews_ltm, dist_centro, neighbourhood, room_type)###################################################################

# Add dummies
#dados_modelo <- cbind(dados_modelo, dummies_room, dummies_neighbourhood)

# Clean column names (remove spaces and special characters)
colnames(dados_modelo) <- make.names(colnames(dados_modelo))

cat("Modelling dataset dimensions:", dim(dados_modelo), "\n")
cat("Variables included:\n")
print(names(dados_modelo))

# ==============================================================================
# 8. STEP 6: TRAIN/TEST SPLIT
# ==============================================================================

set.seed(123)

# Create indices for an 80/20 split
indices_treino <- createDataPartition(dados_modelo$price, p = 0.8, list = FALSE)

# Split the data
dados_treino <- dados_modelo[indices_treino, ]
dados_teste <- dados_modelo[-indices_treino, ]

# Force conversion to data.frame (extra safeguard)
dados_treino <- as.data.frame(dados_treino)
dados_teste <- as.data.frame(dados_teste)

cat("\n=== TRAINING DATA STRUCTURE ===\n")
str(dados_treino)

cat("\n=== TRAIN/TEST SPLIT ===\n")
cat("Training observations:", nrow(dados_treino), "\n")
cat("Test observations:", nrow(dados_teste), "\n")

# Check the price distribution in both sets
cat("\nPrice distribution in the training set:\n")
print(summary(dados_treino$price))
cat("\nPrice distribution in the test set:\n")
print(summary(dados_teste$price))

# ==============================================================================
# 9. STEP 7: MODEL TRAINING
# ==============================================================================

# Set up parallel processing
cl <- makePSOCKcluster(detectCores() - 1) # Use all cores but one
registerDoParallel(cl)
cat("Parallel processing started with", getDoParWorkers(), "cores.\n")

# 10-fold cross-validation setup
ctrl <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = FALSE
)

cat("\n=== MODEL TRAINING ===\n")

# --- Model 1: Linear Regression ---
cat("\n--- Training Linear Regression ---\n")
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

# --- Model 2: Random Forest ---
cat("\n--- Training Random Forest ---\n")
modelo_rf <- train(
  price ~ .,
  data = dados_treino,
  method = "rf",
  trControl = ctrl,
#  tuneLength = 3,  # Try 3 values of mtry
  tuneGrid = mtry_grid,
  ntree = 100
)
print(modelo_rf)

# --- Model 3: Bagging ---
cat("\n--- Training Bagging ---\n")
modelo_bag <- train(
  price ~ .,
  data = dados_treino,
  method = "treebag",
  trControl = ctrl,
  nbagg = 100
)
print(modelo_bag)

# --- Model 4: Gradient Boosting ---
cat("\n--- Training Gradient Boosting ---\n")
modelo_gbm <- train(
  price ~ .,
  data = dados_treino,
  method = "gbm",
  trControl = ctrl,
  verbose = FALSE,
  tuneLength = 3
)
print(modelo_gbm)

# Stop the parallel processing cluster
stopCluster(cl)
registerDoSEQ()
cat("Parallel processing finished.\n")

# ==============================================================================
# 10. STEP 8: VALIDATION (TEST SET PREDICTIONS)
# ==============================================================================

cat("\n=== TEST SET PREDICTIONS ===\n")

# Predictions for each model
pred_lm <- predict(modelo_lm, newdata = dados_teste)
pred_rf <- predict(modelo_rf, newdata = dados_teste)
pred_bag <- predict(modelo_bag, newdata = dados_teste)
pred_gbm <- predict(modelo_gbm, newdata = dados_teste)

# Build a data frame of predictions
resultados_previsoes <- data.frame(
  Real = dados_teste$price,
  LM = pred_lm,
  RF = pred_rf,
  Bagging = pred_bag,
  GBM = pred_gbm
)

cat("\nFirst predictions:\n")
print(head(resultados_previsoes))

# ==============================================================================
# 11. STEP 9: PERFORMANCE EVALUATION
# ==============================================================================

# Function to compute evaluation metrics
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

# Compute metrics for all models
metricas_lm <- calcular_metricas(dados_teste$price, pred_lm, "Linear Regression")
metricas_rf <- calcular_metricas(dados_teste$price, pred_rf, "Random Forest")
metricas_bag <- calcular_metricas(dados_teste$price, pred_bag, "Bagging")
metricas_gbm <- calcular_metricas(dados_teste$price, pred_gbm, "Gradient Boosting")

# Comparison table
tabela_metricas <- rbind(metricas_lm, metricas_rf, metricas_bag, metricas_gbm)
tabela_metricas <- tabela_metricas %>%
  arrange(desc(R2))

cat("\n=== MODEL PERFORMANCE COMPARISON ===\n")
print(tabela_metricas)

# Identify the best model
melhor_modelo <- tabela_metricas$Modelo[1]
cat("\n*** Best model:", melhor_modelo, "***\n")

# --- Performance Visualisations ---

# Metric comparison chart
tabela_metricas_long <- tabela_metricas %>%
  gather(key = "Metrica", value = "Valor", -Modelo)

ggplot(tabela_metricas_long, aes(x = Modelo, y = Valor, fill = Modelo)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Metrica, scales = "free_y") +
  labs(title = "Metric Comparison by Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Predicted vs actual plot for each model
par(mfrow = c(2, 2))

plot(dados_teste$price, pred_lm, 
     main = "Linear Regression: Predicted vs Actual",
     xlab = "Actual Price", ylab = "Predicted Price",
     col = rgb(0.2, 0.4, 0.6, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

plot(dados_teste$price, pred_rf, 
     main = "Random Forest: Predicted vs Actual",
     xlab = "Actual Price", ylab = "Predicted Price",
     col = rgb(0.2, 0.6, 0.4, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

plot(dados_teste$price, pred_bag, 
     main = "Bagging: Predicted vs Actual",
     xlab = "Actual Price", ylab = "Predicted Price",
     col = rgb(0.6, 0.4, 0.2, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

plot(dados_teste$price, pred_gbm, 
     main = "GBM: Predicted vs Actual",
     xlab = "Actual Price", ylab = "Predicted Price",
     col = rgb(0.6, 0.2, 0.4, 0.5), pch = 16)
abline(0, 1, col = "red", lwd = 2)

par(mfrow = c(1, 1))

# Residuals for the best model
if(melhor_modelo == "Random Forest") {
  residuos <- dados_teste$price - pred_rf
} else if(melhor_modelo == "Gradient Boosting") {
  residuos <- dados_teste$price - pred_gbm
} else if(melhor_modelo == "Bagging") {
  residuos <- dados_teste$price - pred_bag
} else {
  residuos <- dados_teste$price - pred_lm
}

# Residual histogram
ggplot(data.frame(residuos = residuos), aes(x = residuos)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  labs(title = paste("Residual Distribution -", melhor_modelo),
       x = "Residuals", y = "Frequency") +
  theme_minimal()

# Residual QQ plot
qqnorm(residuos, main = paste("Residual QQ Plot -", melhor_modelo))
qqline(residuos, col = "red")

# --- Variable Importance (Random Forest) ---
cat("\n=== VARIABLE IMPORTANCE (RANDOM FOREST) ===\n")
importancia_rf <- varImp(modelo_rf)
print(importancia_rf)
plot(importancia_rf, top = 15, main = "Top 15 Most Important Variables - Random Forest")

# ==============================================================================
# 12. FINAL SUMMARY
# ==============================================================================

cat("\n")
cat("==============================================================================\n")
cat("                           FINAL SUMMARY                                      \n")
cat("==============================================================================\n")
cat("\n")
cat("Dataset: listings_Budapest.csv\n")
cat("Observations used:", nrow(dados_modelo), "\n")
cat("Predictor variables:", ncol(dados_modelo) - 1, "\n")
cat("\n")
cat("Models trained:\n")
cat("  1. Linear Regression\n")
cat("  2. Random Forest\n")
cat("  3. Bagging\n")
cat("  4. Gradient Boosting\n")
cat("\n")
cat("Best model:", melhor_modelo, "\n")
cat("  - RMSE:", tabela_metricas$RMSE[1], "\n")
cat("  - MAE:", tabela_metricas$MAE[1], "\n")
cat("  - R²:", tabela_metricas$R2[1], "\n")
cat("\n")
cat("==============================================================================\n")
