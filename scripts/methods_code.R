# =============================================================================
# Rolling Window Forecasting: RW, PCR, DFM, Ridge, Lasso, Elastic Net,
#                             3PRF, PCLAR, PCLAS
#
# Authors: Yuxin Zheng, Gulnara Sadykova, Yipeng Yan
# Course:  Machine Learning for Economists, University of Bologna, 2025
# =============================================================================

rm(list=ls())
cat('\014')

#Load libraries 
library(readr)
library(dplyr)
library(ggplot2)
library(glmnet)
library(tidyr)
library(scales)
library(knitr)
set.seed(12345)

#Load data
input  <- "data/stationary_data.csv"
output_dir  <- "output"
data <- read_csv(input, name_repair = "minimal")
names(data)
#Set y and x
Y <- data$Y_WTI_Price
X <- data[, !(names(data) %in% c("date", "Y_WTI_Price"))]
month <- data$date

#Calculate correlations
correlations_with_Y <- cor(as.matrix(X), Y)
abs_cors <- abs(correlations_with_Y[, 1])

#Set threshold
cor_threshold <- 0.90

#Identify problematic variables
high_cor_idx <- which(abs_cors > cor_threshold)
high_cor_names <- names(abs_cors)[high_cor_idx]
high_cor_values <- abs_cors[high_cor_idx]
X <- X[, abs_cors <= cor_threshold]
high_cor_names    
high_cor_values  
# Variables X_FRED_6...104 (OilPriceX) and X_POILBREUSDM (Brent Crude Price)
# are highly correlated (|r| > 0.9) with the target Y (WTI price)

#Parameters
roll_window <- 100   # length
n_lag <- 6          # For 3PRF: use LagY1~LagY6 as proxy factors

cat("========================================\n")
cat("   ROLLING WINDOW FORECASTING SETUP    \n")
cat("   (Using Bai & Ng IC for Factor Selection)\n")
cat("========================================\n")
cat(sprintf("Total observations: %d\n", length(Y)))
cat(sprintf("Window size: %d\n", roll_window))
cat(sprintf("Number of predictors: %d\n", ncol(X)))
cat(sprintf("Number of forecasts: %d\n\n", length(Y) - roll_window))


#Descriptive statistics of Y
library(moments)

summary_Y <- data.frame(
  Mean = mean(Y, na.rm = TRUE),
  Median = median(Y, na.rm = TRUE),
  SD = sd(Y, na.rm = TRUE),
  Min = min(Y, na.rm = TRUE),
  Max = max(Y, na.rm = TRUE),
  Skewness = moments::skewness(Y, na.rm = TRUE),
  Kurtosis = moments::kurtosis(Y, na.rm = TRUE)
)

p_Y <- ggplot(data.frame(Date = month, Y = Y), aes(x = Date, y = Y)) +
  geom_line(color = "steelblue", linewidth = 0.8) +
  labs(subtitle = "Monthly time series of crude oil price (WTI) changes", 
       x = "Date", y = "ΔWTI (USD per barrel)") +
  theme_minimal(base_size = 14) 

print(p_Y)


#PREPARE DATA FOR 3PRF (needs lagged Y) 
# Create lagged Y for 3PRF (6 lags)
Z_all <- as.data.frame(sapply(1:n_lag, function(l) dplyr::lag(Y, l)))
colnames(Z_all) <- paste0("LagY", 1:n_lag)
valid_idx <- complete.cases(Z_all)
Z_all <- Z_all[valid_idx, ]
Y_valid <- Y[valid_idx]
X_valid <- X[valid_idx, ]  # 
month_valid <- month[valid_idx]

# For PCLAR/PCLAS (only need 1 lag)
Y_lag1 <- dplyr::lag(Y, 1)
valid_idx_lag1 <- complete.cases(Y_lag1)
Y_valid_lag1 <- Y[valid_idx_lag1]
Y_lag1_valid <- Y_lag1[valid_idx_lag1]
X_valid_lag1 <- X[valid_idx_lag1, ]  # 
month_valid_lag1 <- month[valid_idx_lag1]

#INITIALIZE STORAGE FOR ALL MODELS
n_predictions <- length(Y) - roll_window
n_predictions_3prf <- length(Y_valid) - roll_window
n_predictions_pclar <- length(Y_valid_lag1) - roll_window

# Models without lags (RW, PCR, DFM, Ridge, Lasso, ENet)
Y_hat_rw <- numeric(n_predictions)
Y_hat_pcr <- numeric(n_predictions)
Y_hat_dfm <- numeric(n_predictions)
Y_hat_ridge <- numeric(n_predictions)
Y_hat_lasso <- numeric(n_predictions)
Y_hat_enet <- numeric(n_predictions)

# 3PRF (with 6 lags)
Y_hat_3prf <- numeric(n_predictions_3prf)

# PCLAR/PCLAS (with 1 lag)
Y_hat_pclar <- numeric(n_predictions_pclar)
Y_hat_pclas <- numeric(n_predictions_pclar)

# Track non-zero coefficients
non_zero_lasso <- numeric(n_predictions)
non_zero_enet <- numeric(n_predictions)

# Bai & Ng IC function
IC_p2 <- function(q, N, T, V_k) log(V_k) + q * ((N + T) / (N * T)) * log((N * T) / (N + T))


# Function: select_lambda_BIC
# Chooses the lambda that minimizes the Bayesian Information Criterion
# BIC_λ = n * log(MSE_λ) + df_λ * log(n)
# where df_λ is the effective degrees of freedom from glmnet.
select_lambda_BIC <- function(X, y, alpha = 0) {
  # Fit glmnet path
  fit <- glmnet(X, y, alpha = alpha, standardize = FALSE)
  n <- nrow(X)
  
  # Predicted values for all lambdas
  Yhat <- predict(fit, newx = X)
  
  # Compute MSE for each lambda
  mse <- colMeans((matrix(y, n, length(fit$lambda)) - Yhat)^2)
  
  # Effective degrees of freedom (from glmnet)
  df <- fit$df
  
  # Compute BIC values
  bic_vals <- n * log(mse) + df * log(n)
  
  # Identify best lambda
  best_idx <- which.min(bic_vals)
  list(best_lambda = fit$lambda[best_idx],
       fit = fit,
       bic = bic_vals,
       best_idx = best_idx)
}

#ROLLING WINDOW LOOP - MAIN MODELS (RW, PCR, DFM, Ridge, Lasso, ENet)
for (t_start in 1:n_predictions) {
  t_end <- t_start + roll_window - 1
  
  #1. Prepare training and test data (raw)
  X_train_raw <- X[t_start:t_end, ]
  Y_train <- Y[t_start:t_end]
  X_next_raw <- X[t_end + 1, , drop = FALSE]
  
  #2. Standardize predictors using training sample only 
  train_mean <- apply(X_train_raw, 2, mean)
  train_sd <- apply(X_train_raw, 2, sd)
  train_sd[train_sd == 0] <- 1  # avoid division by zero for constant columns
  
  X_train <- scale(X_train_raw, center = train_mean, scale = train_sd)
  X_next  <- scale(X_next_raw,  center = train_mean, scale = train_sd)
  
  #3. Random Walk (RW)
  Y_hat_rw[t_start] <- Y_train[length(Y_train)]
  
  #4. PCA setup with Bai & Ng IC (for PCR and DFM)
  pca_result <- prcomp(X_train, center = TRUE, scale. = TRUE)
  
  N <- ncol(X_train)
  Tn <- nrow(X_train)
  k_max <- min(20, floor(Tn / 5))
  V_k <- numeric(k_max)
  
  for (q in 1:k_max) {
    F_k <- pca_result$x[, 1:q, drop = FALSE]
    Lambda_k <- pca_result$rotation[, 1:q, drop = FALSE]
    X_reconstructed <- F_k %*% t(Lambda_k)
    V_k[q] <- sum((X_train - X_reconstructed)^2) / (N * Tn)
  }
  
  IC_values <- sapply(1:k_max, function(q) IC_p2(q, N, Tn, V_k[q]))
  num_factors <- which.min(IC_values)
  
  train_factors <- as.data.frame(pca_result$x[, 1:num_factors])
  colnames(train_factors) <- paste0("PC", 1:num_factors)
  
  #5. Principal Component Regression (PCR) 
  pcr_data <- cbind(y = Y_train, train_factors)
  pcr_model <- lm(y ~ ., data = pcr_data)
  
  test_factors_pcr <- as.data.frame(predict(pca_result, newdata = X_next))[, 1:num_factors, drop = FALSE]
  colnames(test_factors_pcr) <- paste0("PC", 1:num_factors)
  Y_hat_pcr[t_start] <- predict(pcr_model, newdata = test_factors_pcr)
  
  #6. Dynamic Factor Model (DFM) 
  dfm_data <- data.frame(
    y = Y_train,
    y_lag1 = dplyr::lag(Y_train, 1)
  )
  dfm_data <- cbind(dfm_data, train_factors)
  dfm_data <- na.omit(dfm_data)
  
  dfm_model <- lm(y ~ ., data = dfm_data)
  y_lag_for_forecast <- Y_train[length(Y_train)]
  
  test_factors_dfm <- as.data.frame(predict(pca_result, newdata = X_next))[, 1:num_factors, drop = FALSE]
  colnames(test_factors_dfm) <- paste0("PC", 1:num_factors)
  dfm_newdata <- data.frame(y_lag1 = y_lag_for_forecast)
  dfm_newdata <- cbind(dfm_newdata, test_factors_dfm)
  Y_hat_dfm[t_start] <- predict(dfm_model, newdata = dfm_newdata)
  
  #7. Ridge Regression
  ridge_sel <- select_lambda_BIC(X_train, Y_train, alpha = 0)
  best_lambda_ridge <- ridge_sel$best_lambda
  ridge_fit <- ridge_sel$fit
  Y_hat_ridge[t_start] <- as.numeric(predict(ridge_fit, newx = X_next, s = best_lambda_ridge))
  
  #8. Lasso Regression 
  lasso_sel <- select_lambda_BIC(X_train, Y_train, alpha = 1)
  best_lambda_lasso <- lasso_sel$best_lambda
  lasso_fit <- lasso_sel$fit
  Y_hat_lasso[t_start] <- as.numeric(predict(lasso_fit, newx = X_next, s = best_lambda_lasso))
  non_zero_lasso[t_start] <- lasso_fit$df[lasso_sel$best_idx]
  
  #9. Elastic Net 
  alpha_enet <- 0.5
  enet_sel <- select_lambda_BIC(X_train, Y_train, alpha = alpha_enet)
  best_lambda_enet <- enet_sel$best_lambda
  enet_fit <- enet_sel$fit
  Y_hat_enet[t_start] <- as.numeric(predict(enet_fit, newx = X_next, s = best_lambda_enet))
  non_zero_enet[t_start] <- enet_fit$df[enet_sel$best_idx]
  
  #Progress indicator
  if (t_start %% 50 == 0) {
    cat(sprintf("Main models: Completed %d/%d\n", t_start, n_predictions))
  }
}
 

#ROLLING WINDOW LOOP - 3PRF MODEL (FIXED)
for (t_start in 1:n_predictions_3prf) {
  t_end <- t_start + roll_window - 1
  
  #Get RAW data 
  X_win_raw <- as.matrix(X_valid[t_start:t_end, ])
  Y_win <- Y_valid[t_start:t_end]
  Z_win <- Z_all[t_start:t_end, ]
  X_next_raw <- as.matrix(X_valid[t_end + 1, , drop = FALSE])
  
  #Standardize using training window only
  win_mean <- apply(X_win_raw, 2, mean)
  win_sd <- apply(X_win_raw, 2, sd)
  win_sd[win_sd == 0] <- 1
  
  X_win <- scale(X_win_raw, center = win_mean, scale = win_sd)
  X_next_3prf <- scale(X_next_raw, center = win_mean, scale = win_sd)
  
  #First pass (3PRF) 
  N <- ncol(X_win)
  k <- ncol(Z_win)
  phi_hat <- matrix(NA, nrow = N, ncol = k)
  
  for (i in 1:N) {
    model_i <- lm(X_win[, i] ~ as.matrix(Z_win))
    phi_hat[i, ] <- coef(model_i)[-1]
  }
  
  #Second pass 
  Tn <- nrow(X_win)
  F_hat <- matrix(NA, nrow = Tn, ncol = k)
  
  for (tt in 1:Tn) {
    model_tt <- lm(as.numeric(X_win[tt, ]) ~ as.matrix(phi_hat))
    F_hat[tt, ] <- coef(model_tt)[-1]
  }
  
  #PCA to supplement factors using Bai & Ng IC 
  pca_win <- prcomp(X_win, center = TRUE, scale. = TRUE)
  k_max_3prf <- min(20, floor(Tn / 5))
  V_k_3prf <- numeric(k_max_3prf)
  
  for (q in 1:k_max_3prf) {
    F_k <- pca_win$x[, 1:q, drop = FALSE]
    Lambda_k <- pca_win$rotation[, 1:q, drop = FALSE]
    X_reconstructed <- F_k %*% t(Lambda_k)
    V_k_3prf[q] <- sum((X_win - X_reconstructed)^2) / (N * Tn)
  }
  
  IC_values_3prf <- sapply(1:k_max_3prf, function(q) IC_p2(q, N, Tn, V_k_3prf[q]))
  r_opt <- which.min(IC_values_3prf)
  r_pca <- max(0, r_opt - ncol(F_hat))
  
  if (r_pca > 0) {
    F_pca <- pca_win$x[, 1:r_pca]
    F_hat_reduced <- cbind(F_hat, F_pca)
  } else {
    F_hat_reduced <- F_hat
  }
  
  #Target regression 
  Y_lead <- dplyr::lead(Y_win, 1)
  Y_train_3prf <- Y_lead[1:nrow(F_hat_reduced)]
  F_train <- F_hat_reduced[1:length(Y_train_3prf), ]
  valid <- !is.na(Y_train_3prf)
  Y_train_3prf <- Y_train_3prf[valid]
  F_train <- F_train[valid, ]
  model_final <- lm(Y_train_3prf ~ F_train)
  
  #Predict next point 
  # Project X_next onto phi_hat to get targeted factors
  F_next <- as.numeric(X_next_3prf %*% phi_hat %*% solve(t(phi_hat) %*% phi_hat))
  
  # Add PCA factors if needed
  if (r_pca > 0) {
    # X_next_3prf is already standardized with training window stats
    F_pca_next <- X_next_3prf %*% pca_win$rotation[, 1:r_pca, drop = FALSE]
    F_next <- c(F_next, F_pca_next)
  }
  
  Y_hat_3prf[t_start] <- c(1, F_next) %*% coef(model_final)
  
  if (t_start %% 50 == 0) {
    cat(sprintf("3PRF: Completed %d/%d\n", t_start, n_predictions_3prf))
  }
}

#ROLLING WINDOW LOOP - PCLAR & PCLAS (FIXED)

for (t_start in 1:n_predictions_pclar) {
  t_end <- t_start + roll_window - 1
  
  #Get RAW data 
  X_win_raw <- as.matrix(X_valid_lag1[t_start:t_end, ])
  Y_win <- Y_valid_lag1[t_start:t_end]
  Y_lag1_win <- Y_lag1_valid[t_start:t_end]
  X_next_raw <- as.matrix(X_valid_lag1[t_end + 1, , drop = FALSE])
  Y_lag1_next <- Y_valid_lag1[t_end]
  
  #Standardize using training window only
  win_mean <- apply(X_win_raw, 2, mean)
  win_sd <- apply(X_win_raw, 2, sd)
  win_sd[win_sd == 0] <- 1
  
  X_win <- scale(X_win_raw, center = win_mean, scale = win_sd)
  X_next_pc <- scale(X_next_raw, center = win_mean, scale = win_sd)
  
  N <- ncol(X_win)
  Tn <- nrow(X_win)
  
  #PCLAR: Use all predictors 
  pca_full <- prcomp(X_win, center = FALSE, scale. = FALSE)
  k_max <- min(20, floor(min(N, Tn) / 2))
  V_k <- numeric(k_max)
  
  for (k in 1:k_max) {
    F_k <- pca_full$x[, 1:k, drop = FALSE]
    Lambda_k <- pca_full$rotation[, 1:k, drop = FALSE]
    X_reconstructed <- F_k %*% t(Lambda_k)
    V_k[k] <- sum((X_win - X_reconstructed)^2) / (N * Tn)
  }
  
  IC_values <- sapply(1:k_max, function(k) IC_p2(k, N, Tn, V_k[k]))
  r_pclar <- which.min(IC_values)
  
  F_pclar <- pca_full$x[, 1:r_pclar, drop = FALSE]
  data_pclar <- data.frame(Y = Y_win, Y_lag1 = Y_lag1_win, F_pclar)
  model_pclar <- lm(Y ~ ., data = data_pclar)
  
  F_next_pclar <- as.numeric(X_next_pc %*% pca_full$rotation[, 1:r_pclar, drop = FALSE])
  newdata_pclar <- data.frame(Y_lag1 = Y_lag1_next, matrix(F_next_pclar, nrow = 1))
  colnames(newdata_pclar)[-1] <- colnames(F_pclar)
  Y_hat_pclar[t_start] <- predict(model_pclar, newdata = newdata_pclar)
  
  #PCLAS: Variable selection 
  correlations <- cor(X_win, Y_win)
  cor_abs <- abs(correlations[, 1])
  top_k_idx <- order(cor_abs, decreasing = TRUE)[1:min(30, N)]
  
  X_subset <- X_win[, top_k_idx, drop = FALSE]
  X_next_subset <- X_next_pc[top_k_idx]
  N_sub <- ncol(X_subset)
  
  pca_subset <- prcomp(X_subset, center = FALSE, scale. = FALSE)
  k_max_sub <- min(10, floor(N_sub / 2))
  V_k_sub <- numeric(k_max_sub)
  
  for (k in 1:k_max_sub) {
    F_k_sub <- pca_subset$x[, 1:k, drop = FALSE]
    Lambda_k_sub <- pca_subset$rotation[, 1:k, drop = FALSE]
    X_reconstructed_sub <- F_k_sub %*% t(Lambda_k_sub)
    V_k_sub[k] <- sum((X_subset - X_reconstructed_sub)^2) / (N_sub * Tn)
  }
  
  IC_values_sub <- sapply(1:k_max_sub, function(k) IC_p2(k, N_sub, Tn, V_k_sub[k]))
  r_pclas <- which.min(IC_values_sub)
  
  F_pclas <- pca_subset$x[, 1:r_pclas, drop = FALSE]
  data_pclas <- data.frame(Y = Y_win, Y_lag1 = Y_lag1_win, F_pclas)
  model_pclas <- lm(Y ~ ., data = data_pclas)
  
  F_next_pclas <- as.numeric(X_next_subset %*% pca_subset$rotation[, 1:r_pclas, drop = FALSE])
  newdata_pclas <- data.frame(Y_lag1 = Y_lag1_next, matrix(F_next_pclas, nrow = 1))
  colnames(newdata_pclas)[-1] <- colnames(F_pclas)
  Y_hat_pclas[t_start] <- predict(model_pclas, newdata = newdata_pclas)
  
  if (t_start %% 50 == 0) {
    cat(sprintf("PCLAR/PCLAS: Completed %d/%d\n", t_start, n_predictions_pclar))
  }
}

#EVALUATE ALL MODELS
# True values for each set
Y_true_main <- Y[(roll_window + 1):length(Y)]
Y_true_3prf <- Y_valid[(roll_window + 1):length(Y_valid)]
Y_true_pclar <- Y_valid_lag1[(roll_window + 1):length(Y_valid_lag1)]

# Function to calculate metrics
calc_metrics <- function(y_true, y_pred) {
  resid <- y_true - y_pred
  list(
    MSE = mean(resid^2),
    RMSE = sqrt(mean(resid^2)),
    MAE = mean(abs(resid)),
    R2 = 1 - sum(resid^2) / sum((y_true - mean(y_true))^2)
  )
}

# Calculate for all models
metrics_rw <- calc_metrics(Y_true_main, Y_hat_rw)
metrics_pcr <- calc_metrics(Y_true_main, Y_hat_pcr)
metrics_dfm <- calc_metrics(Y_true_main, Y_hat_dfm)
metrics_ridge <- calc_metrics(Y_true_main, Y_hat_ridge)
metrics_lasso <- calc_metrics(Y_true_main, Y_hat_lasso)
metrics_enet <- calc_metrics(Y_true_main, Y_hat_enet)
metrics_3prf <- calc_metrics(Y_true_3prf, Y_hat_3prf)
metrics_pclar <- calc_metrics(Y_true_pclar, Y_hat_pclar)
metrics_pclas <- calc_metrics(Y_true_pclar, Y_hat_pclas)

# Create comparison table
comparison_table <- data.frame(
  Model = c("Random Walk", "PCR", "DFM", "Ridge", "Lasso", "Elastic Net", "3PRF", "PCLAR", "PCLAS"),
  MSE = c(metrics_rw$MSE, metrics_pcr$MSE, metrics_dfm$MSE, metrics_ridge$MSE, 
          metrics_lasso$MSE, metrics_enet$MSE, metrics_3prf$MSE, 
          metrics_pclar$MSE, metrics_pclas$MSE),
  RMSE = c(metrics_rw$RMSE, metrics_pcr$RMSE, metrics_dfm$RMSE, metrics_ridge$RMSE,
           metrics_lasso$RMSE, metrics_enet$RMSE, metrics_3prf$RMSE,
           metrics_pclar$RMSE, metrics_pclas$RMSE),
  MAE = c(metrics_rw$MAE, metrics_pcr$MAE, metrics_dfm$MAE, metrics_ridge$MAE,
          metrics_lasso$MAE, metrics_enet$MAE, metrics_3prf$MAE,
          metrics_pclar$MAE, metrics_pclas$MAE),
  R2 = c(metrics_rw$R2, metrics_pcr$R2, metrics_dfm$R2, metrics_ridge$R2,
         metrics_lasso$R2, metrics_enet$R2, metrics_3prf$R2,
         metrics_pclar$R2, metrics_pclas$R2)
)

# Add relative RMSE (compared to Random Walk)
comparison_table$Relative_RMSE <- comparison_table$RMSE / comparison_table$RMSE[1]

# Sort by RMSE
comparison_table <- comparison_table %>% arrange(RMSE)

# Model comparison
print(kable(comparison_table, digits = 4, 
            caption = "Out-of-Sample Forecasting Performance (Rolling Window = 100)"))


#VISUALIZATION - INDIVIDUAL PLOTS FOR EACH MODEL
# Prepare data frames for each model
month_main <- month[(roll_window + 1):length(Y)]
month_3prf <- month_valid[(roll_window + 1):length(Y_valid)]
month_pclar <- month_valid_lag1[(roll_window + 1):length(Y_valid_lag1)]

#Plot 1: Random Walk 
df_rw <- data.frame(date = month_main, Actual = Y_true_main, Predicted = Y_hat_rw)
p_rw <- ggplot(df_rw, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("Random Walk - RMSE: %.4f", metrics_rw$RMSE),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())

#Plot 2: PCR 
df_pcr <- data.frame(date = month_main, Actual = Y_true_main, Predicted = Y_hat_pcr)
p_pcr <- ggplot(df_pcr, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("PCR - RMSE: %.4f", metrics_pcr$RMSE),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())

#Plot 3: DFM
df_dfm <- data.frame(date = month_main, Actual = Y_true_main, Predicted = Y_hat_dfm)
p_dfm <- ggplot(df_dfm, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("DFM - RMSE: %.4f", metrics_dfm$RMSE),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())

#Plot 4: Ridge 
df_ridge <- data.frame(date = month_main, Actual = Y_true_main, Predicted = Y_hat_ridge)
p_ridge <- ggplot(df_ridge, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("Ridge - RMSE: %.4f", metrics_ridge$RMSE),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())


#Plot 5: Lasso 
df_lasso <- data.frame(date = month_main, Actual = Y_true_main, Predicted = Y_hat_lasso)
p_lasso <- ggplot(df_lasso, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("Lasso - RMSE: %.4f | Avg Non-Zero: %d", 
                       metrics_lasso$RMSE, round(mean(non_zero_lasso), 0)),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())


#Plot 6: Elastic Net
df_enet <- data.frame(date = month_main, Actual = Y_true_main, Predicted = Y_hat_enet)
p_enet <- ggplot(df_enet, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("Elastic Net - RMSE: %.4f | Avg Non-Zero: %d", 
                       metrics_enet$RMSE, round(mean(non_zero_enet), 0)),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())

#Plot 7: 3PRF
df_3prf <- data.frame(date = month_3prf, Actual = Y_true_3prf, Predicted = Y_hat_3prf)
p_3prf <- ggplot(df_3prf, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("3PRF - RMSE: %.4f", metrics_3prf$RMSE),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())

#Plot 8: PCLAR 
df_pclar <- data.frame(date = month_pclar, Actual = Y_true_pclar, Predicted = Y_hat_pclar)
p_pclar <- ggplot(df_pclar, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("PCLAR - RMSE: %.4f", metrics_pclar$RMSE),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())

#Plot 9: PCLAS 
df_pclas <- data.frame(date = month_pclar, Actual = Y_true_pclar, Predicted = Y_hat_pclas)
p_pclas <- ggplot(df_pclas, aes(x = date)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = sprintf("PCLAS - RMSE: %.4f", metrics_pclas$RMSE),
       y = "ΔWTI (USD per barrel)", x = "Date") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())

# Print all plots
print(p_rw)
print(p_pcr)
print(p_dfm)
print(p_ridge)
print(p_lasso)
print(p_enet)
print(p_3prf)
print(p_pclar)
print(p_pclas)


# SAVE OUTPUTS - PLOTS AND COMPARISON TABLE

# Save Comparison Table
write.csv(comparison_table, 
          file = file.path(output_dir, "comparison_table.csv"), 
          row.names = FALSE)

# Save as formatted text table
sink(file.path(output_dir, "comparison_table.txt"))
print(kable(comparison_table, digits = 4, 
            caption = "Out-of-Sample Forecasting Performance (Rolling Window = 100)"))
sink()

# Save Descriptive Table 
sink(file.path(output_dir, "statistics_table.txt"))
print(kable(summary_Y, digits = 4, 
            caption = "Descriptive Statistics for WTI Oil Price (Y)"))
sink()

# Save Individual Plots 
# Set common dimensions for all plots
plot_width <- 10
plot_height <- 6
plot_dpi <- 300

# Save each plot
ggsave(filename = file.path(output_dir, "plot_00_time_series.png"), 
       plot = p_Y, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_01_random_walk.png"), 
       plot = p_rw, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_02_pcr.png"), 
       plot = p_pcr, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_03_dfm.png"), 
       plot = p_dfm, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_04_ridge.png"), 
       plot = p_ridge, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_05_lasso.png"), 
       plot = p_lasso, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_06_elastic_net.png"), 
       plot = p_enet, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_07_3prf.png"), 
       plot = p_3prf, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_08_pclar.png"), 
       plot = p_pclar, width = plot_width, height = plot_height, dpi = plot_dpi)

ggsave(filename = file.path(output_dir, "plot_09_pclas.png"), 
       plot = p_pclas, width = plot_width, height = plot_height, dpi = plot_dpi)


