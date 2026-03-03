# =============================================================================
# Data Preparation: Stationarity Testing and Differencing
# Input:  data/final_modeling_dataset.csv
# Output: data/stationary_data.csv
#
# Authors: Yuxin Zheng, Gulnara Sadykova, Yipeng Yan
# Course:  Machine Learning for Economists, University of Bologna, 2025
# =============================================================================
rm(list=ls())
cat('\014')

#Load libraries 
library(dplyr)
library(readr)
library(Matrix)
library(tseries)

#Configuration
input  <- "data/final_modeling_dataset.csv"
output <- "data/stationary_data.csv"
df_raw <- read.csv(input)
str(df_raw)
current_class <- class(df_raw)
cat("Current data structure class:", current_class, "\n")
# df_raw$X_FRED_2...85 <- log(df_raw$X_FRED_2...85)
colnames(df_raw)
df <- df_raw %>% 
  select(-c("X_WEO_Total.Primary.Energy.Consumption", "X_WEO_Total.Renewable.Energy.Consumption", 
            "X_WEO_Nuclear.Electric.Power.Production","X_WEO_Total.Renewable.Energy.Production",   
            "X_WEO_Total.Primary.Energy.Production", "X_WEO_Primary.Energy.Imports",                
            "X_WEO_Primary.Energy.Exports", "X_WEO_Total.Fossil.Fuels.Consumption",            
            "X_WEO_Total.Fossil.Fuels.Production", "X_WEO_Nuclear.Electric.Power.Consumption"))
str(df$X_WEO_Primary.Energy.Net.Imports)
############Check whether columns are numeric/int
non_numeric_cols <- names(df)[!sapply(df, is.numeric) & !sapply(df, is.integer)]
if (length(non_numeric_cols) > 0) {
  df_numeric <- df[, !(names(df) %in% non_numeric_cols)]
  cat("Removed the following non-numeric columns:", paste(non_numeric_cols, collapse = ", "), "\n")
  cat("New dimensions:", nrow(df_numeric), "x", ncol(df_numeric), "\n")
} else {
  df_numeric <- df
  cat("All columns are already numeric (int or num).\n")
}
############Augmented Dickey-Fuller (ADF) test, to check for stationarity
#If p-value < 0.05, the series is stationary
adf_results_level <- lapply(df_numeric, function(x) {
  test <- try(adf.test(x), silent = TRUE)
  if (inherits(test, "try-error")) NA else test$p.value
})
adf_results_level_df <- data.frame(
  variable = names(df_numeric),
  p_value = unlist(adf_results_level)
)
non_stationary_vars <- adf_results_level_df$variable[adf_results_level_df$p_value >= 0.05]
cat("Number of non-stationary (I(1)) variables:", length(non_stationary_vars), "\n")
original <- subset(adf_results_level_df, p_value >= 0.05)
#first difference
df_diff_1 <- df_numeric
for (var in non_stationary_vars) {
  x <- df_numeric[[var]]
  df_diff_1[[var]] <- c(NA, diff(x))
}
df_diff_1 <- df_diff_1[-1, ]
adf_results_diff_1 <- lapply(non_stationary_vars, function(var) {
  x <- df_diff_1[[var]]
  test <- try(adf.test(x), silent = TRUE)
  if (inherits(test, "try-error")) NA else test$p.value
})
adf_results_diff_1 <- data.frame(
  variable = non_stationary_vars,
  p_value = unlist(adf_results_diff_1)
)
still_non_stationary <- adf_results_diff_1$variable[adf_results_diff_1$p_value >= 0.05]
cat("Number of variables still non-stationary after 1st diff:", length(still_non_stationary), "\n")
first <- subset(adf_results_diff_1, p_value >= 0.05)
#second difference
df_diff_2 <- df_diff_1
for (var in still_non_stationary) {
  x <- df_diff_1[[var]]
  df_diff_2[[var]] <- c(NA, diff(x))
}
df_diff_2 <- df_diff_2[-1, ]  
adf_results_diff_2 <- lapply(still_non_stationary, function(var) {
  x <- df_diff_2[[var]]
  test <- try(adf.test(x), silent = TRUE)
  if (inherits(test, "try-error")) NA else test$p.value
})
adf_results_diff_df_2 <- data.frame(
  variable = still_non_stationary,
  p_value = unlist(adf_results_diff_2)
)
still_non_stationary_final <- adf_results_diff_df_2$variable[adf_results_diff_df_2$p_value >= 0.05]
cat("Number of variables still non-stationary after 2nd diff:", length(still_non_stationary_final), "\n")

df_diff_1$date <- tail(df$date, nrow(df_diff_1))
df_diff_2$date <- tail(df$date, nrow(df_diff_2))
df_diff_1 <- df_diff_1 %>% relocate(date, .before = 1)
df_diff_2 <- df_diff_2 %>% relocate(date, .before = 1)

write_csv(df_diff_2, output)
############Matrix
non_numeric_cols <- names(df_diff_2)[!sapply(df_diff_2, is.numeric) & !sapply(df_diff_2, is.integer)]
if (length(non_numeric_cols) > 0) {
  df_numeric_1 <- df_diff_2[, !(names(df_diff_2) %in% non_numeric_cols)]
  cat("Removed the following non-numeric columns:", paste(non_numeric_cols, collapse = ", "), "\n")
  cat("New dimensions:", nrow(df_numeric_1), "x", ncol(df_numeric_1), "\n")
} else {
  df_numeric_1 <- df_diff_2
  cat("All columns are already numeric (int or num).\n")
}
current_class <- class(df_numeric_1)
cat("Current data structure class:", current_class, "\n")
my_data_matrix <- as.matrix(df_numeric_1)
cat("Data converted to numeric matrix 'my_data_matrix'.\n")
############Check for NAs
if (any(is.na(my_data_matrix))) {
  num_rows_before <- nrow(my_data_matrix)
  rows_with_na <- sum(rowSums(is.na(my_data_matrix)) > 0)
  cat("\n WARNING: The matrix contains missing values (NA).\n")
  cat(rows_with_na, "out of", num_rows_before, "rows contain at least one NA.\n")
  cat("Proceeding by removing these rows (na.omit).\n")
  my_data_matrix_cleaned <- na.omit(my_data_matrix)
  num_rows_after <- nrow(my_data_matrix_cleaned)
  cat("Removed", num_rows_before - num_rows_after, "rows. New observation count:", num_rows_after, "\n")
} else {
  my_data_matrix_cleaned <- my_data_matrix
  cat("No missing values detected. Proceeding with full dataset.\n")
}
# Calculate rank (R) using a numerically stable method
matrix_rank <- rankMatrix(my_data_matrix_cleaned, tol = .Machine$double.eps)[1]
num_variables <- ncol(my_data_matrix_cleaned)
cat("\n--- Multicollinearity Assessment ---\n")
cat("Total number of variables (p):", num_variables, "\n")
cat("Rank of the data matrix (R):", matrix_rank, "\n")

# Interpretation: R < p indicates perfect multicollinearity
if (matrix_rank < num_variables) {
  cat("\nASSESSMENT RESULT: Perfect Multicollinearity IS present. \n")
  cat("Interpretation: Rank (", matrix_rank, ") < Variables (", num_variables, "). One or more variables are perfectly collinear.\n", sep="")
} else {
  cat("\nASSESSMENT RESULT: Perfect Multicollinearity IS NOT present. \n")
  cat("Interpretation: Rank (", matrix_rank, ") = Variables (", num_variables, "). Variables are linearly independent.\n", sep="")
}

#Define a suitable tolerance for economic data
# 1e-10 is a good starting point to capture near-perfect dependencies.
TOLERANCE <- 1e-10 

#Perform QR Decomposition
qr_result <- qr(my_data_matrix_cleaned, tol = TOLERANCE)

#Identify the Rank and the Indices of Redundant Variables
observed_rank <- qr_result$rank
num_vars <- ncol(my_data_matrix_cleaned)

# Variables retained (Indices of the independent variables in the original matrix):
independent_indices <- qr_result$pivot[1:observed_rank]
independent_variables <- colnames(my_data_matrix_cleaned)[independent_indices]

# Variables to drop (Indices of the dependent/redundant variables):
redundant_indices <- qr_result$pivot[(observed_rank + 1):num_vars]
redundant_variables <- colnames(my_data_matrix_cleaned)[redundant_indices] # CORRECTED ASSIGNMENT

#Report the Findings
cat("\n--- Redundant Variable Identification (Tolerance =", TOLERANCE, ") ---\n")
cat("Total Variables (p):", num_vars, "\n")
cat("Observed Rank (R):", observed_rank, "\n")
cat("Number of Redundant Variables to Drop:", num_vars - observed_rank, "\n\n")

cat("Variables RETAINED (Linearly Independent):\n")
cat(paste(independent_variables, collapse = ", "), "\n\n")

cat("Variables to DROP (Linearly Dependent/Redundant):\n")
if (length(redundant_variables) > 0) {
  cat(paste(redundant_variables, collapse = ", "), "\n")
} else {
  cat("None (The rank equals the number of variables at this tolerance level).\n")
}
