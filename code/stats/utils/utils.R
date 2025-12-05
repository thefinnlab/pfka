# utils.R

###################################
####### Define our paths ##########
###################################

# Set paths -- Use file.path() to join directories
BASE_DIR <- '/Volumes/FinnLab/tommy/isc_asynchrony_behavior'

# Function to set paths globally
set_paths <- function(analysis_name, plots_name) {
  # Set global variables using the <<- operator
  results_dir <<- file.path(BASE_DIR, "derivatives", "results", analysis_name)
  plots_dir <<- file.path(BASE_DIR, "derivatives", "plots", "final", plots_name)
  stim_dir <<- file.path(BASE_DIR, "stimuli", "preprocessed")
}

###################################
######### Load packages ###########
###################################

# Define the default required packages
required_packages <- c("lmerTest", "emmeans", "ggplot2", "tidyverse", "sjPlot", 
                       "rcompanion", "DHARMa", "glmmTMB", "gt", "gamm4")

# Function to load necessary packages
install_and_load <- function(additional_packages = NULL) {
  # Combine the default required packages with any additional packages specified by the user
  all_packages <- c(required_packages, additional_packages)
  
  # Loop through the list of packages and install/load them
  for (pkg in all_packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    }
  }
}

####################################################
####### Data handling functions packages ###########
####################################################

# Function to load data and set factor columns
load_and_prepare_data <- function(file_path, factor_columns) {
  df <- read.csv(file_path)
  df <- convert_columns_to_factors(df, factor_columns)
  
  if ("filter_for_leakage" %in% names(df)){
    df$filter_for_leakage <- as.logical(df$filter_for_leakage)
  }
  
  # Need to adjust lower bound of predictability
  if ("predictability" %in% names(df)){
    df[df$predictability == 0,]$predictability <- 0.01
  }
  
  if ("human_predictability" %in% names(df)){
    df[df$human_predictability == 0,]$human_predictability <- 0.01
    df[df$human_predictability == 1,]$human_predictability <- 0.99
    
    df$log_model_predictability <- log(df$model_predictability)
    df$log_human_predictability <- log(df$human_predictability)
    
    df <- remove_outliers(df, 'log_model_predictability', multiplier=1.25)
  }
  
  return(df)
}

# Function to set factor levels for a given column
set_factor_levels <- function(df, column_name, levels) {
  if (column_name %in% names(df)) {
    df[[column_name]] <- factor(df[[column_name]], levels = levels)
  }
  return(df)
}

# Function to remove outliers based on IQR of a specific column and print the number of removed values
remove_outliers <- function(df, column, multiplier = 1.5) {
  if (!column %in% names(df)) {
    stop("Column not found in dataframe.")
  }
  if (!is.numeric(df[[column]])) {
    stop("Selected column is not numeric.")
  }
  
  initial_rows <- nrow(df)  # Store the initial number of rows
  
  # Calculate IQR bounds for the specified column
  Q1 <- quantile(df[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[column]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - multiplier * IQR
  upper_bound <- Q3 + multiplier * IQR
  
  # Filter rows based on the specified column's bounds
  df_clean <- df[df[[column]] >= lower_bound & df[[column]] <= upper_bound, ]
  
  # Calculate and print the number of rows removed
  removed_rows <- initial_rows - nrow(df_clean)
  cat("Number of rows removed due to outliers in column", column, ":", removed_rows, "\n")
  
  return(df_clean)
}

# Function to convert specified columns to factors
convert_columns_to_factors <- function(df, columns_to_convert) {
  # Check if all specified columns exist in the dataframe
  if (!all(columns_to_convert %in% names(df))) {
    missing_cols <- columns_to_convert[!columns_to_convert %in% names(df)]
    stop(paste("The following columns do not exist in the dataframe:", 
               paste(missing_cols, collapse = ", ")))
  }
  
  # Convert each specified column to factor
  for (col in columns_to_convert) {
    df[[col]] <- factor(df[[col]])
  }
  
  return(df)
}

####################################################
############## LMER setup functions ################
####################################################

# Function to create contrasts for modality
create_column_contrast <- function(df, column_name, n_poly = NULL) {
  # Get the unique values in the specified column
  unique_values <- length(unique(df[[column_name]]))
  
  # If n_poly is NULL, set it to the maximum number of contrasts (unique values - 1)
  if (is.null(n_poly)) {
    n_poly <- unique_values - 1
  }
  
  # Create the polynomial contrasts for the specified column
  column_contrast <- contr.poly(unique_values)
  
  # Apply the contrasts to the specified column in the dataframe
  contrasts(df[[column_name]], how.many = n_poly) <- cbind(column_contrast)
  
  return(df)
}

####################################################
########## Task-based LMER helper function #########
####################################################

# Define a function to fit the model and output results
lmer_task <- function(task_name, df, formula, family=NULL) {
  # Subset the data for the specific task
  df_task <- df[df$task == task_name, ]
  
  # If we've specified a given model family, use glmer
  if (!is.null(family)){
    model <- glmer(formula, data=df_task, family=family)
  }
  # Fit the mixed-effects model
  else{
    model <- lmer(formula, data = df_task)
  }
  
  return (model)
}

####################################################
###### Pretty printing of models/comparisons #######
####################################################
# Helper function to determine the column names and labels based on model type
get_column_labels <- function(is_logistic, is_pairwise = FALSE) {
  if (is_pairwise) {
    if (is_logistic) {
      cols <- c("contrast", "estimate", "SE", "df", "z.ratio", "p.value")
      labels <- c("Comparison", "Difference in Mean", "Standard Error", "Degrees of Freedom", "Z-Statistic", "P-Value")
    } else {
      cols <- c("contrast", "estimate", "SE", "df", "t.ratio", "p.value")
      labels <- c("Comparison", "Difference in Mean", "Standard Error", "Degrees of Freedom", "T-Statistic", "P-Value")
    }
  } else {
    if (is_logistic) {
      cols <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
      labels <- c("Coefficient Estimate", "Standard Error", "Z-Statistic", "P-Value")
    } else {
      cols <- c("Estimate", "Std. Error", "t value", "Pr(>|t|)")
      labels <- c("Coefficient Estimate", "Standard Error", "T-Statistic", "P-Value")
    }
  }
  setNames(labels, cols)
}

# Generalized function to display model summaries or pairwise comparisons
display_summary_table <- function(data, is_pairwise = FALSE) {
  # Determine if the model is logistic by checking for 'z value' or 'z.ratio'
  is_logistic <- any(c("z value", "z.ratio") %in% colnames(data))
  
  # Add the predictor names (row names) to the data for model summaries
  if (!is_pairwise) {
    data$Predictor <- rownames(data)
    
    # Reorder the columns to place 'Predictor' first
    data <- data[, c("Predictor", setdiff(names(data), "Predictor"))]
  }
  
  # Get column labels using the helper function
  col_labels <- get_column_labels(is_logistic, is_pairwise)
  
  # Customize the table output using `gt`
  table_output <- data %>%
    gt() %>%
    tab_header(title = ifelse(is_pairwise, "Pairwise Comparisons", "Model Fixed Effects Summary")) %>%
    cols_label(!!!col_labels) %>%
    tab_spanner(label = ifelse(is_pairwise, "Comparison Results", "Fixed Effects Results"), columns = names(col_labels))
  
  # Apply left alignment for 'Predictor' column only if it's not pairwise
  if (!is_pairwise) {
    table_output <- table_output %>%
      cols_align(align = "left", columns = "Predictor") # Align 'Predictor' column to the left
  }
  
  table_output
}

# Function to display model summary in a nicely formatted table
display_model_summary <- function(model) {
  # Get the summary of the model and extract the coefficients table (fixed effects)
  model_summary <- summary(model)
  fixed_effects_df <- as.data.frame(model_summary$coefficients)
  
  # Call the generalized function to display the model summary
  display_summary_table(fixed_effects_df)
}

# Function to display pairwise comparisons in a nicely formatted table
display_pairwise_comparisons <- function(emmeans_results) {
  # Extract the pairwise contrasts (comparisons) from the emmeans results
  pairwise_comparisons <- emmeans_results$contrasts %>%
    as.data.frame()
  
  # Call the generalized function to display pairwise comparisons
  display_summary_table(pairwise_comparisons, is_pairwise = TRUE)
}