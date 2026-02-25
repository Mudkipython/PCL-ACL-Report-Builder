######################################################
# MGSC 661: PART 2 - AUTOMATED PDF REPORT GENERATOR
# Focus: Pure Macro Factors (No GILS)
# Output: A comprehensive PDF with 2 pages per bank
# Feature: Added Model Scorecard (R2, MAPE, RMSE, MAE)
######################################################

# -------- 1. Setup --------
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(readxl)) install.packages("readxl")
if(!require(fpp2)) install.packages("fpp2")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(reshape2)) install.packages("reshape2")
if(!require(grid)) install.packages("grid") 

library(tidyverse); library(readxl); library(fpp2); library(gridExtra); library(reshape2); library(grid)

# -------- 2. Data Loading --------
# Read Data
raw_df <- read_excel("Forecast Data.xlsx")

colnames(raw_df) <- c("Bank", "Year", "Quarter", "Macro_Date", "Release_Date", 
                      "PCL", "PCL_next", "GILS", "ACL", "Net_WriteOffs", 
                      "Total_Loans", "Unemp_Change_3m", "GDP_Q_Change", 
                      "GDP_Y_Change", "Unemp_Rate_Q_Forecast", "Unemp_Rate_Annual_Forecast", "Central_Bank_Rate_Forecast")

df_clean <- raw_df %>%
  mutate(Q_num = as.numeric(sub("Q", "", Quarter))) %>%
  # Clean numeric columns
  mutate(across(c(ACL, Total_Loans, GILS), ~as.numeric(gsub(",", "", as.character(.))))) %>%
  mutate(
    PCL_Rate  = (PCL / Total_Loans) * 100
  ) %>%
  arrange(Bank, Year, Q_num)

bank_list <- unique(df_clean$Bank)
leaderboard <- data.frame() 
all_plots_list <- list() # Storage for PDF pages

# ====================================================
# MAIN ANALYSIS LOOP
# ====================================================

cat("\n>>> STARTING ANALYSIS FOR PDF GENERATION...\n")

for(b in bank_list) {
  
  cat(paste("Processing Bank:", b, "...\n"))
  
  # 1. Data Prep
  b_data <- df_clean %>% filter(Bank == b)
  start_yr <- min(b_data$Year); start_q <- min(b_data$Q_num[b_data$Year == start_yr])
  y_ts <- ts(b_data$PCL_Rate, start=c(start_yr, start_q), frequency=4)
  
  # Regressors (Macro Only: GDP, Unemp, Rate)
  x_mat <- b_data %>% 
    select(GDP_Y_Change, Unemp_Rate_Annual_Forecast, Central_Bank_Rate_Forecast) %>% 
    as.matrix()
  colnames(x_mat) <- c("GDP", "Unemp", "Rate")
  
  # ------------------------------------------------
  # PAGE 1: PROCESS & DRIVERS (EDA)
  # ------------------------------------------------
  
  # [Chart A] Decomposition
  decomp <- decompose(y_ts, type="additive")
  p_decomp <- autoplot(decomp) + 
    labs(title=paste("1. Decomposition:", b), y="") + 
    theme_bw() + theme(plot.title = element_text(size=10, face="bold"))
  
  # [Chart B] Variable Trends (Z-Score) - Macro Only
  scaled_data <- scale(cbind(PCL=y_ts, x_mat))
  df_scaled <- as.data.frame(scaled_data); df_scaled$Time <- as.numeric(time(y_ts))
  melted_data <- melt(df_scaled, id.vars="Time") 
  
  p_vars <- ggplot(melted_data, aes(x=Time, y=value, color=variable)) +
    geom_line(size=0.8) +
    facet_wrap(~variable, scales="free_y", ncol=2) +
    labs(title=paste("2. Macro Trends (Z-Score):", b), y="Std Dev") +
    theme_minimal() + theme(legend.position="none", plot.title = element_text(size=10, face="bold"))
  
  # [Chart C] Correlation Heatmap - Macro Only
  cor_dat <- cor(cbind(PCL=y_ts, x_mat), use="complete.obs")
  melted_cor <- melt(cor_dat)
  p_corr <- ggplot(melted_cor, aes(x=Var1, y=Var2, fill=value)) + 
    geom_tile(color="white") +
    geom_text(aes(label=round(value, 2)), size=3) +
    scale_fill_gradient2(low="blue", high="red", mid="white", limit=c(-1,1)) +
    labs(title=paste("3. Macro Correlation:", b), x="", y="") +
    theme_minimal() + theme(axis.text.x=element_text(angle=45, hjust=1), legend.position="none", plot.title = element_text(size=10, face="bold"))
  
  # Store Page 1
  all_plots_list[[paste0(b, "_P1")]] <- arrangeGrob(p_decomp, p_vars, p_corr, layout_matrix = rbind(c(1,2), c(3,2)), top=paste("PART A: PROCESS ANALYSIS -", b))
  
  
  # ------------------------------------------------
  # PAGE 2: FORECAST & IMPACT
  # ------------------------------------------------
  
  # Modeling
  h <- 4
  len <- length(y_ts) - h
  y_tr <- head(y_ts, len); y_te <- tail(y_ts, h)
  x_tr <- head(x_mat, len); x_te <- tail(x_mat, h)
  
  fit <- auto.arima(y_tr, xreg=x_tr, seasonal=TRUE)
  fc  <- forecast(fit, xreg=x_te, h=h)
  
  # Metrics Calculation
  coefs <- coef(fit)
  c_gdp  <- round(if("GDP" %in% names(coefs)) coefs["GDP"] else 0, 4)
  c_unemp<- round(if("Unemp" %in% names(coefs)) coefs["Unemp"] else 0, 4)
  c_rate <- round(if("Rate" %in% names(coefs)) coefs["Rate"] else 0, 4)
  
  r2 <- 1 - (sum(residuals(fit)^2) / sum((y_tr - mean(y_tr))^2))
  mape <- accuracy(fc, y_te)[2, "MAPE"]
  rmse <- accuracy(fc, y_te)[2, "RMSE"]
  mae  <- accuracy(fc, y_te)[2, "MAE"]
  
  leaderboard <- rbind(leaderboard, data.frame(Bank=b, R2=round(r2,3), MAPE=round(mape,2), RMSE=round(rmse,4), MAE=round(mae,4), GDP=c_gdp, Unemp=c_unemp, Rate=c_rate))
  
  # Scenarios (Macro Driven)
  fc_hor <- 8; vars <- colnames(x_mat)
  f_base <- matrix(NA, fc_hor, 3); colnames(f_base)<-vars
  f_bear <- matrix(NA, fc_hor, 3); colnames(f_bear)<-vars
  f_bull <- matrix(NA, fc_hor, 3); colnames(f_bull)<-vars
  
  for(v in vars) {
    f_m <- forecast(auto.arima(x_mat[,v]), h=fc_hor, level=95)
    f_base[,v] <- f_m$mean
    if(v=="GDP") {f_bear[,v]<-f_m$lower; f_bull[,v]<-f_m$upper} 
    else {f_bear[,v]<-f_m$upper; f_bull[,v]<-f_m$lower}
  }
  
  full_fit <- auto.arima(y_ts, xreg=x_mat, seasonal=TRUE)
  fc_base <- forecast(full_fit, xreg=f_base, h=fc_hor)
  fc_bear <- forecast(full_fit, xreg=f_bear, h=fc_hor)
  fc_bull <- forecast(full_fit, xreg=f_bull, h=fc_hor)
  
  # Plot Data
  d_h <- time(y_ts); d_f <- seq(max(d_h)+0.25, by=0.25, length.out=fc_hor)
  p_df <- data.frame(Date=c(as.numeric(d_h), rep(as.numeric(d_f),3)), 
                     Rate=c(as.numeric(y_ts), as.numeric(fc_base$mean), as.numeric(fc_bear$mean), as.numeric(fc_bull$mean)),
                     Type=c(rep("Hist",length(y_ts)), rep("Base",fc_hor), rep("Bear",fc_hor), rep("Bull",fc_hor)))
  r_df <- data.frame(Date=as.numeric(d_f), Ymin=as.numeric(fc_bull$mean), Ymax=as.numeric(fc_bear$mean))
  
  # Fan Chart Plot
  p_fan <- ggplot() +
    geom_ribbon(data=r_df, aes(x=Date, ymin=Ymin, ymax=Ymax, fill="Range"), alpha=0.15) +
    geom_line(data=p_df, aes(x=Date, y=Rate, color=Type, linetype=Type), size=1) +
    scale_fill_manual(values="red") + scale_color_manual(values=c("black","blue","red","green")) + 
    scale_linetype_manual(values=c("solid","solid","dashed","dashed")) +
    labs(title=paste("4. Scenario Forecast:", b), subtitle="Risk Range (95% CI)", y="PCL Rate (%)") + theme_minimal() + theme(legend.position="bottom")
  
  # Coefficients Bar Chart Plot
  coef_df <- data.frame(Variable = c("GDP", "Unemp", "Rate"), Impact = c(c_gdp, c_unemp, c_rate))
  p_coef <- ggplot(coef_df, aes(x=reorder(Variable, abs(Impact)), y=Impact, fill=Impact>0)) +
    geom_bar(stat="identity") + coord_flip() + geom_text(aes(label=Impact), hjust=-0.2) +
    scale_fill_manual(values=c("TRUE"="#E46726", "FALSE"="#6D9EC1")) +
    labs(title="5. Macro Sensitivity (Coef)", subtitle="Pos = Increases Risk", x="", y="Impact") + theme_minimal() + theme(legend.position="none")
  
  # CREATE MODEL SCORECARD TABLE FOR PDF
  metric_df <- data.frame(
    Metric = c("R-Squared", "MAPE (%)", "RMSE", "MAE"),
    Value = c(round(r2, 3), round(mape, 2), round(rmse, 4), round(mae, 4))
  )
  # Format table to look nice
  p_table <- tableGrob(metric_df, rows=NULL, theme=ttheme_minimal(base_size=10, padding=unit(c(4, 4), "mm")))
  title_table <- textGrob("6. Model Scorecard", gp=gpar(fontsize=12, fontface="bold", col="#333333"))
  p_scorecard <- arrangeGrob(title_table, p_table, heights=c(0.15, 0.85))
  
  # Combine Coefficients Chart and Scorecard Table side-by-side
  bottom_row <- arrangeGrob(p_coef, p_scorecard, ncol=2, widths=c(2, 1))
  
  # Store Page 2
  all_plots_list[[paste0(b, "_P2")]] <- arrangeGrob(p_fan, bottom_row, heights=c(2, 1), top=paste("PART B: FORECAST & IMPACT -", b))
}

# ====================================================
# FINAL PDF GENERATION (NO TITLE PAGE)
# ====================================================

cat("\n>>> GENERATING PDF REPORT...\n")

pdf("Bank_Analysis_Report.pdf", width=11, height=8.5)

# 1. Loop through saved plots and print to PDF
if(length(all_plots_list) > 0) {
  for(p_name in names(all_plots_list)) {
    grid.arrange(all_plots_list[[p_name]])
  }
} else {
  cat("WARNING: No plots found. Please check if the loop ran correctly.\n")
}

dev.off()

cat(">>> SUCCESS! PDF Saved as 'Bank_Analysis_Report.pdf'.\n")
cat(">>> Check your working directory folder.\n")

# Print Leaderboard to Console for quick check
cat("\n--- PERFORMANCE LEADERBOARD (SORTED BY R-SQUARED) ---\n")
if(nrow(leaderboard) > 0) {
  print(leaderboard %>% arrange(desc(R2)))
}