#######################################################
# MGSC 661: PART 1 - INDUSTRY AGGREGATE (PURE MACRO)
# Focus: External Variables Only (GDP, Unemp, Rates)
# Mode: Interactive (Press Enter to see next plot)
#######################################################

# -------- 1. Setup --------
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(readxl)) install.packages("readxl")
if(!require(fpp2)) install.packages("fpp2")
if(!require(corrplot)) install.packages("corrplot")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(smooth)) install.packages("smooth")

library(tidyverse); library(readxl); library(fpp2); library(corrplot); library(gridExtra); library(smooth)

# -------- 2. Data Loading --------
raw_df <- read_excel("Forecast Data.xlsx")

colnames(raw_df) <- c("Bank", "Year", "Quarter", "Macro_Date", "Release_Date", 
                      "PCL", "PCL_next", "GILS", "ACL", "Net_WriteOffs", 
                      "Total_Loans", "Unemp_Change_3m", "GDP_Q_Change", 
                      "GDP_Y_Change", "Unemp_Rate_Q_Forecast", "Unemp_Rate_Annual_Forecast", "Central_Bank_Rate_Forecast")

# Data Cleaning & Aggregation
df_industry <- raw_df %>%
  mutate(Q_num = as.numeric(sub("Q", "", Quarter))) %>%
  mutate(across(c(ACL, Total_Loans, GILS), ~as.numeric(gsub(",", "", as.character(.))))) %>%
  group_by(Year, Q_num) %>%
  summarise(
    Total_PCL = sum(PCL, na.rm=T), 
    Total_Loans = sum(Total_Loans, na.rm=T), 
    # Macro Indicators (Mean)
    Avg_GDP_YY = mean(GDP_Y_Change, na.rm=T),
    Avg_Unemp_A = mean(Unemp_Rate_Annual_Forecast, na.rm=T),
    Avg_BankRate = mean(Central_Bank_Rate_Forecast, na.rm=T),
    .groups = "drop"
  ) %>%
  mutate(
    PCL_Rate = (Total_PCL / Total_Loans) * 100,  # Target
    Date_Dec = Year + (Q_num-1)/4
  ) %>% 
  arrange(Year, Q_num)

# Time Series Object
start_yr <- min(df_industry$Year); start_q <- min(df_industry$Q_num[df_industry$Year == start_yr])
y_rate <- ts(df_industry$PCL_Rate, start=c(start_yr, start_q), frequency=4)

cat("\n>>> PART 1: INTERACTIVE PRESENTATION MODE <<<\n")
cat(">>> Please look at the Plots pane and press [ENTER] in the Console to advance.\n")

# ====================================================
# STEP 1: MACRO-ONLY EXPLORATION
# ====================================================
cat("\n--- STEP 1: MACRO DRIVER EXPLORATION ---\n")

# Regressors (External Only)
x_reg_macro <- df_industry %>% 
  select(Avg_GDP_YY, Avg_Unemp_A, Avg_BankRate) %>% 
  as.matrix()
colnames(x_reg_macro) <- c("GDP_YY", "Unemp_Rate", "Bank_Rate")

# [Chart 1] Correlation: Macro vs PCL
cor_macro <- cor(cbind(PCL_Rate=df_industry$PCL_Rate, x_reg_macro), use="complete.obs")
corrplot(cor_macro, method="number", type="upper", tl.col="black", 
         title="1. Correlation: PCL vs External Variables", mar=c(0,0,2,0))

readline(prompt=">> [1/6] Correlation Matrix displayed. Press [Enter] for Trend Dashboard...")

# [Chart 2] Trend Dashboard
p1 <- autoplot(y_rate) + ggtitle("Target: PCL Rate") + theme_bw()
p2 <- autoplot(ts(df_industry$Avg_GDP_YY, start=start(y_rate), frequency=4)) + ggtitle("GDP Growth (Y/Y)") + theme_bw()
p3 <- autoplot(ts(df_industry$Avg_Unemp_A, start=start(y_rate), frequency=4)) + ggtitle("Unemployment Forecast") + theme_bw()
p4 <- autoplot(ts(df_industry$Avg_BankRate, start=start(y_rate), frequency=4)) + ggtitle("Central Bank Rate") + theme_bw()
grid.arrange(p1, p2, p3, p4, nrow=2, ncol=2)

readline(prompt=">> [2/6] Trend Dashboard displayed. Press [Enter] for Decomposition...")


# ====================================================
# STEP 3: TIME SERIES PROPERTIES & SMOOTHING
# ====================================================
cat("\n--- STEP 3: TIME SERIES PROPERTIES ---\n")

# [Chart 3] Decomposition
decomp <- decompose(y_rate, type="additive")
print(autoplot(decomp) + ggtitle("3. Decomposition of PCL Rate"))

readline(prompt=">> [3/6] Decomposition displayed. Press [Enter] for Smoothing Check...")

# [Chart 4] SMA Smoothing
# >> JUSTIFICATION FOR LOUIS <<
cat("\n>> REASONING: SMA(2) vs SMA(4) <<\n")
cat(" - SMA(2): Semi-annual smoothing (2 Qtrs). Smooths out immediate noise to capture short-term credit momentum.\n")
cat(" - SMA(4): Annual smoothing (4 Qtrs). Removes seasonality entirely to reveal the underlying structural credit cycle.\n")

sma2 = sma(y_rate, order=2)
sma4 = sma(y_rate, order=4)
p_raw = autoplot(y_rate) + ggtitle("Raw Data") + theme_bw()
p_sma2 = autoplot(fitted(sma2)) + ggtitle("SMA(2) - Semi-Annual Trend") + theme_bw()
p_sma4 = autoplot(fitted(sma4)) + ggtitle("SMA(4) - Annual Structural Trend") + theme_bw()
grid.arrange(p_raw, p_sma2, p_sma4, nrow=3, top="4. Smoothing Analysis (Noise Reduction)")

readline(prompt=">> [4/6] Smoothing displayed. Press [Enter] for Validation Split...")


# ====================================================
# STEP 4: VALIDATION
# ====================================================
cat("\n--- STEP 4: VALIDATION DESIGN ---\n")

h_test <- 4
len <- length(y_rate) - h_test
y_train <- head(y_rate, len); y_test <- tail(y_rate, h_test)
x_train <- head(x_reg_macro, len); x_test <- tail(x_reg_macro, h_test)

# [Chart 5] Validation Split
df_industry$Split <- c(rep("Training", len), rep("Test", h_test))
p_split <- ggplot(df_industry, aes(x=Date_Dec, y=PCL_Rate, color=Split)) +
  geom_line(size=1) + geom_point() +
  geom_vline(xintercept = df_industry$Date_Dec[len], linetype="dashed") +
  scale_color_manual(values=c("Test"="red", "Training"="blue")) +
  labs(title="5. Validation Strategy", subtitle="Train (Blue) vs Test (Red)", x="Year", y="PCL Rate") +
  theme_minimal() + theme(legend.position="bottom")
print(p_split)

readline(prompt=">> [5/6] Validation Split displayed. Press [Enter] for Model Results...")


# ====================================================
# STEP 5: MODELING, METRICS & P-VALUES
# ====================================================
cat("\n--- STEP 5: MACRO-MODELING & METRICS ---\n")

# Baseline
m_naive <- naive(y_train, h=h_test)

# ARIMAX (Macro Only)
fit <- auto.arima(y_train, xreg=x_train, seasonal=TRUE)
fc  <- forecast(fit, xreg=x_test, h=h_test)

# [Chart 6] Forecast Validation
print(autoplot(window(y_rate, start=c(2019,1))) +
        autolayer(m_naive, series="Naive", PI=F) +
        autolayer(fc, series="Macro-Model", PI=F, size=1.2) +
        autolayer(y_test, series="Actual", color="black", size=1) +
        labs(title="6. Forecast Validation: Pure Macro Model vs Naive") + theme_minimal())

# Calculate Metrics
ss_res <- sum(residuals(fit)^2)
ss_tot <- sum((y_train - mean(y_train))^2)
r_squared  <- 1 - (ss_res / ss_tot)

rmse_macro <- accuracy(fc, y_test)[2, "RMSE"]
mae_macro  <- accuracy(fc, y_test)[2, "MAE"]
mape_macro <- accuracy(fc, y_test)[2, "MAPE"]

rmse_naive <- accuracy(m_naive, y_test)[2, "RMSE"]
mae_naive  <- accuracy(m_naive, y_test)[2, "MAE"]
mape_naive <- accuracy(m_naive, y_test)[2, "MAPE"]

# Output Justifications & Scorecard
cat("\n>>> METRICS JUSTIFICATION <<<\n")
cat("1. MAPE: Highly interpretable relative percentage error. Excellent for executive communication.\n")
cat("2. RMSE: Penalizes large forecasting errors heavily. Crucial for banking risk management to avoid massive capital shortfalls.\n")
cat("3. MAE: Shows average absolute deviation in target units (PCL Rate %), giving a baseline expected variance.\n")

cat("\n>>> MODEL SCORECARD <<<\n")
print(data.frame(
  Metric = c("R-Squared (Fit)", "MAPE (%)", "RMSE", "MAE"),
  Macro_Model = c(round(r_squared, 3), round(mape_macro, 2), round(rmse_macro, 4), round(mae_macro, 4)),
  Naive_Model = c(NA, round(mape_naive, 2), round(rmse_naive, 4), round(mae_naive, 4))
))

# Output P-Values
cat("\n>>> MACRO COEFFICIENTS & P-VALUES <<<\n")
# Extracting coefficients and calculating p-values manually since auto.arima doesn't print them by default
coefs  <- fit$coef
se     <- sqrt(diag(fit$var.coef))
t_stat <- coefs / se
p_val  <- 2 * (1 - pnorm(abs(t_stat)))

# Add Significance Stars
sig_stars <- ifelse(p_val < 0.001, "***",
                    ifelse(p_val < 0.01, "**",
                           ifelse(p_val < 0.05, "*",
                                  ifelse(p_val < 0.1, ".", " "))))

coef_df <- data.frame(
  Estimate  = round(coefs, 4),
  Std_Error = round(se, 4),
  z_value   = round(t_stat, 2),
  Pr_z      = format.pval(p_val, digits = 4),
  Signif    = sig_stars
)

print(coef_df)
cat("--- Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ---\n")

readline(prompt=">> [6/6] Model Results displayed. Press [Enter] for Stress Test...")


# ====================================================
# STEP 6: STRESS TESTING
# ====================================================
cat("\n--- STEP 6: MACRO SCENARIOS ---\n")

fc_horizon <- 8
vars <- colnames(x_reg_macro)
f_base <- matrix(NA, fc_horizon, 3); colnames(f_base) <- vars
f_bear <- matrix(NA, fc_horizon, 3); colnames(f_bear) <- vars
f_bull <- matrix(NA, fc_horizon, 3); colnames(f_bull) <- vars

for(v in vars) {
  f_m <- forecast(auto.arima(x_reg_macro[,v]), h=fc_horizon, level=95)
  f_base[,v] <- f_m$mean
  # Logic: Low GDP = Bad. High Unemp/Rate = Bad.
  if(v == "GDP_YY") {
    f_bear[,v] <- f_m$lower; f_bull[,v] <- f_m$upper
  } else {
    f_bear[,v] <- f_m$upper; f_bull[,v] <- f_m$lower
  }
}

full_fit <- auto.arima(y_rate, xreg=x_reg_macro, seasonal=TRUE)
fc_base  <- forecast(full_fit, xreg=f_base, h=fc_horizon)
fc_bear  <- forecast(full_fit, xreg=f_bear, h=fc_horizon)
fc_bull  <- forecast(full_fit, xreg=f_bull, h=fc_horizon)

# Fan Chart
d_h <- time(y_rate); d_f <- seq(max(d_h)+0.25, by=0.25, length.out=fc_horizon)
p_data <- data.frame(
  Date = c(as.numeric(d_h), rep(as.numeric(d_f), 3)),
  Rate = c(as.numeric(y_rate), as.numeric(fc_base$mean), as.numeric(fc_bear$mean), as.numeric(fc_bull$mean)),
  Type = c(rep("Hist", length(y_rate)), rep("Base", fc_horizon), rep("Bear", fc_horizon), rep("Bull", fc_horizon))
)
r_data <- data.frame(Date=as.numeric(d_f), Ymin=as.numeric(fc_bull$mean), Ymax=as.numeric(fc_bear$mean))

p_final <- ggplot() +
  geom_ribbon(data=r_data, aes(x=Date, ymin=Ymin, ymax=Ymax, fill="Range"), alpha=0.15) +
  geom_line(data=p_data, aes(x=Date, y=Rate, color=Type, linetype=Type), size=1) +
  scale_fill_manual(values="red") +
  scale_color_manual(values=c("black", "blue", "red", "green")) +
  scale_linetype_manual(values=c("solid", "solid", "dashed", "dashed")) +
  labs(title="7. Final Macro Stress Test", subtitle="Impact of GDP, Unemployment, and Rates Only", y="PCL Rate") +
  theme_minimal() + theme(legend.position="bottom")

print(p_final)
cat("\n>>> PART 1 PRESENTATION COMPLETE <<<\n")