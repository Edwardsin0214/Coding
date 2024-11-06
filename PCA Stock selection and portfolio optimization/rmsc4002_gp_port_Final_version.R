# RMSC 4002 Financial Data Analytics with Machine Learning
# Group 2
# Cornelius Gilbert 1155147665
# Teng Koon Dexter Woo 1155149528
# Sin Chi Man 1155159539
# Lam Yat Hin 1155157629
# Kwok Chun Nam 1155143692


library(tseries)
library(zoo)
library(psych)
library(rmgarch)
library(riskParityPortfolio)
library(PerformanceAnalytics)
library(xts)
library(quantmod)
library(nortest)
library(car)
library(moments)
library(lubridate)
library(psych)
library(ggplot2)

my_log <- file("outputs_v2.txt")
sink(my_log, append = TRUE, type = "output") # Writing console output to log file
sink(my_log, append = TRUE, type = "message")
cat(readChar(rstudioapi::getSourceEditorContext()$path, # Writing currently opened R script to file
             file.info(rstudioapi::getSourceEditorContext()$path)$size))

### Collect and Clean Data
getwd()
setwd("C:/Users/s1411/OneDrive/Desktop/Courses/RMSC 4002 Project_v7")
stock = read.csv("snp500_constituents.csv")

stock1 = stock[,1][stock[,1] != ""]
stock2 = stock[,2][stock[,2] != ""]
stock3 = stock[,3][stock[,3] != ""]
stock4 = stock[,4][stock[,4] != ""]
stock5 = stock[,5][stock[,5] != ""]
stock6 = stock[,6][stock[,6] != ""]
stock7 = stock[,7][stock[,7] != ""]


collect_clean_data = function(stock, start, end, universe=TRUE, clean=TRUE){
  if (universe){
    stock = read.csv("snp500_members.csv")[,]
  }
  list = list()
  error_stock = c()
  for (i in stock){
    tryCatch(
      expr = {
        list[[i]] = get.hist.quote(instrument=i, start=start, end=end, quote="Adjusted")
      },
      error = function(e){
        error_stock <<- append(error_stock, i)
        message("Error: Unable to get ", i, " data")
      }
    )
  }
  df = as.data.frame(do.call(merge.zoo, list))
  stock1 = setdiff(stock, error_stock)
  colnames(df) = stock1
  if (clean){
    df = df[, colSums(is.na(df))==0]
  }
  return (df)
}

subset_and_clean = function(df, stock, start, end){
  stock_available = colnames(df)[colnames(df) %in% stock]
  df = df[stock_available]
  df = df[, colSums(is.na(df))==0]
  date = (row.names(df) >= start) & (row.names(df) <= end)
  df = df[date,]
  return (df)
}


lret = function(df){
  return (as.data.frame(apply(log(df), 2, diff)))
}

sret = function(df){
  sret = data.frame(row.names=rownames(df)[-1])
  for (i in colnames(df)){
    price = df[[i]]
    sret[[i]] = diff(price)/price[-length(price)]
  }
  return (sret)
}


stock_selection = function(ret, get_var_only=FALSE){
  final_stocks = colnames(ret)
  #print(KMO(ret))
  num_of_deletion = 0
  while (1){
    ret = ret[final_stocks]
    n = length(final_stocks)
    pca = princomp(ret, cor=T, )
    var = pca$sdev^2
    t = sum(var)
    cumsum = cumsum(var/t)
    if (get_var_only){
      return (cumsum[1])
    }
    if (var[n] > 0.5){
      print(paste(c("Total of Deletion Cycle: ", num_of_deletion), collapse=" "))
      print(paste(c("Final Stocks: ", final_stocks), collapse=" "))
      return (final_stocks)
    }
    else{
      del_stocks = c()
      n_col_to_del = sum(var > 1)
      pca_loadings = pca$loadings[,(n-n_col_to_del-1):n]
      for (i in (n-n_col_to_del-1):n){
        loadings = abs(pca$loadings[,i])
        stock = final_stocks[which.max(loadings)]
        del_stocks = append(del_stocks, stock)
      }
      num_of_deletion = num_of_deletion + 1
      final_stocks = setdiff(final_stocks, del_stocks)
      #print(paste("Deletion Cycle: ", num_of_deletion))
      #print(paste(c("Delete Stocks: ", del_stocks), collapse=" "))
    }
  }
}
# Test
#sret[(row.names(sret) <= rownames(tail(df1_t,1))),]
#rownames(df1_hist_t)
#df1_hist_t[1,]
#####################################################
portfolio_construction = function(stock_selected, df, df_hist, sret, start_trading, end_trading, initial_wealth_erc = 1000, initial_wealth_equal = 1000, initial_wealth_snp = 1000){
  select_date_trading = (row.names(sret) >= start_trading) & (row.names(sret) <= end_trading)
  ret_trading = sret[select_date_trading,]
  ret_trading = ret_trading[stock_selected]
  
  select_dcc = (row.names(sret) <= end_trading)
  ret_dcc_input = sret[select_dcc,]
  ret_dcc_input = ret_dcc_input[stock_selected]
  
  #Stock Price Movement
  stock_price = df[stock_selected]
  select_date_trading = (row.names(stock_price) >= start_trading) & (row.names(stock_price) <= end_trading)
  stock_price = stock_price[select_date_trading,]
  
  #####################################################
  # Historical Price
  hist_stock_price = df_hist[stock_selected]
  
  # Perform historical simulation Price
  hist_stock_price_T <- tail(hist_stock_price, n=1)
  hist_sim_stock_price <- hist_stock_price
  for(i in 1:nrow(hist_stock_price)){
    if (i==1){
      sim_price <- hist_stock_price_T
    } else {
      sim_price <- hist_stock_price_T * hist_stock_price[i,]/hist_stock_price[i-1,]
    }
    rownames(sim_price) <- rownames(hist_stock_price)[i]
    hist_sim_stock_price <- rbind(hist_sim_stock_price, sim_price)
  }
  #####################################################
  
  ##ERP
  # Initialize a matrix to hold the portfolio weights
  weights_backtest <- matrix(NA, nrow = nrow(stock_price), ncol = ncol(stock_price))
  position_over_time <-matrix(NA, nrow = nrow(stock_price), ncol = ncol(stock_price))
  starting_index <- which(rownames(ret_dcc_input)==rownames(stock_price)[1])
  wealth_over_time_erc <- c()
  wealth_over_time_erc <- c(wealth_over_time_erc, initial_wealth_erc)
  
  
  for (i in starting_index:nrow(ret_dcc_input)) {
    if((i-starting_index) %% 22 == 0){
      spec = multispec(replicate(ncol(ret_dcc_input), ugarchspec(mean.model = list(armaOrder = c(0, 0)), variance.model = list(garchOrder = c(1, 1), model = "sGARCH"), distribution.model = "norm")))
      dcc_garch_spec = dccspec(uspec = spec, dccOrder = c(1, 1), distribution = "mvnorm")
      dccfit = dccfit(dcc_garch_spec, data = ret_dcc_input[(i-252):i,]) #use one year data
      # Get the forecasted covariance matrix
      cov_matrix = rcov(dccforecast(dccfit, n.ahead = 1, n.roll = 0))[[1]][,,1]
      rpp_vanilla <- riskParityPortfolio(cov_matrix)  # weight of each shares
      position_over_time[i-starting_index+1,] = as.numeric(rpp_vanilla$w * wealth_over_time_erc[i-starting_index+1] / stock_price[i-starting_index+1,]) # No. of each shares
    }
    else{
      position_over_time[i-starting_index+1,] = position_over_time[i-starting_index,]
    }
    weights_backtest[i-starting_index+1,] = rpp_vanilla$w
    wealth_over_time_erc <- c(wealth_over_time_erc, sum(stock_price[i-starting_index+1,]*position_over_time[i-starting_index+1,]))
  }
  wealth_over_time_erc <- wealth_over_time_erc[2:length(wealth_over_time_erc)]
  
  #####################################################################################################################################
  # Initialize as numeric vector
  historical_PNL_erc <- numeric(nrow(hist_stock_price))  
  historical_PNL_erc[1] = 0
  historical_sim_PNL_erc <- numeric(nrow(hist_stock_price))  
  historical_sim_PNL_erc[1] = 0
  historical_return_erc <- numeric(nrow(hist_stock_price))  
  historical_return_erc[1] = 0
  trading_PNL_erc <- numeric(nrow(df[stock_selected]))  
  trading_PNL_erc[1] = 0
  for (i in 2:nrow(hist_stock_price)) {
    # Finding Historical PnL
    historical_PNL_erc[i] <- rowSums(position_over_time * (hist_stock_price[i,]-hist_stock_price[i-1,]))
    historical_sim_PNL_erc[i] <- rowSums(position_over_time * (hist_sim_stock_price[504+i,]-hist_sim_stock_price[504+i-1,]))
    historical_return_erc[i] <- rowSums(position_over_time * (hist_stock_price[i,]-hist_stock_price[i-1,]))/rowSums(position_over_time * hist_stock_price[i-1,])
  }
  for (i in 2:nrow(df[stock_selected])) {
    # Finding Trading PnL
   trading_PNL_erc[i] <- wealth_over_time_erc[i]-wealth_over_time_erc[i-1] 
  }
  # Summary
  summary_erc_hist <- summary(historical_PNL_erc)
  mean_erc_hist <- mean(historical_PNL_erc)
  standard_deviation_erc_hist <- sd(historical_PNL_erc)
  kurtosis_erc_hist <- kurtosis(historical_PNL_erc)
  v_erc_hist <- round(6/kurtosis_erc_hist+4)
  hist_5th_erc <- quantile(historical_PNL_erc, 0.05, na.rm=TRUE)
  hist_10th_erc <- quantile(historical_PNL_erc, 0.1, na.rm=TRUE)
  hist_sim_VaR97.5_erc <- quantile(-historical_sim_PNL_erc, 0.975, na.rm=TRUE)
  hist_sim_VaR99_erc <- quantile(-historical_sim_PNL_erc, 0.99, na.rm=TRUE)
  norm_VaR97.5_erc <- -mean_erc_hist + qnorm(0.975)*standard_deviation_erc_hist
  norm_VaR99_erc <- -mean_erc_hist + qnorm(0.99)*standard_deviation_erc_hist
  t_VaR97.5_erc <- -mean_erc_hist + qt(0.975,df=v_erc_hist)*standard_deviation_erc_hist*sqrt((v_erc_hist-2)/v_erc_hist)
  t_VaR99_erc <- -mean_erc_hist + qt(0.99,df=v_erc_hist)*standard_deviation_erc_hist*sqrt((v_erc_hist-2)/v_erc_hist)
  Backtest_hist_sim_VaR97.5_erc <- sum(-historical_PNL_erc > hist_sim_VaR97.5_erc)
  Backtest_hist_sim_VaR99_erc <- sum(-historical_PNL_erc > hist_sim_VaR99_erc)
  Backtest_norm_VaR97.5_erc <- sum(-historical_PNL_erc > norm_VaR97.5_erc)
  Backtest_norm_VaR99_erc <- sum(-historical_PNL_erc > norm_VaR99_erc)
  Backtest_t_VaR97.5_erc <- sum(-historical_PNL_erc > t_VaR97.5_erc)
  Backtest_t_VaR99_erc <- sum(-historical_PNL_erc > t_VaR99_erc)
  results_erc_hist <- c(summary_erc_hist, "s.d." = standard_deviation_erc_hist, "5th_perc" = hist_5th_erc, "10th_perc" = hist_10th_erc) 
  results_erc_hist_VaR <- c("hist_sim_VaR" = hist_sim_VaR97.5_erc, "hist_sim_VaR" = hist_sim_VaR99_erc,
                            "norm_VaR.97.5%" = norm_VaR97.5_erc, "norm_VaR.99%" = norm_VaR99_erc,
                            "t_VaR.97.5%" = t_VaR97.5_erc, "t_VaR.99%" = t_VaR99_erc)
  results_erc_trading_backtest <- c("BT_hist_VaR975" = Backtest_hist_sim_VaR97.5_erc, "BT_hist_VaR99" = Backtest_hist_sim_VaR99_erc,
                                    "BT_norm_VaR975" = Backtest_norm_VaR97.5_erc, "BT_norm_VaR99" = Backtest_norm_VaR99_erc,
                                    "BT_t_VaR975" = Backtest_t_VaR97.5_erc, "BT_t_VaR99" = Backtest_t_VaR99_erc)
  
  
  summary_erc_trading <- summary(trading_PNL_erc)
  standard_deviation_erc_trading <- sd(trading_PNL_erc, na.rm=TRUE)
  trading_5th_erc <- quantile(trading_PNL_erc, 0.05, na.rm=TRUE)
  trading_10th_erc <- quantile(trading_PNL_erc, 0.1, na.rm=TRUE)
  results_erc_trading <- c(summary_erc_trading, "s.d." = standard_deviation_erc_trading, "5th_perc" = trading_5th_erc, "10th_perc" = trading_10th_erc)

  
  summary_erc_return <- summary(historical_return_erc)
  mean_erc_return <- mean(historical_return_erc)
  standard_deviation_erc_return <- sd(historical_return_erc)
  ks_norm_erc_return <- ks.test(historical_return_erc, pnorm, mean=0, sd=standard_deviation_erc_return) # KS Test: normal
  jb_test_erc <- jarque.test(historical_return_erc) # JB Test: normal
  #For QQt-plot
  u1 <- historical_return_erc
  s1 <- standard_deviation_erc_return
  su1 <- sort(u1)                 
  n1 <- length(u1)               
  s1 <- sd(u1)                    
  ku1 <- sum(u1^4)/(n1*s1^4)-3      
  v1 <- round(6/ku1+4)            
  v1.unrounded <- 6/ku1+4       
  i1 <- ((1:n1)-0.5)/n1           
  q1 <- qt(i1,v1)                 
  results_erc_return <- c(summary_erc_return, "s.d." = standard_deviation_erc_return)
  results_erc_normality <- c("ks.stat" = ks_norm_erc_return$statistic, "ks.p.value" = ks_norm_erc_return$p.value, "jb.stat" = jb_test_erc$statistic, "jb.p.value" = jb_test_erc$p.value)

  print(paste("----------------------------------------------------------ERC--------------------------------------------------------------"))
  print(paste("PnL during Historical period", rownames(hist_stock_price)[1], "to", rownames(hist_stock_price)[nrow(hist_stock_price)])) 
  print(round(results_erc_hist, 4))
  print(round(results_erc_hist_VaR, 4))
  print(round(results_erc_trading_backtest, 4))
  print(paste("PnL during Trading period", rownames(stock_price)[1], "to", rownames(stock_price)[nrow(stock_price)])) 
  print(round(results_erc_trading, 4))
  print(paste("Return during Historical period", rownames(hist_stock_price)[1], "to", rownames(hist_stock_price)[nrow(hist_stock_price)])) 
  print(round(results_erc_return, 4))
  print(round(results_erc_normality, 4))
  #####################################################################################################################################
  
  ##Equal Weight
  # Initialize a matrix to hold the portfolio weights
  weights_backtest_equal <- matrix(1/ncol(ret_trading), nrow = nrow(ret_trading), ncol = ncol(ret_trading))
  position_over_time_equal <-matrix(NA, nrow = nrow(ret_trading), ncol = ncol(ret_trading))
  starting_index <- which(rownames(ret_dcc_input)==rownames(ret_trading)[1])
  wealth_over_time_equal <- c()
  wealth_over_time_equal <- c(wealth_over_time_equal, initial_wealth_equal)
  
  for (i in starting_index:nrow(ret_dcc_input)) {
    if((i-starting_index) %% 22 == 0){
      position_over_time_equal[i-starting_index+1,] = as.numeric(weights_backtest_equal[i-starting_index+1,] * wealth_over_time_equal[i-starting_index+1] / stock_price[i-starting_index+1,])
    }
    else{
      position_over_time_equal[i-starting_index+1,] = position_over_time_equal[i-starting_index,]
    }
    wealth_over_time_equal <- c(wealth_over_time_equal, sum(stock_price[i-starting_index+1,]*position_over_time_equal[i-starting_index+1,]))
  }
  wealth_over_time_equal <- wealth_over_time_equal[2:length(wealth_over_time_equal)]
  
  #####################################################################################################################################
  # Initialize as numeric vector
  historical_PNL_equal <- numeric(nrow(hist_stock_price))  
  historical_PNL_equal[1] = 0
  historical_sim_PNL_equal <- numeric(nrow(hist_stock_price))  
  historical_sim_PNL_equal[1] = 0
  historical_return_equal <- numeric(nrow(hist_stock_price))  
  historical_return_equal[1] = 0
  trading_PNL_equal <- numeric(nrow(df[stock_selected]))  
  trading_PNL_equal[1] = 0
  for (i in 2:nrow(hist_stock_price)) {
    # Finding Historical PnL
    historical_PNL_equal[i] <- rowSums(position_over_time_equal * (hist_stock_price[i,]-hist_stock_price[i-1,]))
    historical_sim_PNL_equal[i] <- rowSums(position_over_time_equal * (hist_sim_stock_price[504+i,]-hist_sim_stock_price[504+i-1,]))
    historical_return_equal[i] <- rowSums(position_over_time_equal * (hist_stock_price[i,]-hist_stock_price[i-1,]))/ rowSums(position_over_time_equal * hist_stock_price[i-1,]) 
  }
  for (i in 2:nrow(df[stock_selected])) {
    # Finding Trading PnL
    trading_PNL_equal[i] <- wealth_over_time_equal[i] - wealth_over_time_equal[i-1] 
  }
  # Summary
  summary_equal_hist <- summary(historical_PNL_equal)
  mean_equal_hist <- mean(historical_PNL_equal)
  standard_deviation_equal_hist <- sd(historical_PNL_equal)
  kurtosis_equal_hist <- kurtosis(historical_PNL_equal)
  v_equal_hist <- round(6/kurtosis_equal_hist+4)
  hist_5th_equal <- quantile(historical_PNL_equal, 0.05, na.rm=TRUE)
  hist_10th_equal <- quantile(historical_PNL_equal, 0.1, na.rm=TRUE)
  hist_sim_VaR97.5_equal <- quantile(-historical_sim_PNL_equal, 0.975, na.rm=TRUE)
  hist_sim_VaR99_equal <- quantile(-historical_sim_PNL_equal, 0.99, na.rm=TRUE)
  norm_VaR97.5_equal <- -mean_equal_hist + qnorm(0.975)*standard_deviation_equal_hist
  norm_VaR99_equal <- -mean_equal_hist + qnorm(0.99)*standard_deviation_equal_hist
  t_VaR97.5_equal <- -mean_equal_hist + qt(0.975,df=v_equal_hist)*standard_deviation_equal_hist*sqrt((v_equal_hist-2)/v_equal_hist)
  t_VaR99_equal <- -mean_equal_hist + qt(0.99,df=v_equal_hist)*standard_deviation_equal_hist*sqrt((v_equal_hist-2)/v_equal_hist)
  Backtest_hist_sim_VaR97.5_equal <- sum(-historical_PNL_equal > hist_sim_VaR97.5_equal)
  Backtest_hist_sim_VaR99_equal <- sum(-historical_PNL_equal > hist_sim_VaR99_equal)
  Backtest_norm_VaR97.5_equal <- sum(-historical_PNL_equal > norm_VaR97.5_equal)
  Backtest_norm_VaR99_equal <- sum(-historical_PNL_equal > norm_VaR99_equal)
  Backtest_t_VaR97.5_equal <- sum(-historical_PNL_equal > t_VaR97.5_equal)
  Backtest_t_VaR99_equal <- sum(-historical_PNL_equal > t_VaR99_equal)
  results_equal_hist <- c(summary_equal_hist, "s.d." = standard_deviation_equal_hist, "5th_perc" = hist_5th_equal, "10th_perc" = hist_10th_equal) 
  results_equal_hist_VaR <- c("hist_VaR" = hist_sim_VaR97.5_equal, "hist_VaR" = hist_sim_VaR99_equal,
                            "norm_VaR.97.5%" = norm_VaR97.5_equal, "norm_VaR.99%" = norm_VaR99_equal,
                            "t_VaR.97.5%" = t_VaR97.5_equal, "t_VaR.99%" = t_VaR99_equal)
  results_equal_trading_backtest <- c("BT_hist_VaR975" = Backtest_hist_sim_VaR97.5_equal, "BT_hist_VaR99" = Backtest_hist_sim_VaR99_equal,
                                    "BT_norm_VaR975" = Backtest_norm_VaR97.5_equal, "BT_norm_VaR99" = Backtest_norm_VaR99_equal,
                                    "BT_t_VaR975" = Backtest_t_VaR97.5_equal, "BT_t_VaR99" = Backtest_t_VaR99_equal)
  
  
  summary_equal_trading <- summary(trading_PNL_equal)
  standard_deviation_equal_trading <- sd(trading_PNL_equal, na.rm=TRUE)
  trading_5th_equal <- quantile(trading_PNL_equal, 0.05, na.rm=TRUE)
  trading_10th_equal <- quantile(trading_PNL_equal, 0.1, na.rm=TRUE)
  results_equal_trading <- c(summary_equal_trading, "s.d." = standard_deviation_equal_trading, "5th_perc" = trading_5th_equal, "10th_perc" = trading_10th_equal)

  
  summary_equal_return <- summary(historical_return_equal)
  mean_equal_return <- mean(historical_return_equal)
  standard_deviation_equal_return <- sd(historical_return_equal)
  ks_norm_equal_return <- ks.test(historical_return_equal, pnorm, mean=0, sd=standard_deviation_equal_return) # KS Test: normal
  jb_test_equal <- jarque.test(historical_return_equal) # JB Test: normal
  #For QQt-plot
  u2 <- historical_return_equal
  s2 <- standard_deviation_equal_return
  su2 <- sort(u2)                 
  n2 <- length(u2)               
  s2 <- sd(u2)                    
  ku2 <- sum(u2^4)/(n2*s2^4)-3      
  v2 <- round(6/ku2+4)            
  v2.unrounded <- 6/ku2+4       
  i2 <- ((1:n2)-0.5)/n2           
  q2 <- qt(i2,v2)                 
  results_equal_return <- c(summary_equal_return, "s.d." = standard_deviation_equal_return)
  results_equal_normality <- c("ks.stat" = ks_norm_equal_return$statistic, "ks.p.value" = ks_norm_equal_return$p.value, "jb.stat" = jb_test_equal$statistic, "jb.p.value" = jb_test_equal$p.value)
  
  print(paste("----------------------------------------------------------Equal Weight Strategy-------------------------------------------------------------"))
  print(paste("PnL during Historical period", rownames(hist_stock_price)[1], "to", rownames(hist_stock_price)[nrow(hist_stock_price)])) 
  print(round(results_equal_hist, 4))
  print(round(results_equal_hist_VaR, 4))
  print(round(results_equal_trading_backtest, 4))
  print(paste("PnL during Trading period", rownames(stock_price)[1], "to", rownames(stock_price)[nrow(stock_price)])) 
  print(round(results_equal_trading, 4))
  print(paste("Return during Historical period", rownames(hist_stock_price)[1], "to", rownames(hist_stock_price)[nrow(hist_stock_price)])) 
  print(round(results_equal_return, 4))
  print(round(results_equal_normality, 4))
  #####################################################################################################################################  
  ##S&P500 buy and hold
  sp500 <- new.env()
  getSymbols("^GSPC", env = sp500, src = "yahoo", from = as.Date(start_trading), to = as.Date(end_trading)+1)
  GSPC <- sp500$GSPC
  GSPC1 <- get("GSPC", envir = sp500)
  GSPC2 <- with(sp500, GSPC)
  rm(GSPC1)
  rm(GSPC2)
  
  snp_index <- as.data.frame(GSPC$GSPC.Close)
  date <- row.names(snp_index)
  select_date_trading = (row.names(snp_index) >= start_trading) & (row.names(snp_index) <= end_trading)
  date <- date[select_date_trading]
  snp_index <-snp_index[select_date_trading,]
  
  wealth_over_time_snp <- c()
  wealth_over_time_snp <- c(wealth_over_time_snp, initial_wealth_snp)
  
  for (i in 1:length(snp_index)) {
    value = snp_index[i]/snp_index[1]*initial_wealth_snp
    wealth_over_time_snp <- c(wealth_over_time_snp, value)
  }
  wealth_over_time_snp <- wealth_over_time_snp[2:length(wealth_over_time_snp)]
  ###
  sp500_hist <- new.env()
  getSymbols("^GSPC", env = sp500_hist, src = "yahoo", from = as.Date(rownames(head(df_hist,1))), to = as.Date(rownames(tail(df_hist,1)))+1)
  GSPC_hist <- sp500_hist$GSPC
  GSPC1_hist <- get("GSPC", envir = sp500_hist)
  GSPC2_hist <- with(sp500_hist, GSPC_hist)
  rm(GSPC1_hist)
  rm(GSPC2_hist)
  snp_index_hist <- as.data.frame(GSPC_hist$GSPC.Close)
  date_hist <- row.names(snp_index_hist)
  select_date_trading_hist = (row.names(snp_index_hist) >= rownames(head(df_hist,1))) & (row.names(snp_index_hist) <= rownames(tail(df_hist,1)))
  date_hist <- date[select_date_trading_hist]
  snp_index_hist <-snp_index_hist[select_date_trading_hist,]
  wealth_over_time_snp_hist <- c()
  wealth_over_time_snp_hist <- c(wealth_over_time_snp_hist, initial_wealth_over_time_snp)
  for (i in 1:length(snp_index_hist)) {
    value_hist = snp_index_hist[i]/snp_index_hist[1]*initial_wealth_over_time_snp
    wealth_over_time_snp_hist <- c(wealth_over_time_snp_hist, value_hist)
  }
  wealth_over_time_snp_hist <- wealth_over_time_snp_hist[2:length(wealth_over_time_snp_hist)]
  ###
  
  #####################################################
  # Perform historical simulation Price of snp
  snp_index_hist_T <- tail(snp_index_hist, n=1)
  sim_snp_index_hist <- snp_index_hist_T
  for(i in 1:length(snp_index_hist)){
    if (i==1){
      sim_price2 <- snp_index_hist_T
    } else {
      sim_price2 <- snp_index_hist_T * snp_index_hist[i]/snp_index_hist[i-1]
    }
    names(sim_price2) <- names(snp_index_hist)[i]
    sim_snp_index_hist <- c(sim_snp_index_hist, sim_price2)
  }
  #####################################################
  
  #####################################################################################################################################
  # Initialize as numeric vector
  historical_PNL_snp <- numeric(nrow(hist_stock_price))  
  historical_PNL_snp[1] = 0
  historical_sim_PNL_snp <- numeric(nrow(hist_stock_price))  
  historical_sim_PNL_snp[1] = 0
  historical_return_snp <- numeric(nrow(hist_stock_price))  
  historical_return_snp[1] = 0
  trading_PNL_snp <- numeric(nrow(df[stock_selected]))  
  trading_PNL_snp[1] = 0
  for (i in 2:nrow(hist_stock_price)) {
    # Finding Historical PnL
    historical_PNL_snp[i] <- sum((wealth_over_time_snp_hist[1]/snp_index_hist[1])*(snp_index_hist[i]-snp_index_hist[i-1]))
    historical_sim_PNL_snp[i] <- sum((wealth_over_time_snp_hist[1]/snp_index_hist[1])*(sim_snp_index_hist[i]-sim_snp_index_hist[i-1]))
    historical_return_snp[i] <- sum((wealth_over_time_snp_hist[1]/snp_index_hist[1])*(snp_index_hist[i]-snp_index_hist[i-1]))/sum((wealth_over_time_snp_hist[1]/snp_index_hist[1])*snp_index_hist[i-1]) 
  }
  for (i in 2:nrow(df[stock_selected])) {
    # Finding Trading PnL
    trading_PNL_snp[i] <- wealth_over_time_snp[i]-wealth_over_time_snp[i-1] 
  }
  # Summary
  summary_snp_hist <- summary(historical_PNL_snp)
  mean_snp_hist <- mean(historical_PNL_snp)
  standard_deviation_snp_hist <- sd(historical_PNL_snp)
  kurtosis_snp_hist <- kurtosis(historical_PNL_snp)
  v_snp_hist <- round(6/kurtosis_snp_hist+4)
  hist_5th_snp <- quantile(historical_PNL_snp, 0.05, na.rm=TRUE)
  hist_10th_snp <- quantile(historical_PNL_snp, 0.1, na.rm=TRUE)
  hist_sim_VaR97.5_snp <- quantile(-historical_sim_PNL_snp, 0.975, na.rm=TRUE)
  hist_sim_VaR99_snp <- quantile(-historical_sim_PNL_snp, 0.99, na.rm=TRUE)
  norm_VaR97.5_snp <- -mean_snp_hist + qnorm(0.975)*standard_deviation_snp_hist
  norm_VaR99_snp <- -mean_snp_hist + qnorm(0.99)*standard_deviation_snp_hist
  t_VaR97.5_snp <- -mean_snp_hist + qt(0.975,df=v_snp_hist)*standard_deviation_snp_hist*sqrt((v_snp_hist-2)/v_snp_hist)
  t_VaR99_snp <- -mean_snp_hist + qt(0.99,df=v_snp_hist)*standard_deviation_snp_hist*sqrt((v_snp_hist-2)/v_snp_hist)
  Backtest_hist_sim_VaR97.5_snp <- sum(-historical_PNL_snp > hist_sim_VaR97.5_snp)
  Backtest_hist_sim_VaR99_snp <- sum(-historical_PNL_snp > hist_sim_VaR99_snp)
  Backtest_norm_VaR97.5_snp <- sum(-historical_PNL_snp > norm_VaR97.5_snp)
  Backtest_norm_VaR99_snp <- sum(-historical_PNL_snp > norm_VaR99_snp)
  Backtest_t_VaR97.5_snp <- sum(-historical_PNL_snp > t_VaR97.5_snp)
  Backtest_t_VaR99_snp <- sum(-historical_PNL_snp > t_VaR99_snp)
  results_snp_hist <- c(summary_snp_hist, "s.d." = standard_deviation_snp_hist, "5th_perc" = hist_5th_snp, "10th_perc" = hist_10th_snp) 
  results_snp_hist_VaR <- c("hist_VaR" = hist_sim_VaR97.5_snp, "hist_VaR" = hist_sim_VaR99_snp,
                              "norm_VaR.97.5%" = norm_VaR97.5_snp, "norm_VaR.99%" = norm_VaR99_snp,
                              "t_VaR.97.5%" = t_VaR97.5_snp, "t_VaR.99%" = t_VaR99_snp)
  results_snp_trading_backtest <- c("BT_hist_VaR975" = Backtest_hist_sim_VaR97.5_snp, "BT_hist_VaR99" = Backtest_hist_sim_VaR99_snp,
                                      "BT_norm_VaR975" = Backtest_norm_VaR97.5_snp, "BT_norm_VaR99" = Backtest_norm_VaR99_snp,
                                      "BT_t_VaR975" = Backtest_t_VaR97.5_snp, "BT_t_VaR99" = Backtest_t_VaR99_snp)
  
  summary_snp_trading <- summary(trading_PNL_snp)
  standard_deviation_snp_trading <- sd(trading_PNL_snp, na.rm=TRUE)
  trading_5th_snp <- quantile(trading_PNL_snp, 0.05, na.rm=TRUE)
  trading_10th_snp <- quantile(trading_PNL_snp, 0.1, na.rm=TRUE)
  results_snp_trading <- c(summary_snp_trading, "s.d." = standard_deviation_snp_trading, "5th_perc" = trading_5th_snp, "10th_perc" = trading_10th_snp)

  
  summary_snp_return <- summary(historical_return_snp)
  mean_snp_return <- mean(historical_return_snp)
  standard_deviation_snp_return <- sd(historical_return_snp)
  ks_norm_snp_return <- ks.test(historical_return_snp, pnorm, mean=0, sd=standard_deviation_snp_return) # KS Test: normal
  jb_test_snp <- jarque.test(historical_return_snp) # JB Test: normal
  #For QQt-plot
  u3 <- historical_return_snp
  s3 <- standard_deviation_snp_return
  su3 <- sort(u3)                 
  n3 <- length(u3)               
  s3 <- sd(u3)                    
  ku3 <- sum(u3^4)/(n3*s3^4)-3      
  v3 <- round(6/ku3+4)            
  v3.unrounded <- 6/ku3+4       
  i3 <- ((1:n3)-0.5)/n3           
  q3 <- qt(i3,v3)                 
  results_snp_return <- c(summary_snp_return, "s.d." = standard_deviation_snp_return)
  results_snp_normality <- c("ks.stat" = ks_norm_snp_return$statistic, "ks.p.value" = ks_norm_snp_return$p.value, "jb.stat" = jb_test_snp$statistic, "jb.p.value" = jb_test_snp$p.value)
  
  print(paste("----------------------------------------------------------Buy and Hold Strategy-------------------------------------------------------------"))
  print(paste("PnL during Historical period", rownames(hist_stock_price)[1], "to", rownames(hist_stock_price)[nrow(hist_stock_price)])) 
  print(round(results_snp_hist, 4))
  print(round(results_snp_hist_VaR, 4))
  print(round(results_snp_trading_backtest, 4))
  print(paste("PnL during Trading period", rownames(stock_price)[1], "to", rownames(stock_price)[nrow(stock_price)])) 
  print(round(results_snp_trading, 4))
  print(paste("Return during Historical period", rownames(hist_stock_price)[1], "to", rownames(hist_stock_price)[nrow(hist_stock_price)])) 
  print(round(results_snp_return, 4))
  print(round(results_snp_normality, 4))
  
  #####################################################################################################################################  
  # Graphs
  layout_matrix <- rbind(c(1, 1, 1),
                         c(2, 3, 4),
                         c(5, 6, 7),
                         c(8, 8, 8))
  heights <- c(2.5, 1.5, 1.5, 2.5)  
  layout(layout_matrix, heights = heights)
  
  #(1) Create an empty plot with the desired x and y ranges, suppressing the x-axis
  par(mar=c(5, 4, 4, 2))
  plot(x = as.Date(date), y = wealth_over_time_erc, type = "n", ylim = c(min(min(wealth_over_time_erc), min(wealth_over_time_snp), min(wealth_over_time_equal)), 
                                                                         max(max(wealth_over_time_erc), max(wealth_over_time_snp), max(wealth_over_time_equal))), 
       xlab = "", ylab = "Portfolio Values", main = "Portfolio Comparison", xaxt = "n")
  # Add the lines to the plot
  lines(as.Date(date), wealth_over_time_erc, type = "l", col = "red")
  lines(as.Date(date), wealth_over_time_snp, col = "darkgreen")
  lines(as.Date(date), wealth_over_time_equal, col = "blue")
  
  axis.Date(1, at = seq(min(as.Date(date)), max(as.Date(date)), by = "month"), format = "%Y-%m")
  mtext("Dates", side = 1, line = 2)
  # Add a legend
  legend("topleft", legend = c("ERC", "Buy and Hold S&P", "Equal-Weight"), col = c("red", "darkgreen", "blue"), lty = 1)
  ###############################################################################################
  #(2)(3)(4) Normal QQ Plot, (5)(6)(7) QQt Plot
  par(mar=c(2, 2, 2, 2))
  qqnorm(historical_PNL_erc, pch = 1, frame = FALSE)
  qqline(historical_PNL_erc, col = "steelblue", lwd = 2)
  
  par(mar=c(2, 2, 2, 2))
  qqnorm(historical_PNL_equal, pch = 1, frame = FALSE)
  qqline(historical_PNL_equal, col = "steelblue", lwd = 2)
  
  par(mar=c(2, 2, 2, 2))
  qqnorm(historical_PNL_snp, pch = 1, frame = FALSE)
  qqline(historical_PNL_snp, col = "steelblue", lwd = 2)
  
  par(mar=c(2, 2, 2, 2))
  plot(q1,su1,main="qq-t plot")
  abline(lsfit(q1,su1),col = "steelblue", lwd = 2)        
  legend("topleft", legend = paste("df =", v1), lty = 0) 
  
  par(mar=c(2, 2, 2, 2))
  plot(q2,su2,main="qq-t plot") 
  abline(lsfit(q2,su2),col = "steelblue", lwd = 2)         
  legend("topleft", legend = paste("df =", v2), lty = 0)
  
  par(mar=c(2, 2, 2, 2))
  plot(q3,su3,main="qq-t plot") 
  abline(lsfit(q3,su3),col = "steelblue", lwd = 2)         
  legend("topleft", legend = paste("df =", v3), lty = 0) 
  ###############################################################################################
  #(8) Create Density of Historical Return
  par(mar=c(5, 4, 4, 2))
  plot(density(historical_return_erc), type = "n", xlim = range(c(historical_return_erc, historical_return_equal, historical_return_snp)),
                                                   ylim = c(0, max(density(historical_return_erc)$y,density(historical_return_equal)$y,density(historical_return_snp)$y)),
       main = "Density of Historical return", xlab = "Daily return", ylab = "Density")
  # Add the lines to the plot
  lines(density(historical_return_erc), type = "l", col = "red", lwd=2)
  lines(density(historical_return_snp),type = "l", col = "darkgreen", lwd=2)
  lines(density(historical_return_equal), type = "l", col = "blue", lwd=2)
  # Add a legend
  legend("topleft", legend = c("ERC", "Buy and Hold S&P", "Equal-Weight"), col = c("red", "darkgreen", "blue"), lty = 1)
  ###############################################################################################

  ### Performance Analysis
  df_portfolio <- data.frame(Date = as.Date(date), erc = wealth_over_time_erc, equal = wealth_over_time_equal, snp = wealth_over_time_snp)
  row.names(df_portfolio) <- df_portfolio$Date
  df_portfolio$Date <- NULL
  
  port_ret = data.frame(row.names=rownames(df_portfolio)[-1])
  for (i in colnames(df_portfolio)){
    price = df_portfolio[[i]]
    port_ret[[i]] = diff(price)/price[-length(price)]
  }
  
  
  # Performance analytics
  # Initialize an empty data frame
  df_performance <- data.frame()
  
  # Loop over the portfolio names
  for (i in c('erc','equal','snp')){
    # Calculate the performance metrics
    sharpe_ratio <- as.numeric(round(SharpeRatio(as.xts(port_ret[,i,drop=FALSE]), Rf = 0, annualize = TRUE), 4)[1,])
    calmar_ratio <- as.numeric(round(CalmarRatio(as.xts(port_ret[,i,drop=FALSE]), scale = 252), 4))
    sortino_ratio <- as.numeric(round(SortinoRatio(as.xts(port_ret[,i,drop=FALSE])), 4))
    max_drawdown <- as.numeric(round(maxDrawdown(as.xts(port_ret[,i,drop=FALSE])), 4))
    tot_return <- tail(df_portfolio,1)[,i]/head(df_portfolio,1)[,i] -1
    
    # Create a data frame for the current portfolio
    df_i <- data.frame(
      Portfolio = i,
      SharpeRatio = sharpe_ratio,
      CalmarRatio = calmar_ratio,
      SortinoRatio = sortino_ratio,
      MaxDrawdown = max_drawdown,
      TotReturn = tot_return
    )
    
    # Append the data frame to the main data frame
    df_performance <- rbind(df_performance, df_i)
  }
  
  # Print the data frame
  return (list(date, wealth_over_time_erc, wealth_over_time_equal, wealth_over_time_snp))
}

##################################################################################################
portfolio_construction2 = function(stock_selected, df, df_hist, sret, start_trading, end_trading, initial_wealth_pca_small = 1000, initial_wealth_pca_large = 1000, initial_wealth_snp = 1000, initial_wealth_pca_ls = 1000, initial_wealth_equal_ls=1000){
  select_date_trading = (row.names(sret) >= start_trading) & (row.names(sret) <= end_trading)
  ret_trading = sret[select_date_trading,]
  ret_trading = ret_trading[stock_selected]
  
  select_date_ret = (row.names(sret) <= end_trading)
  ret_pca_small = sret[select_date_ret,]
  ret_pca_small = ret_pca_small[stock_selected]
  
  ret_pca_large = sret[select_date_ret,]
  
  
  ## PCA small and PCA Long/Short
  #Stock Price Movement
  stock_price = df[stock_selected]
  select_date_trading = (row.names(stock_price) >= start_trading) & (row.names(stock_price) <= end_trading)
  stock_price = stock_price[select_date_trading,]
  
  # Initialize a matrix to hold the portfolio weights
  position_over_time <-matrix(NA, nrow = nrow(stock_price), ncol = ncol(stock_price))
  position_over_time_equal_ls <-matrix(NA, nrow = nrow(stock_price), ncol = ncol(stock_price))
  position_over_time_pca_ls <-matrix(NA, nrow = nrow(stock_price), ncol = ncol(stock_price))
  
  starting_index <- which(rownames(ret_pca_small)==rownames(stock_price)[1])
  wealth_over_time_pca_small <- c()
  wealth_over_time_pca_ls <- c()
  wealth_over_time_equal_ls <- c()
  wealth_over_time_pca_small <- c(wealth_over_time_pca_small, initial_wealth_pca_small)
  wealth_over_time_pca_ls <- c(wealth_over_time_pca_ls, initial_wealth_pca_ls)
  wealth_over_time_equal_ls <- c(wealth_over_time_equal_ls, initial_wealth_equal_ls)
  
  
  for (i in starting_index:nrow(ret_pca_small)) {
    if((i-starting_index) %% 30 == 0){ #basically will only be triggered on day 1 (can adjust based on rebalancing period for current stocks)
      # Using PC1 as the weight (long-only)
      pca <- princomp(ret_pca_small[(max(1,i-504)):i,], cor = TRUE)
      pc1 <- pca$loadings[,1]
      weights <- abs(pc1)/sum(abs(pc1))
      position_over_time[i-starting_index+1,] = as.numeric(weights * wealth_over_time_pca_small[i-starting_index+1] / stock_price[i-starting_index+1,]) # No. of each shares
      
      # for long/short equal-weight and PC1 weight
      goodness_index <- list()
      for (stock in stock_selected) {
        spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), 
                           mean.model = list(armaOrder = c(0, 0), include.mean = TRUE), 
                           distribution.model = "norm")
        
        fit <- ugarchfit(spec, data = ret_pca_small[(i-252):i,stock])
        sigma_squared <- fit@fit$sigma[253]^2
        mean_return <- mean(ret_pca_small[(i-66):i,stock]) #use 3 months average return (can be changed to 1-month)
        goodness_index[[stock]] <- mean_return / sigma_squared
      }
      goodness_index_df <- data.frame(matrix(unlist(goodness_index), nrow=length(goodness_index), byrow=T))
      colnames(goodness_index_df) <-'index'
      bottom_stocks <- goodness_index_df[goodness_index_df <= quantile(goodness_index_df$index, 0.2) & goodness_index_df <= 0] #use bottom 20% to short (can be changed to 40%)
      #bottom_stocks <- goodness_index_df[goodness_index_df <= quantile(goodness_index_df[goodness_index_df <= 0], 0.2)]
      index <- match(bottom_stocks, goodness_index_df$index)
      bottom_stocks = stock_selected[index]
      
      weights_equal_ls <- rep(1/length(stock_selected), length(stock_selected))
      weights_equal_ls[index] <- -weights_equal_ls[index]
      names(weights_equal_ls) <- stock_selected
      
      weights_pca_ls <-weights
      weights_pca_ls[index] <- -weights_pca_ls[index]
      names(weights_pca_ls) <- stock_selected
      
      
      position_over_time_equal_ls[i-starting_index+1,] = as.numeric(weights_equal_ls * wealth_over_time_equal_ls[i-starting_index+1] / stock_price[i-starting_index+1,]) # No. of each shares
      position_over_time_pca_ls[i-starting_index+1,] = as.numeric(weights_pca_ls * wealth_over_time_pca_ls[i-starting_index+1] / stock_price[i-starting_index+1,]) # No. of each shares
      
    }
    else{
      position_over_time[i-starting_index+1,] = position_over_time[i-starting_index,]
      position_over_time_equal_ls[i-starting_index+1,] = position_over_time_equal_ls[i-starting_index,]
      position_over_time_pca_ls[i-starting_index+1,] = position_over_time_pca_ls[i-starting_index,]
      
    }
    wealth_over_time_pca_small <- c(wealth_over_time_pca_small, sum(stock_price[i-starting_index+1,]*position_over_time[i-starting_index+1,]))
    if(i-starting_index+1 > 1){
      wealth_over_time_pca_ls <- c(wealth_over_time_pca_ls, wealth_over_time_pca_ls[i-starting_index+1] + sum((stock_price[i-starting_index+1,]-stock_price[i-starting_index,])*position_over_time_pca_ls[i-starting_index+1,]))
      wealth_over_time_equal_ls <- c(wealth_over_time_equal_ls, wealth_over_time_equal_ls[i-starting_index+1] + sum((stock_price[i-starting_index+1,]-stock_price[i-starting_index,])*position_over_time_equal_ls[i-starting_index+1,]))
      
    } else{
      wealth_over_time_pca_ls <- c(wealth_over_time_pca_ls, wealth_over_time_pca_ls[i-starting_index+1] + sum((stock_price[i-starting_index+1,]-stock_price[i-starting_index+1,])*position_over_time_pca_ls[i-starting_index+1,]))
      wealth_over_time_equal_ls <- c(wealth_over_time_equal_ls, wealth_over_time_equal_ls[i-starting_index+1] + sum((stock_price[i-starting_index+1,]-stock_price[i-starting_index+1,])*position_over_time_equal_ls[i-starting_index+1,]))
      
    }
  }
  wealth_over_time_pca_small <- wealth_over_time_pca_small[2:length(wealth_over_time_pca_small)]
  wealth_over_time_pca_ls <- wealth_over_time_pca_ls[2:length(wealth_over_time_pca_ls)]
  wealth_over_time_equal_ls <- wealth_over_time_equal_ls[2:length(wealth_over_time_equal_ls)]
  
  ######################################
  # Initialize as numeric vector
  trading_PNL_pca_ls <- numeric(nrow(df[stock_selected]))  
  trading_PNL_pca_ls[1] = 0
  trading_PNL_equal_ls <- numeric(nrow(df[stock_selected]))  
  trading_PNL_equal_ls[1] = 0
  trading_PNL_pca_small <- numeric(nrow(df[stock_selected]))  
  trading_PNL_pca_small[1] = 0
  for (i in 2:nrow(df[stock_selected])) {
    # Finding Trading PnL
    trading_PNL_pca_ls[i] <- wealth_over_time_pca_ls[i]-wealth_over_time_pca_ls[i-1] 
    trading_PNL_equal_ls[i] <- wealth_over_time_equal_ls[i]-wealth_over_time_equal_ls[i-1]
    trading_PNL_pca_small[i] <- wealth_over_time_pca_small[i]-wealth_over_time_pca_small[i-1] 
  }
  summary_pca_ls_trading <- summary(trading_PNL_pca_ls)
  summary_equal_ls_trading <- summary(trading_PNL_equal_ls)
  summary_pca_small_trading <- summary(trading_PNL_pca_small)
  standard_deviation_pca_ls_trading <- sd(trading_PNL_pca_ls, na.rm=TRUE)
  standard_deviation_equal_ls_trading <- sd(trading_PNL_equal_ls, na.rm=TRUE)
  standard_deviation_pca_small_trading <- sd(trading_PNL_pca_small, na.rm=TRUE)
  trading_5th_pca_ls <- quantile(trading_PNL_pca_ls, 0.05, na.rm=TRUE)
  trading_10th_pca_ls <- quantile(trading_PNL_pca_ls, 0.1, na.rm=TRUE)
  trading_5th_equal_ls <- quantile(trading_PNL_equal_ls, 0.05, na.rm=TRUE)
  trading_10th_equal_ls <- quantile(trading_PNL_equal_ls, 0.1, na.rm=TRUE)
  trading_5th_pca_small <- quantile(trading_PNL_pca_small, 0.05, na.rm=TRUE)
  trading_10th_pca_small <- quantile(trading_PNL_pca_small, 0.1, na.rm=TRUE)
  
  
  results_pca_ls_trading <- c(summary_pca_ls_trading, "s.d." = standard_deviation_pca_ls_trading, "5th_perc" = trading_5th_pca_ls, "10th_perc" = trading_10th_pca_ls)
  results_equal_ls_trading <- c(summary_equal_ls_trading, "s.d." = standard_deviation_equal_ls_trading, "5th_perc" = trading_5th_equal_ls, "10th_perc" = trading_10th_equal_ls)
  results_pca_small_trading <- c(summary_pca_small_trading, "s.d." = standard_deviation_pca_small_trading, "5th_perc" = trading_5th_pca_small, "10th_perc" = trading_10th_pca_small)
  
  print(paste("PnL during Trading period", rownames(stock_price)[1], "to", rownames(stock_price)[nrow(stock_price)]))
  print(paste("---------------------------------------------------------- PC1 weight on selected stocks with L&S Strategy-------------------------------------------------------------"))
  print(round(results_pca_ls_trading, 4))
  print(paste("---------------------------------------------------------- Equal weight on selected stocks with L&S Strategy-------------------------------------------------------------"))
  print(round(results_equal_ls_trading, 4))
  print(paste("---------------------------------------------------------- PC1 weight on selected stocks Long only (Reference)-------------------------------------------------------------"))
  print(round(results_pca_small_trading, 4))
  ######################################
  
  #####################################################################################################################################
  
  ## PCA Large
  stock_price = df
  select_date_trading = (row.names(stock_price) >= start_trading) & (row.names(stock_price) <= end_trading)
  stock_price = stock_price[select_date_trading,]
  
  # Initialize a matrix to hold the portfolio weights
  position_over_time <-matrix(NA, nrow = nrow(stock_price), ncol = ncol(stock_price))
  starting_index <- which(rownames(ret_pca_large)==rownames(stock_price)[1])
  wealth_over_time_pca_large <- c()
  wealth_over_time_pca_large <- c(wealth_over_time_pca_large, initial_wealth_pca_large)
  
  for (i in starting_index:nrow(ret_pca_large)) {
    if((i-starting_index) %% 22 == 0){
      pca <- princomp(ret_pca_large[(max(1,i-504):i),], cor = TRUE)
      pc1 <- pca$loadings[,1]
      weights <- abs(pc1)/sum(abs(pc1))
      position_over_time[i-starting_index+1,] = as.numeric(weights * wealth_over_time_pca_large[i-starting_index+1] / stock_price[i-starting_index+1,]) # No. of each shares
    }
    else{
      position_over_time[i-starting_index+1,] = position_over_time[i-starting_index,]
    }
    wealth_over_time_pca_large <- c(wealth_over_time_pca_large, sum(stock_price[i-starting_index+1,]*position_over_time[i-starting_index+1,]))
  }
  wealth_over_time_pca_large <- wealth_over_time_pca_large[2:length(wealth_over_time_pca_large)]
  ######################################
  # Initialize as numeric vector
  trading_PNL_pca_large <- numeric(nrow(df[stock_selected]))  
  trading_PNL_pca_large[1] = 0
  for (i in 2:nrow(df[stock_selected])) {
    # Finding Trading PnL
    trading_PNL_pca_large[i] <- wealth_over_time_pca_large[i]-wealth_over_time_pca_large[i-1] 
  }
  summary_pca_large_trading <- summary(trading_PNL_pca_large)
  standard_deviation_pca_large_trading <- sd(trading_PNL_pca_large, na.rm=TRUE)
  trading_5th_pca_large <- quantile(trading_PNL_pca_large, 0.05, na.rm=TRUE)
  trading_10th_pca_large <- quantile(trading_PNL_pca_large, 0.1, na.rm=TRUE)
  
  results_pca_large_trading <- c(summary_pca_large_trading, "s.d." = standard_deviation_pca_large_trading, "5th_perc" = trading_5th_pca_large, "10th_perc" = trading_10th_pca_large)
  
  print(paste("---------------------------------------------------------- PC1 weight on all stock of S&P Long only (Reference)-------------------------------------------------------------"))
  print(round(results_pca_large_trading, 4))
  ######################################
  
  #####################################################################################################################################  
  ##S&P500 buy and hold
  sp500 <- new.env()
  getSymbols("^GSPC", env = sp500, src = "yahoo", from = as.Date(start_trading), to = as.Date(end_trading)+1)
  GSPC <- sp500$GSPC
  GSPC1 <- get("GSPC", envir = sp500)
  GSPC2 <- with(sp500, GSPC)
  rm(GSPC1)
  rm(GSPC2)
  
  snp_index <- as.data.frame(GSPC$GSPC.Close)
  date <- row.names(snp_index)
  select_date_trading = (row.names(snp_index) >= start_trading) & (row.names(snp_index) <= end_trading)
  date <- date[select_date_trading]
  snp_index <-snp_index[select_date_trading,]
  
  wealth_over_time_snp <- c()
  wealth_over_time_snp <- c(wealth_over_time_snp, initial_wealth_snp)
  
  for (i in 1:length(snp_index)) {
    value = snp_index[i]/snp_index[1]*initial_wealth_snp
    wealth_over_time_snp <- c(wealth_over_time_snp, value)
  }
  wealth_over_time_snp <- wealth_over_time_snp[2:length(wealth_over_time_snp)]
  ###
  sp500_hist <- new.env()
  getSymbols("^GSPC", env = sp500_hist, src = "yahoo", from = as.Date(rownames(head(df_hist,1))), to = as.Date(rownames(tail(df_hist,1)))+1)
  GSPC_hist <- sp500_hist$GSPC
  GSPC1_hist <- get("GSPC", envir = sp500_hist)
  GSPC2_hist <- with(sp500_hist, GSPC_hist)
  rm(GSPC1_hist)
  rm(GSPC2_hist)
  snp_index_hist <- as.data.frame(GSPC_hist$GSPC.Close)
  date_hist <- row.names(snp_index_hist)
  select_date_trading_hist = (row.names(snp_index_hist) >= rownames(head(df_hist,1))) & (row.names(snp_index_hist) <= rownames(tail(df_hist,1)))
  date_hist <- date[select_date_trading_hist]
  snp_index_hist <-snp_index_hist[select_date_trading_hist,]
  wealth_over_time_snp_hist <- c()
  wealth_over_time_snp_hist <- c(wealth_over_time_snp_hist, initial_wealth_snp)
  for (i in 1:length(snp_index_hist)) {
    value_hist = snp_index_hist[i]/snp_index_hist[1]*initial_wealth_snp
    wealth_over_time_snp_hist <- c(wealth_over_time_snp_hist, value_hist)
  }
  wealth_over_time_snp_hist <- wealth_over_time_snp_hist[2:length(wealth_over_time_snp_hist)]
  
  
  # Print the data frame
  return (list(date, wealth_over_time_pca_small, wealth_over_time_pca_large, wealth_over_time_snp, wealth_over_time_pca_ls, wealth_over_time_equal_ls))
}


### ---------------------
### Main 
### ---------------------

all_possible_stocks = c(stock1, stock2, stock3, stock4, stock5, stock6, stock7)
all_possible_stocks = unique(all_possible_stocks)
# df = collect_clean_data(all_possible_stocks, "2013-01-01", "2023-10-31", FALSE, FALSE)
# write.csv(df, "df_price.csv")
df = read.csv("df_price.csv", row.names=1)

stock_universe_year <- list("2017" = stock1, "2018" = stock2, "2019" = stock3, "2020" = stock4, "2021" = stock5, "2022" = stock6, "2023" = stock7)
stocks_selected_list = list()
num_stocks_selected = c()
trading_dates = c()
var_list = c()

for(year in names(stock_universe_year)) {
  initial_wealth_over_time_erc = 1000
  initial_wealth_over_time_equal = 1000
  initial_wealth_over_time_snp = 1000
  date_1_year = c()
  wealth_over_time_erc_1_year = c()
  wealth_over_time_equal_1_year = c()
  wealth_over_time_snp_1_year = c()
  finish_month = 12
  if (year == 2023){
    finish_month = 10
  }
  for(month in 1:finish_month) {
    # start of the month for the current year
    month_start <- as.Date(paste(year, month, "01", sep = "-"))
    month_end <- ceiling_date(month_start, "month") - days(1)
    
    # Look back 2 years before the start of the month
    lookback_start <- as.Date(paste(as.numeric(year) - 2, month, "01", sep = "-"))
    lookback_end <- as.Date(paste(as.numeric(year), month, "01", sep = "-")) - days(1)    
    # PCA
    df1 = subset_and_clean(df, stock_universe_year[[year]], lookback_start, lookback_end)
    lret1 = lret(df1)
    sret1 = sret(df1)
    
    #Stock selection
    stock_selected1 = stock_selection(sret1)
    get_var = stock_selection(sret1, TRUE)
    
    #### Save what stocks we selected
    stocks_selected_list[[year]] = append(stocks_selected_list[[year]], stock_selected1)
    num_stocks_selected = append(num_stocks_selected, length(stock_selected1))
    trading_dates = append(trading_dates, as.Date(paste(as.numeric(year), month, "01", sep = "-")))
    var_list = append(var_list, get_var)
    
    df_selected = df[stock_selected1]
    sret1 = sret(df_selected)
    
    df1_t = subset_and_clean(df, stock_selected1, month_start, month_end)
    df1_hist_t = subset_and_clean(df, stock_selected1, lookback_start, lookback_end)
    
    portfolio_1 = portfolio_construction(stock_selected1, df1_t, df1_hist_t, sret1, rownames(head(df1_t,1)), rownames(tail(df1_t,1)), initial_wealth_over_time_erc, initial_wealth_over_time_equal, initial_wealth_over_time_snp)
    date_1_year <- c(date_1_year, portfolio_1[[1]][1:length(portfolio_1[[1]])])
    wealth_over_time_erc_1_year <- c(wealth_over_time_erc_1_year, portfolio_1[[2]][1:length(portfolio_1[[2]])])
    wealth_over_time_equal_1_year <-c(wealth_over_time_equal_1_year, portfolio_1[[3]][1:length(portfolio_1[[3]])])
    wealth_over_time_snp_1_year <-c(wealth_over_time_snp_1_year, portfolio_1[[4]][1:length(portfolio_1[[4]])])
    
    #update inital wealth
    initial_wealth_over_time_erc = wealth_over_time_erc_1_year[length(wealth_over_time_erc_1_year)]
    initial_wealth_over_time_equal = wealth_over_time_equal_1_year[length(wealth_over_time_equal_1_year)]
    initial_wealth_over_time_snp = wealth_over_time_snp_1_year[length(wealth_over_time_snp_1_year)]
    }
  
  ### Performance Analysis
  df_portfolio <- data.frame(Date = as.Date(date_1_year), erc = wealth_over_time_erc_1_year, equal = wealth_over_time_equal_1_year, snp = wealth_over_time_snp_1_year)
  row.names(df_portfolio) <- df_portfolio$Date
  df_portfolio$Date <- NULL
  
  port_ret = data.frame(row.names=rownames(df_portfolio)[-1])
  for (i in colnames(df_portfolio)){
    price = df_portfolio[[i]]
    port_ret[[i]] = diff(price)/price[-length(price)]
  }

  par(mfrow = c(1, 1))
  # Create an empty plot with the desired x and y ranges, suppressing the x-axis
  plot(x = as.Date(date_1_year), y = wealth_over_time_erc_1_year, type = "n", ylim = c(min(min(wealth_over_time_erc_1_year), min(wealth_over_time_snp_1_year), min(wealth_over_time_equal_1_year)), 
                                                                         max(max(wealth_over_time_erc_1_year), max(wealth_over_time_snp_1_year), max(wealth_over_time_equal_1_year))), 
       xlab = "", ylab = "Portfolio Values", main = "Portfolio Comparison", xaxt = "n")
  
  # Add the lines to the plot
  lines(as.Date(date_1_year), wealth_over_time_erc_1_year, type = "l", col = "red")
  lines(as.Date(date_1_year), wealth_over_time_snp_1_year, col = "green")
  lines(as.Date(date_1_year), wealth_over_time_equal_1_year, col = "blue")
  
  axis.Date(1, at = seq(min(as.Date(date_1_year)), max(as.Date(date_1_year)), by = "month"), format = "%Y-%m")
  mtext("Dates", side = 1, line = 2)
  
  
  # Add a legend
  legend("topleft", legend = c("ERC", "Buy and Hold S&P", "Equal-Weight"), col = c("red", "green", "blue"), lty = 1)
  
  
  # Performance analytics
  # Initialize an empty data frame
  df_performance <- data.frame()
  
  # Loop over the portfolio names
  for (i in c('erc','equal','snp')){
    # Calculate the performance metrics
    sharpe_ratio <- as.numeric(round(SharpeRatio(as.xts(port_ret[,i,drop=FALSE]), Rf = 0, annualize = TRUE), 4)[1,])
    calmar_ratio <- as.numeric(round(CalmarRatio(as.xts(port_ret[,i,drop=FALSE]), scale = 252), 4))
    sortino_ratio <- as.numeric(round(SortinoRatio(as.xts(port_ret[,i,drop=FALSE])), 4))
    max_drawdown <- as.numeric(round(maxDrawdown(as.xts(port_ret[,i,drop=FALSE])), 4))
    tot_return <- tail(df_portfolio,1)[,i]/head(df_portfolio,1)[,i] -1
    
    # Create a data frame for the current portfolio
    df_i <- data.frame(
      Portfolio = i,
      SharpeRatio = sharpe_ratio,
      CalmarRatio = calmar_ratio,
      SortinoRatio = sortino_ratio,
      MaxDrawdown = max_drawdown,
      TotReturn = tot_return
    )
    
    # Append the data frame to the main data frame
    df_performance <- rbind(df_performance, df_i)
  }
  print(df_performance)
}



### ---------------------
### Main 2: Extension Long/Short
### ---------------------

all_possible_stocks = c(stock1, stock2, stock3, stock4, stock5, stock6, stock7)
all_possible_stocks = unique(all_possible_stocks)
# df = collect_clean_data(all_possible_stocks, "2013-01-01", "2023-10-31", FALSE, FALSE)
# write.csv(df, "df_price.csv")
df = read.csv("df_price.csv", row.names=1)

stock_universe_year <- list("2017" = stock1, "2018" = stock2, "2019" = stock3, "2020" = stock4, "2021" = stock5, "2022" = stock6, "2023" = stock7)

for(year in names(stock_universe_year)) {
  initial_wealth_over_time_pca_small = 1000
  initial_wealth_over_time_pca_large = 1000
  initial_wealth_over_time_snp = 1000
  initial_wealth_over_time_pca_ls = 1000
  initial_wealth_over_time_equal_ls = 1000
  
  
  date_1_year = c()
  wealth_over_time_pca_small_1_year = c()
  wealth_over_time_pca_large_1_year = c()
  wealth_over_time_snp_1_year = c()
  wealth_over_time_pca_ls_1_year = c()
  wealth_over_time_equal_ls_1_year = c()
  
  
  finish_month = 12
  if (year == 2023){
    finish_month = 10
  }
  for(month in 1:finish_month) {
    # start of the month for the current year
    month_start <- as.Date(paste(year, month, "01", sep = "-"))
    month_end <- ceiling_date(month_start, "month") - days(1)
    
    # Look back 2 years before the start of the month
    lookback_start <- as.Date(paste(as.numeric(year) - 2, month, "01", sep = "-"))
    lookback_end <- as.Date(paste(as.numeric(year), month, "01", sep = "-")) - days(1)    
    # PCA
    df1 = subset_and_clean(df, stock_universe_year[[year]], lookback_start, month_end)
    lret1 = lret(df1)
    sret1 = sret(df1)
    
    #Stock selection
    stock_selected1 = stock_selection(sret1)
    
    df1_t = subset_and_clean(df, stock_selected1, month_start, month_end)
    df1_hist_t = subset_and_clean(df, stock_selected1, lookback_start, lookback_end)
    
    portfolio_2 = portfolio_construction2(stock_selected1, df1, df1_hist_t, sret1, rownames(head(df1_t,1)), rownames(tail(df1_t,1)), initial_wealth_over_time_pca_small, initial_wealth_over_time_pca_large, initial_wealth_over_time_snp, initial_wealth_over_time_pca_ls, initial_wealth_over_time_equal_ls)
    date_1_year <- c(date_1_year, portfolio_2[[1]][1:length(portfolio_2[[1]])])
    wealth_over_time_pca_small_1_year <- c(wealth_over_time_pca_small_1_year, portfolio_2[[2]][1:length(portfolio_2[[2]])])
    wealth_over_time_pca_large_1_year <-c(wealth_over_time_pca_large_1_year, portfolio_2[[3]][1:length(portfolio_2[[3]])])
    wealth_over_time_snp_1_year <-c(wealth_over_time_snp_1_year, portfolio_2[[4]][1:length(portfolio_2[[4]])])
    wealth_over_time_pca_ls_1_year <- c(wealth_over_time_pca_ls_1_year, portfolio_2[[5]][1:length(portfolio_2[[5]])])
    wealth_over_time_equal_ls_1_year <- c(wealth_over_time_equal_ls_1_year, portfolio_2[[6]][1:length(portfolio_2[[5]])])
    
    
    #update inital wealth
    initial_wealth_over_time_pca_small = wealth_over_time_pca_small_1_year[length(wealth_over_time_pca_small_1_year)]
    initial_wealth_over_time_pca_large = wealth_over_time_pca_large_1_year[length(wealth_over_time_pca_large_1_year)]
    initial_wealth_over_time_snp = wealth_over_time_snp_1_year[length(wealth_over_time_snp_1_year)]
    initial_wealth_over_time_pca_ls = wealth_over_time_pca_ls_1_year[length(wealth_over_time_pca_ls_1_year)]
    initial_wealth_over_time_equal_ls = wealth_over_time_equal_ls_1_year[length(wealth_over_time_equal_ls_1_year)]
    
  }
  
  
  
  ### Performance Analysis
  df_portfolio <- data.frame(Date = as.Date(date_1_year), pca_small = wealth_over_time_pca_small_1_year, pca_large = wealth_over_time_pca_large_1_year, snp = wealth_over_time_snp_1_year, pca_ls =wealth_over_time_pca_ls_1_year, equal_ls =wealth_over_time_equal_ls_1_year )
  row.names(df_portfolio) <- df_portfolio$Date
  df_portfolio$Date <- NULL
  
  port_ret = data.frame(row.names=rownames(df_portfolio)[-1])
  for (i in colnames(df_portfolio)){
    price = df_portfolio[[i]]
    port_ret[[i]] = diff(price)/price[-length(price)]
  }
  
  
  # Create an empty plot with the desired x and y ranges, suppressing the x-axis
  plot(x = as.Date(date_1_year), y = wealth_over_time_pca_small_1_year, type = "n", ylim = c(min(min(wealth_over_time_pca_small_1_year), min(wealth_over_time_snp_1_year), min(wealth_over_time_pca_large_1_year), min(wealth_over_time_pca_ls_1_year), min(wealth_over_time_equal_ls_1_year)), 
                                                                                             max(max(wealth_over_time_pca_small_1_year), max(wealth_over_time_snp_1_year), max(wealth_over_time_pca_large_1_year), max(wealth_over_time_pca_ls_1_year), max(wealth_over_time_equal_ls_1_year))), 
       xlab = "", ylab = "Portfolio Values", main = "Portfolio Comparison", xaxt = "n")
  
  # Add the lines to the plot
  lines(as.Date(date_1_year), wealth_over_time_pca_small_1_year, type = "l", col = "red")
  lines(as.Date(date_1_year), wealth_over_time_snp_1_year, col = "green")
  #lines(as.Date(date_1_year), wealth_over_time_pca_large_1_year, col = "gray")
  lines(as.Date(date_1_year), wealth_over_time_pca_ls_1_year, col = "black")
  lines(as.Date(date_1_year), wealth_over_time_equal_ls_1_year, col = "blue")
  
  
  
  axis.Date(1, at = seq(min(as.Date(date_1_year)), max(as.Date(date_1_year)), by = "month"), format = "%Y-%m")
  mtext("Dates", side = 1, line = 2)
  
  
  # Add a legend
  #legend("topright", inset=c(-1, 0), legend = c("PCA after stock selection", "PCA L/S After stock selection", "Buy and Hold S&P", "Direct PCA on S&P", "Equal Weight L/S"), col = c("red", "black", "green", "blue", "grey"), lty = 1)
  legend("topleft", legend = c("PCS (PC1 Long-only on Selected Stocks)", "PCLS (PC1 Long/Short on Selected Stocks)", "B&H (Buy and Hold S&P500", "EWLS (Equal-Weight Long/Short on Selected Stocks)"), col = c("red", "black", "green", "blue"), cex = 0.7, lty = 1)
  
  
  # Performance analytics
  # Initialize an empty data frame
  df_performance <- data.frame()
  
  # Loop over the portfolio names
  for (i in c('pca_small','pca_large','snp','pca_ls','equal_ls')){
    # Calculate the performance metrics
    sharpe_ratio <- as.numeric(round(SharpeRatio(as.xts(port_ret[,i,drop=FALSE]), Rf = 0, annualize = TRUE), 4)[1,])
    calmar_ratio <- as.numeric(round(CalmarRatio(as.xts(port_ret[,i,drop=FALSE]), scale = 252), 4))
    sortino_ratio <- as.numeric(round(SortinoRatio(as.xts(port_ret[,i,drop=FALSE])), 4))
    max_drawdown <- as.numeric(round(maxDrawdown(as.xts(port_ret[,i,drop=FALSE])), 4))
    tot_return <- tail(df_portfolio,1)[,i]/head(df_portfolio,1)[,i] -1
    
    # Create a data frame for the current portfolio
    df_i <- data.frame(
      Portfolio = i,
      SharpeRatio = sharpe_ratio,
      CalmarRatio = calmar_ratio,
      SortinoRatio = sortino_ratio,
      MaxDrawdown = max_drawdown,
      TotReturn = tot_return
    )
    
    # Append the data frame to the main data frame
    df_performance <- rbind(df_performance, df_i)
  }
  print(df_performance)
}

plot(x=trading_dates, y=num_stocks_selected, type="l", 
     main="Number of Stocks Selected Over Time", xlab="Trading Months", ylab="Number of Stocks",
     xaxt="n")
abline(v=trading_dates[39], col="red")
abline(v=trading_dates[61], col="blue")
axis.Date(1, at = seq(min(trading_dates), max(trading_dates), by = "year"), format = "%Y-%m")
legend("bottomleft", legend = c("2020-03: COVID-19 Market Crash", "2022-01: End of 2-Year Bull, Start of 1-Year Bear"), col = c("red","blue"), lty=1, )
mean1 = mean(num_stocks_selected[1:39])
mean2 = mean(num_stocks_selected[39:61])
mean3 = mean(num_stocks_selected[61:82])
segments(x0 = trading_dates[1], x1 = trading_dates[39], y0 = mean1, y1 = mean1, col = "darkgreen", lty=2)
segments(x0 = trading_dates[39], x1 = trading_dates[61], y0 = mean2, y1 = mean2, col = "darkgreen", lty=2)
segments(x0 = trading_dates[61], x1 = trading_dates[82], y0 = mean3, y1 = mean3, col = "darkgreen", lty=2)

yearly_returns = function(stocks, start, end){
  p1 = subset_and_clean(df, stocks, start, end)
  s1 = sret(p1)
  rows = nrow(s1)
  prods = apply(s1 + 1, 2, prod)
  returns_in_pct = (prods ^ (1 / rows) - 1) * 100
  return (returns_in_pct)
}
starts = c("2017-01-01","2018-01-01","2019-01-01","2020-01-01","2021-01-01","2022-01-01","2023-01-01")
ends = c("2017-12-31","2018-12-31","2019-12-31","2020-12-31","2021-12-31","2022-12-31","2023-08-31")
for (i in 1:7){
  ret = yearly_returns(stocks_selected_list[[i]], starts[i], ends[i])
  freq = as.vector(table(stocks_selected_list[[i]]))
  df2 = data.frame(x=freq, y=ret)
  plotxy = ggplot(df2, aes(x=x, y=y, group=x)) +
    geom_boxplot() +
    ggtitle(paste0(i+2016, ": Stock Return vs Selected Frequency")) +
    xlab('Selected Frequency') +
    ylab('Geometric Mean of Daily Return (%)') +
    scale_x_continuous(breaks = seq(min(freq)-1, max(freq)+1, by=1))
  print(plotxy)
}

plot(trading_dates, var_list, type='l', main="Variance of PC1", xlab="Date", ylab="Variance",
     xaxt="n")
abline(v=trading_dates[39], col="red")
abline(v=trading_dates[63], col="blue")
axis.Date(1, at = seq(min(trading_dates), max(trading_dates), by = "month"), format = "%Y-%m")

sp500_hist <- new.env()
getSymbols("^GSPC", env = sp500_hist, src = "yahoo", from = as.Date("2017-01-01"), to = as.Date("2023-10-31"), periodicity = 'monthly')
GSPC_hist <- sp500_hist$GSPC
GSPC1_hist <- get("GSPC", envir = sp500_hist)
GSPC2_hist <- with(sp500_hist, GSPC_hist)
rm(GSPC1_hist)
rm(GSPC2_hist)
snp_index_hist <- as.data.frame(GSPC_hist$GSPC.Close)
date_hist <- row.names(snp_index_hist)
lines(snp_index_hist, type = "l")
par(mar=c(5, 4, 4, 6) + 0.1)
plot(trading_dates, var_list, axes=FALSE, ylim=c(min(var_list), max(var_list)), xlab="", ylab="",
     type="l", col="black", main="Variance Explained by PC1 vs S&P Index (Changepoint in 2020-03 and 2022-03)")
axis(2, ylim=c(min(var_list), max(var_list)),col="black",las=1)
mtext("Variance Explained by PC1",side=2,line=2.9)
box()
## Allow a second plot on the same graph
par(new=TRUE)
plot(trading_dates, GSPC_hist$GSPC.Close, xlab="", ylab="",
     ylim=c(min(GSPC_hist$GSPC.Close), max(GSPC_hist$GSPC.Close)), axes=FALSE, type="l", col="red")
mtext("S&P Index",side=4,col="red",line=4) 
axis(4, ylim=c(min(GSPC_hist$GSPC.Close), max(GSPC_hist$GSPC.Close)), col="red",col.axis="red",las=1)
## Draw the time axis
axis.Date(1, at = seq(min(trading_dates), max(trading_dates), by = "month"), format = "%Y-%m")
mtext("Date",side=1,col="black",line=2.5)  
abline(v=trading_dates[39], col="blue")
abline(v=trading_dates[63], col="blue")

gics = read.csv("GICS.csv")
del = setdiff(colnames(df), gics$Symbol)
df2 = df[, (colnames(df) %in% gics$Symbol)]
df3 = tail(df2, 504) 
df3 = df3[, colSums(is.na(df3))==0]
sret3 = sret(df3)
sret3 = sret3[order(names(sret3))]
pca = princomp(sret3, cor=T, )
gics = gics[gics$Symbol %in% colnames(df3),]
gics <- gics[order(gics$Symbol),]
df5 = data.frame(PC1=pca$loadings[,1], PC2=pca$loadings[,2], GICS=gics$GICS.Sector)
ggplot(df5, aes(x=PC1, y=PC2, colour=GICS)) +
  ggtitle("Biplot: PC1 vs PC2") +
  geom_point(size=2)

closeAllConnections()
plots.dir.path <- list.files(tempdir(), pattern="rs-graphics", full.names = TRUE); 
plots.png.paths <- list.files(plots.dir.path, pattern=".png", full.names = TRUE)
file.copy(from=plots.png.paths, to="C:/Users/s1411/OneDrive/Desktop/Courses/RMSC 4002 Project_v7/Final_Result_v2")






warnings()


