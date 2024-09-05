function x = Project1_Function(periodReturns, periodFactRet, x0)

    % INPUTS: periodReturns, periodFactRet, x0 (current portfolio weights)
    % OUTPUTS: x (optimal portfolio)
    %
    % MIQP implementation with LASSO regression is given below.

    %----------------------------------------------------------------------

    % Example: subset the data to consistently use the most recent 3 years
    % for parameter estimation
    returns = periodReturns(end-35:end,:);
    factRet = periodFactRet(end-35:end,:);
    
    % Example: Use an LASSO regression to estimate mu and Q
    assetData  = 'MIE377_AssetPrices.csv';
      adjClose = readtable(assetData);
      adjClose = adjClose(2:end,:);
      n = size(adjClose,2);
      m = round(0.75*n);
    
    [mu, Q] = LASSO(returns, factRet);
   
    % Example: Use MIQP to optimize our portfolio
    x = MIQP2(mu, Q, n);
    %----------------------------------------------------------------------
end