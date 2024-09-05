function  [mu, Q] = PCA(returns, factRet)
    
    % Use this function to perform a basic OLS regression with all factors. 
    % You can modify this function (inputs, outputs and code) as much as
    % you need to.
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % Number of observations and factors
    [T, p] = size(factRet); 
    [orderpcafactors, scores, latent] = pca(returns);
    P = 5; %based on latency which is how much variance each fact explains
    bestpcafacts = orderpcafactors(:,1:P);
    newfacts = returns*bestpcafacts;
    P_n = P;
    facts = [factRet newfacts];
    % Data matrix
    X = [ones(T,1) facts];
    
    % Regression coefficients
    B = (X' * X) \ X' * returns;
    
    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance
    ep       = returns - X * B;
    sigma_ep = 1/(T - P_n - 1) .* sum(ep .^2, 1);
    D        = diag(sigma_ep);
    
    % Factor expected returns and covariance matrix
    f_bar = mean(facts,1)';
    F     = cov(facts);
    
    % Calculate the asset expected returns and covariance matrix
    mu = a + V' * f_bar;
    Q  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
    
    %----------------------------------------------------------------------
    
end