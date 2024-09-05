function  [mu, Q] = OLS(returns, factRet)
    
    % Use this function to perform a basic OLS regression with all factors. 
    % You can modify this function (inputs, outputs and code) as much as
    % you need to.
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % Number of observations and factors
    [T, p] = size(factRet); 
    
    
    % Data matrix
    X = [ones(T,1) factRet];
    
    % Regression coefficients
    B = (X' * X) \ X' * returns;
    
    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance
    ep       = returns - X * B;
    sigma_ep = 1/(T - p - 1) .* sum(ep .^2, 1);
    D        = diag(sigma_ep);
    

    % LASSO
    % i is asset number, t is total time
    %X is the matrix of returns for all factors and all time
    %B is the vector of factor coeffs for asset i, B= B(:,i)
    r_avgs = zeros(20,1);
    V_new = zeros(8,20);
    lambda = 1e-03;
    F     = cov(factRet);
    for i =1:20
        B_i = B(:,i);
        r_i = X*B_i;
        B_inew = lasso(X,r_i,'Lambda',lambda); %B_inew has wrogm dims
        r_inew = X*B_inew;
        r_avgs(i) = mean(r_inew);
        V_new(:,i) = B_inew(2:end);
        
    end
    mu = r_avgs;
    Q = V_new' * F * V_new + D;

    % Factor expected returns and covariance matrix
    f_bar = mean(factRet,1)';
    
    
    % Calculate the asset expected returns and covariance matrix
    %mu = a + V' * f_bar;
    %Q  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
    
    %----------------------------------------------------------------------
    
end