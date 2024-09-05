function  [mu, Q] = LASSO(returns, factRet)

    % (Least Absolute Shrinkage and Selection Operator) is a regularization
    % technique used in linear regression analysis to prevent overfitting
    % of the model.The basic idea of LASSO is to add a penalty term to the
    % sum of the squared errors of the regression model. This penalty term
    % is a linear function of the absolute values of the regression
    % coefficients, multiplied by a tuning parameter lambda. By increasing
    % the value of lambda, LASSO shrinks the values of the regression
    % coefficients towards zero, effectively reducing the complexity of the
    % model.
 
    %----------------------------------------------------------------------
    
    % Number of observations and factors
    [T, p] = size(factRet);  
    N = size(returns,2);
%     [orderpcafactors, scores, latent] = pca(returns);
%     P = 5; %based on latency which is how much variance each fact explains
%     bestpcafacts = orderpcafactors(:,1:P);
%     newfacts = returns*bestpcafacts;
%     P_n = P+p;
%     facts = [factRet newfacts];
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
    r_avgs = zeros(N,1);
    V_new = zeros(p,N);
    F     = cov(factRet);
    lambda = 0.0015;
    for i =1:N
        B_i = B(:,i);
        r_i = X*B_i;
        B_inew = lasso(X,r_i,'Lambda',lambda); %B_inew has wrogm dims- FIXED
        r_inew = X*B_inew;
        r_avgs(i) = mean(r_inew);
        V_new(:,i) = B_inew(2:end);
        
    end
    mu = r_avgs;
    Q = V_new' * F * V_new + D;

    % Factor expected returns and covariance matrix
    %f_bar = mean(factRet,1)';
    
    
    % Calculate the asset expected returns and covariance matrix
    %mu = a + V' * f_bar;
    %Q  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
    
    %----------------------------------------------------------------------
    
end