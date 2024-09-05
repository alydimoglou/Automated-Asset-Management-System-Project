function  x = MVO(mu, Q)
    
    % Use this function to construct an example of a MVO portfolio.
    %
    % An example of an MVO implementation is given below. You can use this
    % version of MVO if you like, but feel free to modify this code as much
    % as you need to. You can also change the inputs and outputs to suit
    % your needs. 
    
    % You may use quadprog, Gurobi, or any other optimizer you are familiar
    % with. Just be sure to include comments in your code.

    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % Find the total number of assets
    n = size(Q,1); 
    
    % Set the target as the average expected return of all assets
    targetRet = mean(mu);
    
    % Disallow short sales
    lb = .01*ones(n,1); % need to change this but 
    ub = ones(n,1);

    % Add the expected return constraint *-ve because we want mu^Tx>= target
    % ret
    A = [-1 .* mu'];
    b = [-1 * targetRet];
    f = -1*mu;
    %constrain weights to sum to 1
    Aeq = ones(1,n);
    beq = 1;

    % Set the quadprog options 
    options = optimoptions( 'quadprog', 'TolFun', 1e-9, 'Display','off');
    
    % Optimal asset weights
    x = quadprog( 2 * Q, f, A, b, Aeq, beq, lb, ub, [], options);
    
    %----------------------------------------------------------------------
    
end