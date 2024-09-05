function x = MIQP2(mu, Q, n)
    
    % MIQP (Mixed-Integer Quadratic Programming) is a type of optimization
    % problem that involves minimizing or maximizing a quadratic objective
    % function subject to linear constraints, where some of the variables
    % are restricted to be integer values. In other words, MIQP is a
    % quadratic programming problem with integer constraints.
    %----------------------------------------------------------------------
   
   % Find the total number of assets
   r = mu;
   N = size(Q,1);

   % Set the target as the average expected return of all assets
   % set up optimum variables to make MIQP, v is the binary var
   % and z is the slack var to make the QP linear
   xvars = 1:N;
   vvars = N+1:2*N;
   zvar = 2*N+1;
   lb = [-Inf(N,1); zeros(N+1,1)];
   ub = ones(2*N+1,1);
   ub(zvar) = Inf;

   % Impose lower and upper bound on the number of assets invested in
   % optimize_params(mu,Q);
   M = n;
   m = round(0.75*n);
   A = zeros(1,2*N+1); % Allocate A matrix
   A(vvars) = 1; % A*x represents the sum of the v(i)
   A = [A;-A];
   b = zeros(2,1); % Allocate b vector
   b(1) = M;
   b(2) = -m;
   % impose constraints on the weight of assets you do buy into
   fmin = 0.015;
   fmax = 0.42;
   % weight constraints
   Atemp = eye(N);
   Amax = horzcat(Atemp,-Atemp*fmax,zeros(N,1));
   A = [A;Amax];
   b = [b;zeros(N,1)];
   Amin = horzcat(-Atemp,Atemp*fmin,zeros(N,1));
   A = [A;Amin];
   b = [b;zeros(N,1)];
   Aeq = zeros(1,2*N+1); % Allocate Aeq matrix
   Aeq(xvars) = 1;
   beq = 1;
   % Set up the objective and set a value for the risk aversion coefficient
   lambda = 6000;
   f = [-r;zeros(N,1);lambda];
   % Set the quadprog options begin by solving the problem with the current constraints,
   % which do not yet reflect any linearization.
   options = optimoptions(@intlinprog,'Display','off'); % Suppress iterative display
   [xLinInt,fval,exitFlagInt,output] = intlinprog(f,vvars,A,b,Aeq,beq,lb,ub,options);
   % Prepare a stopping condition for the iterations:
   % stop when the slack variable z is within 0.01% of the true quadratic value
   thediff = 1e-4;
   iter = 1; % iteration counter
   assets = xLinInt(xvars); % the x variables
   truequadratic = assets'*Q*assets;
   zslack = xLinInt(zvar); % slack variable value
   options = optimoptions(options,'LPOptimalityTolerance',1e-10,'RelativeGapTolerance',1e-8,...
                     'ConstraintTolerance',1e-9,'IntegerTolerance',1e-6);
   %history for plotting
   history = [truequadratic,zslack];
   options = optimoptions(options,'LPOptimalityTolerance',1e-10,'RelativeGapTolerance',1e-8,...
                     'ConstraintTolerance',1e-9,'IntegerTolerance',1e-6);
   % finding the soln
   while abs((zslack - truequadratic)/truequadratic) > thediff % relative error
   newArow = horzcat(2*assets'*Q,zeros(1,N),-1); % Linearized constraint
   rhs = assets'*Q*assets;                       % right hand side of the linearized constraint
   A = [A;newArow];
   b = [b;rhs];
   % Solve the problem with the new constraints
   [xLinInt,fval,exitFlagInt,output] = intlinprog(f,vvars,A,b,Aeq,beq,lb,ub,options);
   assets = (assets+xLinInt(xvars))/2; % Midway from the previous to the current
%     assets = xLinInt(xvars); % Use the previous line or this one
   truequadratic = xLinInt(xvars)'*Q* xLinInt(xvars);
   zslack = xLinInt(zvar);
   history = [history;truequadratic,zslack];
   iter = iter + 1;
   end
    
   x = xLinInt(xvars);
   %----------------------------------------------------------------------
  
end
