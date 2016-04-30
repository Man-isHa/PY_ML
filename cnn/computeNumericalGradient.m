 function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

epsilon = 1e-4;

for i =1:length(numgrad)
    oldT = theta(i);
    theta(i)=oldT+epsilon;
    pos = J(theta);
    theta(i)=oldT-epsilon;
    neg = J(theta);
    numgrad(i) = (pos-neg)/(2*epsilon);
    theta(i)=oldT;
    if mod(i,100)==0
       fprintf('Done with %d\n',i);
    end;
end;





%% ---------------------------------------------------------------
end



% function numgrad = computeNumericalGradient(J, theta)
% %COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
% %and gives us a numerical estimate of the gradient.
% %   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
% %   gradient of the function J around theta. Calling y = J(theta) should
% %   return the function value at theta.
% 
% % Notes: The following code implements numerical gradient checking, and 
% %        returns the numerical gradient.It sets numgrad(i) to (a numerical 
% %        approximation of) the partial derivative of J with respect to the 
% %        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
% %        be the (approximately) the partial derivative of J with respect 
% %        to theta(i).)
% %                
% 
% numgrad = zeros(size(theta));
% perturb = zeros(size(theta));
% e = 1e-4;
% for p = 1:numel(theta)
%     % Set perturbation vector
%     perturb(p) = e;
%     loss1 = J(theta - perturb);
%     loss2 = J(theta + perturb);
%     % Compute Numerical Gradient
%     numgrad(p) = (loss2 - loss1) / (2*e);
%     perturb(p) = 0;
% end
% 
% end

