function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%theta and grad 2*2 ka nikla conclusion me
% X          m*2 ka hai
z = X*theta;  %KYU bhai
% Matrix Multi valid hona chahiye so Reverse Logiv Laga

% h = 1 ./ (1+exp(-z));    bhai isliye toh sigmoid fn banaya
%dusre file me bhi use kar sakte hai function ko
h = sigmoid(z);
%yaha transpose kyu liya
% y' kyu (a-y)' ?????????

J = -((y'*log(h))+((1-y)'*log(1-(h))))/m ;
%---------------ANSWER------------
% z   m*2 hai so,h bhi m*2 hai 
%y m*2 hai h bhi m*2 hai so for valid multiplication.  
%gradient is dJ(theta)/d(theta) hai ullo bana rahe hai
grad = (h - y)'*X/m;
% =============================================================

end
