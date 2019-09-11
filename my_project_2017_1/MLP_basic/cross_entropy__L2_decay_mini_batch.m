clc;
clear;


input = [0,0,0 ; 0,0,1 ; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0; 1,1,1]; % row : input dimension, column : mini batch size
output = [1,1 ; 1,0 ; 0,0 ; 0,1 ;0, 1 ; 0, 0 ; 1, 0 ; 1,1]; % row : 1, column : mini batch size
input = transpose(input);
output = transpose(output);
%%

b = 8; % mini-batch size
alpha = 0.2; % learning gain
ramda = 0.000001; % decay para

l = 4; % total layers
n1 = 3; % 1st layer input dimension
n2 = 4; % 2nd layer
n3 = 3; % 3rd layerk
n4 = 2; % 3rd layerk  out dimension



%% 난수 seed 설정
rng('default');
rng(1);

%% 선언해야 하는 변수
W1 = randn(n2,n1)*sqrt(0.1) +0;
b1 = randn(n2,1)*sqrt(0.1) +0;

W2 = randn(n3,n2)*sqrt(0.1) + 0;
b2 = randn(n3,1)*sqrt(0.1) +0;

W3 = randn(n4,n3)*sqrt(0.1) + 0;
b3 = randn(n4,1)*sqrt(0.1) +0;

a1 = zeros(n1, b);

z2 = zeros(n2, b);
a2 = zeros(n2, b);

z3 = zeros(n3, b);
a3 = zeros(n3, b);

z4 = zeros(n4, b);
a4 = zeros(n4, b);


T = zeros(n4, b);

delta4 = zeros(b,n4);
delta3 = zeros(b,n3);
delta2 = zeros(b,n2);

delta_W3 = zeros(n4,n3);
delta_b3 = zeros(n4,1);

delta_W2 = zeros(n3,n2);
delta_b2 = zeros(n3,1);

delta_W1 = zeros(n2,n1);
delta_b1 = zeros(n2,1);

%%
a1 = input;
T = output;
W = 0;
%%

z2 =W1*a1;% X -> (input dimension, mini-batch), z2 -> (n2, mini-batch)
z2 = z2 + b1; % z2는 행렬 b1은 벡터
a2 = F(z2); % a2 -> (n2, mini-batch)
W = W + sum(sum(W1.*W1));
%%
z3 = W2*a2; % a2 -> (n2, mini-batch), z3 -> (n3, mini-batch)
z3 = z3 + b2; % z3는 행렬 b2는 벡터
a3 = F(z3); % a3 -> (n3, mini-batch)
W = W + sum(sum(W2.*W2));
%%
z4 = W3*a3;% a3 -> (n3, mini-batch), z4 -> (n4, mini-batch)
z4 = z4 + b3; %z4는 행렬 b3은 벡터
a4 = F(z4);  % a4 -> (n4, mini-batch)
W = W + sum(sum(W3.*W3));
%%
y = a4;

%함수를 만들어서 계산
cross_entropy = -0.5*(T.*log(y) + (1 - T).*log(1-y))/b; % cross entropy ->(n4, mini-batch)
J = sum(sum(cross_entropy));

%%
temp = (y - T)/(2*b); %temp -> (n4, mini-batch)
delta4 = transpose(temp); % delta4 -> (mini-batch, n4) 

delta3 = delta4*W3;
temp = F_inv(z3);
temp1 = transpose(temp);
delta3 = delta3.*temp1; % delta3 -> (mini-batch, n3)


delta2 = delta3*W2;
temp = F_inv(z2);
temp1 = transpose(temp);
delta2 = delta2.*temp1; % delta2 -> (mini-batch, n2) 




%%
temp = a3*delta4;
delta_W3 = transpose(temp); %delta_W3 = transpose(a3*delta4)
delta_b3 = ones(1,b)*delta4; 

W3 = W3 - alpha*(delta_W3 + ramda*W3);
b3 = b3 - alpha*transpose(delta_b3);

%%
temp = a2*delta3;
delta_W2 = transpose(temp);
delta_b2 = ones(1,b)*delta3;

W2 = W2 - alpha*(delta_W2 + ramda*W2);
b2 = b2 - alpha*transpose(delta_b2);

%%
temp = a1*delta2;
delta_W1 = transpose(temp);
delta_b1 =ones(1,b)*delta2;

W1 = W1 - alpha*(delta_W1 + ramda*W1);
b1 = b1 - alpha*transpose(delta_b1);

%%
function result = F_tanh(x)

result = (1 - exp(-2*x))./(1 + exp(-2*x));

end

function result = F_inv_tanh(x)

result = (1+ F_tanh(x)).*(1-F_tanh(x));

end

function result = F(x)

result = 1./(1 + exp(-x));

end

function result = F_inv(x)

result = F(x).*(1-F(x));

end
