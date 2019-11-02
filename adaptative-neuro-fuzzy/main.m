%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                                                                        %
%    Implementation of Adaptative Neuro-Fuzzy Inference System (ANFIS)   % 
%                                                                        %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com

%Create new ANFIS object
Nr = 50; %Number of Rules
model = anfis(Nr);

%Create training dataset
m = 100; %Sample ammount
syms f(x)
% f(x) = x.^2; %Target Function
f(x) = sin(x);
x_0 = -10; %Starting Point
x_f = 10; %Ending Point
x = linspace(x_0,x_f,m);
y=double(f(x));

%Algorithm Startup
model.trainModelGaussian(x,y);

%Algorithm Testing
m = m;
x = linspace(x_0,x_f,m);
y=double(f(x));
testing = zeros(1,m);
for i = 1:1:m %for each data point
        testing(i) = model.runModel(x(i));
end

%Comparison between data:
figure();
plot(x,y,'-r');
hold on;
plot(x,testing,'.g');
xlabel('x');
ylabel('Output');
legend('Original Function','Approximate Model');
title('ANFIS approximation');