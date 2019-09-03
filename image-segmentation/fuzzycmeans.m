%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com

function [ c, u ] = fuzzycmeans(data, k, m, nIter0, deltaC0 )
%Method that implements the Fuzzy C-Means (FCM) algorithm
%Receives: Data for clustering (data), %Number of desired clusters (k),
%   level of cluster fuzzyness (m), maximum number of iterations nIter0,
%   Center variation sensitivity threshold (deltaC0)
%Return: Clusters data c and degree of belonging matrix

%Define problem scope
[n_point,p_dim] = size(data); %[Number of points | Problem dimension]

%Define random starting points for the algorithm
a = data;
b = data;
for i=1:p_dim %fetch from the data the maximum and minimum values
    a = (min(a));
    b = (max(b));
end
c = a + (b-a)*rand(k,p_dim); %Centers matrix

%Initial Settings
deltaC = 1e9;
nIter = 0; 
u = zeros(n_point, k);
D = u;

%Start algorithm loop
while ((nIter<=nIter0)&&(deltaC>deltaC0))
    %Calculate distance matrices
    for i=1:k
       D(:,i) = distanceNDim(data(:,:),c(i,:), p_dim);
    end
    coefs = (1./D).^(2/(m-1));
    
    %update degree of belonging matrix
    for i=1:k
        u(:,i) = coefs(:,i)./sum(coefs,2);
    end
    
    c0 = c; %Save previous center matrix
    %Update clusters centers
    for i=1:k
        c(i,:) = sum(u(:,i).^m.*data,1)./sum(u(:,i).^m,1);
    end
    
    nIter=nIter+1; %update iteration counter
    deltaC = abs(sortrows(c0)-sortrows(c)); %Calculate deviation from all centers
    %update maximum center deviation from previous iteration
    for i=1:p_dim
        deltaC=max(deltaC);
    end
    
end

end

