%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                                                                        %
%               Implementation of Fuzzy C-Means Algorithm                % 
%                                                                        %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com


%Target vector for clustering
p1 = 1;
p2 = 100;
data = p1 + (p2-p1)*rand(800,2);

%Image Segmentation Settings
k = 15; %Number of desired clusters
m = 2; %level of cluster fuzzyness
%Convergence Criteria
nIter0 = 100; %Define maximum number of iterations
deltaC0 = 1e-3; %Define center variation sensitivity threshold 

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


deltaC = 1e9; %Starting center variation from last iteration
nIter = 0; %Starting number of iterations
u = zeros(n_point, k); %Starting degree of belonging matriz
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

%Show original data in blue and clustering centers in red for 2-Dimension problem
if p_dim ==2
    figure();
    stem(data(:,1),data(:,2),'g.', 'LineStyle','none','Linewidth',3);
    hold on;
    stem(c(:,1),c(:,2),'rO', 'LineStyle','none','Linewidth',3);
    title('Fuzzy C-means algorithm');
    legend('Modelling Data','Clustering Centers');
    xlabel('X_1');
    ylabel('X_2');
end