%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com

function D = distanceNDim(data, center, n_dim)
%method that calculates the euclidean distance between 2 sets of points
%Receives: Points p1 and p2
%Returns: Distance D

%Define vector point p1 and p2, where p1 refers to a matrix with N data points 
%and p2 to a matrix with a specified center point duplicated N times
p1 = data;
p2 = center.*ones(length(p1(:,1)), n_dim);

%Calculate the distance from all the point specified in data according to
%the specified center
D = zeros(length(p1(:,1)),1);
for i=1:n_dim
    D = D + (p1(:,i)-p2(:,i)).^2;
end
    D = sqrt(D);
 
end