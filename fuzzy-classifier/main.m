%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                                                                        %
%   Implementation of a bi-dimensional, two classes fuzzy Classifier     % 
%                                                                        %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Current limitation, points must be within [0,1] space

%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com

%Create new classifier
K = 10;
fuzzy_classifier_prod = Classifier(K);
fuzzy_classifier_min = Classifier(K);

%Define training dataset
np = 500; %Amount of training samples

dataSet = rand(np,2); %Variavel de entrada padrão
dataSetClass = zeros(np,1);%Clssiicação de variavel para cada amostra

%Apply delimitating function to training dataset
for i=1:np
    dataSetClass(i) = fuzzy_classifier_prod.example_delimiter_function_1(dataSet(i,:));
end
% 
% %Check Points
% figure();
% hold on;
% for i=1:nt
%     if(dataSetClass(i)==1)
%         plot(dataSet(i,1),dataSet(i,2),'b.','MarkerSize',10);
%     else
%         plot(dataSet(i,1),dataSet(i,2),'r.','MarkerSize',10);
%     end
% end


% %Check pertinence functions
% x  = linspace(0,1,np);
% for i=1:K
% 
%     Matriz_de_pertinencias(:,i) = fuzzy_classifier_prod.pertinence_triangular(x,a(i),b);
% 
% end
% figure();
% hold on;
% for i = 1:K
% 
%     plot(x,Matriz_de_pertinencias(:,i),'k-');
% 
% end
% hold off;

%Test Classifier
fuzzy_classifier_prod.train_triangular(dataSet,dataSetClass);
fuzzy_classifier_min.train_triangular(dataSet,dataSetClass, true);

%Create new testing Set
nt = 600; %amount of testing samples
testingData = rand(nt,2);
testingDataClassCheck = zeros(nt,1);
classifiedClass_prod = zeros(nt,1);
classifiedClass_min = zeros(nt,1);

%Apply testing set
for i=1:nt
    %Check data real class
    testingDataClassCheck(i) = fuzzy_classifier_prod.example_delimiter_function_1(testingData(i,:));
    
    %Use Classifier
    classifiedClass_prod(i) = fuzzy_classifier_prod.classifyData_triangular(testingData(i,:));
    classifiedClass_min(i) = fuzzy_classifier_min.classifyData_triangular(testingData(i,:));
end

%check results (productory)
figure();
hold on;
for i=1:nt
    if(classifiedClass_prod(i)==1)
        plot(testingData(i,1),testingData(i,2),'bo','MarkerSize',10);
    else
        plot(testingData(i,1),testingData(i,2),'ro','MarkerSize',10);
    end
    if(testingDataClassCheck(i)==1)
        plot(testingData(i,1),testingData(i,2),'b.','MarkerSize',10);
    else
        plot(testingData(i,1),testingData(i,2),'r.','MarkerSize',10);
    end
end
title('T-Product');

%check results (minimum)
figure();
hold on;
for i=1:nt
    if(classifiedClass_min(i)==1)
        plot(testingData(i,1),testingData(i,2),'bo','MarkerSize',10);
    else
        plot(testingData(i,1),testingData(i,2),'ro','MarkerSize',10);
    end
    if(testingDataClassCheck(i)==1)
        plot(testingData(i,1),testingData(i,2),'b.','MarkerSize',10);
    else
        plot(testingData(i,1),testingData(i,2),'r.','MarkerSize',10);
    end
end
title('T-Minimum');