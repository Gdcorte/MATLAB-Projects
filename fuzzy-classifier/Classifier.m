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

%Define a classifier object
classdef Classifier < handle
    %Classifier object for fuzzy classification
       %Current limitation, points must be within [0,1] space,
       %bi-dimensional input and two classes separation only
    
    properties
        K; %Number of pertinence functions
        a; %Triangular pertinence function center
        b; %triangular pertinence functions spacing
        ruleSet; %inference ruleset
    end
    
    methods(Static)
       %Defines a triangular pertinence function
       function u = pertinence_triangular(x, a, b)
           u = max(1-abs(x-a)./b,0);  
       end
       
       function dataSetClass = example_delimiter_function_1(dataSet)
           v = (dataSet(1)+dataSet(2)-1)*(-dataSet(1)+dataSet(2));
           dataSetClass= v >=0;  
           dataSetClass=dataSetClass+1;
       end
       
       function dataSetClass = example_delimiter_function_2(dataSet)
           v = -1/4*sin(2*pi*dataSet(1))+dataSet(2)-0.5;
           dataSetClass= v >=0;  
           dataSetClass=dataSetClass+1;
       end
   end
    
    methods
        %CONSTRUCTOR
        function obj = Classifier(K)
            obj.K = K;
            
            %Determine a and b
            obj.b = 1/(K-1);
            
            obj.a = zeros(1,K);
            for i=1:K
                obj.a(1,i) = (i-1)/(K-1);
            end
            
            %Create empty ruleset matrix
            obj.ruleSet = zeros(K^2,4);
            
            %Determine possible ruleSet combination
            counter = 1;
            for i=1:K
                for j=1:K
                    obj.ruleSet(counter,1) = i; 
                    obj.ruleSet(counter,2) = j;    
                    counter = counter+1;
                end
            end
        end
        
        %t-productory norm
        function train_triangular(obj, dataSet, dataSetClass, minimum)
             if ~exist('minimum','var')
                 % third parameter does not exist, so default it to something
                  minimum = false;
             end
            
            counter = 1;
            np = size(dataSet,1);
            
            for m=1:obj.K            
                for i=1:obj.K
                    %Starting compatibility for given ruleset
                    B1 = 0; 
                    B2 = 0;
                    
                    %for each training dataSet
                    if (minimum)
                        for j=1:np
                            if (dataSetClass(j)==1)
                                B1 = B1 + min(obj.pertinence_triangular(dataSet(j,1),obj.a(m),obj.b), ...
                                              obj.pertinence_triangular(dataSet(j,2),obj.a(i),obj.b));
                            else
                                B2 = B2 + min(obj.pertinence_triangular(dataSet(j,1),obj.a(m),obj.b), ...
                                              obj.pertinence_triangular(dataSet(j,2),obj.a(i),obj.b));
                            end

                        end
                    else
                        for j=1:np
                            if (dataSetClass(j)==1)
                                B1 = B1 + obj.pertinence_triangular(dataSet(j,1),obj.a(m),obj.b)* ...
                                          obj.pertinence_triangular(dataSet(j,2),obj.a(i),obj.b);
                            else
                                B2 = B2 + obj.pertinence_triangular(dataSet(j,1),obj.a(m),obj.b)* ...
                                          obj.pertinence_triangular(dataSet(j,2),obj.a(i),obj.b);
                            end

                        end
                    end
                    
                    %Determine given ruleset compatibility
                    if(B1 > B2)
                        obj.ruleSet(counter,3) = 1;
                    else
                        obj.ruleSet(counter,3) = 2;
                    end
                    
                    %Determine Certainty degree
                    obj.ruleSet(counter,4) = abs(B1-B2)/(B1+B2);
                    
                    %update Counter
                    counter = counter + 1;
                end 
            end
            
        end
        
        %Classify data according to training
        function output = classifyData_triangular(obj, data)
            
            %Degree of certainty
            max_value = 0;
            max_idx = 1;
            
            %check ruleset
            for j=1:obj.K^2
                i1 = obj.ruleSet(j,1);
                i2 = obj.ruleSet(j,2);
                
                alpha =  obj.pertinence_triangular(data(1),obj.a(i1),obj.b)* ...
                         obj.pertinence_triangular(data(2),obj.a(i2),obj.b)* ...
                         obj.ruleSet(j,4);
                     
                %Define higher certainty class
                if(alpha > max_value)
                    max_value = alpha;
                    max_idx = j;
                end
            end
            
             output = obj.ruleSet(max_idx,3);
        end
        
    end
    
end

