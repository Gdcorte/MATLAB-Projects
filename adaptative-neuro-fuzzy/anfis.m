%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                                                                        %
%    Implementation of Adaptative Neuro-Fuzzy Inference System (ANFIS)   % 
%                                                                        %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com

classdef anfis < handle
    %ANFIS class handles the creation of a MISO (multiple inputs, single output) 
    %model and its uses against real datasets
    
    properties
        %(GetAccess=private)
        rules_number;   %number of rules to be applied
        model_dim;      %dimension of model
        sig;            %Sigma parameter from gaussian pertinence function
        c;              %c parameter from gaussian pertinence function
        p;              %Output coefficients for y calculation (each rule)
%     end
%     properties
        alpha = 0.1;    %Learning Rate
        itMAX = 100;    %Maximum Iteration
        acc = 1e-5;     %Minimum Accuracy
        n = 1;          %problem's dimension
    end
    
    methods(Access=private)
        %Calculates pertinence value
        function ujk_model=ujk_gaussian(obj, x, c, sig)
            ujk_model = exp(-( (x-c)^2 )/( 2*((sig)^2) ) );
        end
        
        %Determines yk value for a given rule
        %x is a vector containing all the inputs [x1, x1, x2, ..., xn]^T
        %p is the coefficients for this function [p0, p1, p2, ..., pn]
        %yk= p0 + sum_{j=0->n}(xj*pj)
        function yk_model=yk(obj, x, p)
            p0 = p(1);
            y_p = p;
            y_p(1)=[];
            yk_model= p0+x*y_p;
        end
        
        %u is the pertinence value for a given rule for all the inputs
        function wk_model=wk(obj, u)
            wk_model = prod(u);
        end
        
    end
    methods
        %CONSTRUCTOR
        function obj=anfis(ruleSet)
            obj.rules_number = ruleSet;
        end
        
        function setModel_dim(CA,valor)
           CA.model_dim = valor; 
        end
        
        function [output, Y, W]=runModel(obj, input)

            %define pertinence matrix

            U = zeros(obj.model_dim,obj.rules_number);
            for j = 1:1:obj.model_dim
                for k = 1:1:obj.rules_number
                    U(j,k) = obj.ujk_gaussian(input(j),obj.c(j,k),obj.sig(j,k));
                end
            end
        
            %define weight vector
            W = zeros(1,obj.rules_number);
            for k = 1:1:obj.rules_number
                W(k) = obj.wk(U(:,k));
            end
        
            %define model inner function output vector
            Y= zeros(1,obj.rules_number);
            for k = 1:1:obj.rules_number
                Y(k) = obj.yk(input,obj.p(:,k));
            end
        
            %Define model output
            output = (W * Y.')/( sum(W) );
        end
        
        %Uses the gaussian model to train the dataset
        function trainModelGaussian(obj, dataSet,outputSet)
            dataset_length = size(dataSet,2);
            obj.model_dim = size(dataSet,1);
            
            Mx=max(max(dataSet));
            mx=min(min(dataSet));
            %Define initial parameters State
            obj.c = mx + (Mx-mx).*rand(obj.model_dim,obj.rules_number);
            obj.sig =  mx + (Mx-mx).*rand(obj.model_dim,obj.rules_number);
            obj.p =  mx + (Mx-mx).*rand(obj.model_dim+1,obj.rules_number);
            
            %Define starting breaking conditions
            accuracy = 1e10;
            count_iter = 0; 
            
            %Define initial partial derivative matrices
            dj_dc = zeros(obj.model_dim,obj.rules_number);
            dj_dp = zeros(obj.model_dim+1,obj.rules_number); 
            dj_dsig = zeros(obj.model_dim,obj.rules_number);
            
            %Start Training
            while ( (accuracy>=obj.acc)&&(count_iter<=obj.itMAX) )  
                
                estimative=zeros(1,dataset_length);
                for i=1:dataset_length
                    %Calculate output
                    [estimative(i), Y, W] = obj.runModel(dataSet(:,i));
                    
                    %Partial derivative (Gradient Descent method)
                    sum_weight = sum(W);
                    for j = 1:1:obj.model_dim
                        for k = 1:1:obj.rules_number
                            %Common Elements
                            diff_model_real =  estimative(i) - outputSet(i);
                            diff_rule_estimate = Y(k)-estimative(i);

                            %dJ/dc_jk
                            numerator = diff_model_real * diff_rule_estimate * W(k) * (dataSet(j,i) - obj.c(j,k));
                            denominator = sum_weight * ((obj.sig(j,k))^2);
                            dj_dc(j,k) = numerator/denominator;

                            %del sigma
                            numerator =(dataSet(j,i) - obj.c(j,k));
                            denominator = (obj.sig(j,k));
                            dj_dsig(j,k) = dj_dc(j,k)*(numerator/denominator);
                        end
                    end
                    
                    %del p
                    for j = 0:1:(obj.model_dim)
                        for k = 1:1:obj.rules_number
                            if j==0
                               x_j = 1;
                            else
                               x_j = dataSet(j,i);
                            end
                            numerator = (estimative(i) - outputSet(i)) * W(k) * x_j;
                            dj_dp(j+1,k) = numerator/sum_weight;
                        end
                    end
                    %Atualiza constantes
                    obj.c = obj.c - obj.alpha*dj_dc;
                    obj.p = obj.p - obj.alpha*dj_dp;
                    obj.sig = obj.sig - obj.alpha*dj_dsig;
        
                end
                
                %Update breaking conditions
                accuracy = sum((estimative-outputSet).^2);
                count_iter = count_iter + 1;
            end         
        end

    end
    
end

