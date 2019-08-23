%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                                                                        %
%          Implementation of Image Segmentation Algorithm                % 
%                                                                        %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com

%Image Segmentation Settings
[fileName, pathDir] = uigetfile('*.jpg'); %Get target image path
[imageFile] = imread(strcat(pathDir,fileName)); %Read target image from file
k = 5; %Number of desired clusters
m = 2; %level of cluster fuzzyness
%Convergence Criteria
nIter0 = 100; %Define maximum number of iterations
deltaC0 = 1e-3; %Define center variation sensitivity threshold 

%Segment Image
segImage = segmentImage(imageFile, k, m, nIter0, deltaC0);

%Show segmented image
imshow(segImage);
