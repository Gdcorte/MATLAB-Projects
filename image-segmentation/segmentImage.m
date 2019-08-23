%Author: Gustavo Diniz da Corte
%Contact: gustavodacorte@gmail.com

function [ segmentedImage ] = segmentImage(imageFile, k, m, nIter0, deltaC0  )
%Method that implements an image segmentation algorithm using a fuzzy c-means
% method.
%Receives: : ImageFile for segmentation, Data for clustering (data),
%Number of desired clusters (k),level of cluster fuzzyness (m), 
%maximum number of iterations nIter0, Center variation sensitivity threshold (deltaC0)
%Return: Segmented Image

%Prepare image data for processing
img_size = size(imageFile);
[img_keep, keep_index]  =max(img_size(1:2));
img_append =min(img_size(1:2));


data=zeros(1,3); %pre-insertion of value
%Convert matrix of 3 dimensions to 2 dimensions
for i=1:img_append
    if keep_index==2
        data = [data;squeeze(imageFile(i,:,:))];
    else
        data = [data;squeeze(imageFile(:,i,:))];
    end
end
data(1,:)=[];  % remove of pre-inserted value
data = double(data);

%Apply fuzzy C-means to define image segmentation
[centers,u] = fuzzycmeans(data,k,m,nIter0,deltaC0);
colors = uint8(round(centers)); %Changes clusters into color format

%Segment Image
[~,p_index] = max(u,[],2); %Retrieve clusters information for each point
segData = colors(p_index,:); %Create segmented image data

vectorkeep = img_keep*ones(size(data,1)/img_keep,1); %Create verctor for segmented data partition
rows = mat2cell(segData,vectorkeep,size(data,2)); %Split segmented data into chunks to be aggegated into final segmented image

%Convert segmented data chunks into segmented image
segImage = zeros(img_size);
for i=1:img_append
    if keep_index==2
        rows{i} = reshape(rows{i},[1,img_keep,size(data,2)]);
        segImage(i,:,:) = rows{i};
    else
        rows{i} = reshape(rows{i},[img_keep,1,size(data,2)]);
        segImage(:,i,:) = rows{i};
    end
end
segmentedImage=uint8(segImage); %Convert segmented image into unsigned integer 8bits

end

