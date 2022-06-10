function [selected,f]=fscore(data,labels,selected_count)

classes=unique(labels);

m=mean(data,2);

b=zeros(size(m));
w=zeros(size(m));

for i=1:length(classes) 
	idx=find(labels==classes(i));
    ni=length(idx);
    
    mi = mean(data(:,idx),2);
	vi = var(data(:,idx),[],2) * ((ni-1)/ni);
    
    b = b + ni * ((mi-m).^2);
    w = w + ni * vi;
end

df1 = length(classes) - 1;
df2 = size(data,2) - length(classes);
f=(b/df1) ./ (w/df2);

[~,fidx]=sort(f,'descend');

selected=fidx(1:selected_count);
