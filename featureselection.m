% Read data from file
originaldata = readtable('peptidome2_240.csv');

%seprate names from numeric data
features_names = originaldata(2:end, 1);
labels = originaldata(1, 2:end);
data = originaldata(2:end, 2:end);

%convert to matrics
numdata = table2array(data);
labels = table2array(labels);

% Separate to training and test data
cv = cvpartition(size(numdata,2),'HoldOut',0.3);
idx = cv.test;
%transpose() because the SVM works with rows as different cases
dataTrain = transpose(numdata(:, ~idx));
classTrain = transpose(labels(:, ~idx));
dataTest  = transpose(numdata(:, idx));
classTest = transpose(labels(:, idx));

%[Mdl, fitinfo] = fitclinear(dataTrain, classTrain);
Mdl = fitcsvm(dataTrain, classTrain, 'KernelFunction','linear');
test_labels = predict(Mdl, dataTest);

importance = array2table(Mdl.Beta);
peptide_imp = [features_names, importance];
peptide_imp = sortrows(peptide_imp, "Var1", "descend");




















