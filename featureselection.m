% Read data from file
originaldata = readtable(['Peptidome2_240x60.xlsx']);
data = originaldata(:, 2:end);

! 
cv = cvpartition(size(data,2),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
dataTrain = data(:, ~idx);
dataTest  = data(2:end, idx);

