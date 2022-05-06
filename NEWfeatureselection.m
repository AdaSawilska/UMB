clear
clc

% Odczyt danych z pliku
originaldata = readtable('peptidome2_240.csv','VariableNamingRule','preserve');

data_name = originaldata.Properties.VariableNames{1};
originaldata.Properties.VariableNames{1} = 'Feature';
labels = table2array(originaldata(1, 2:end));
P = size(originaldata,1)-1;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Fisher score na originaldata (algorytm RFE usuwa cechy z originaldata)

important_features = 50;
f_score = fisher_score(originaldata, labels, important_features);

function f_score = fisher_score(originaldata, labels, important_features)
features_names = originaldata(2:end, 1);
data = table2array(originaldata(2:end, 2:end));
data  = transpose(data);
classes = transpose(labels);

%wyliczenie f-score dla całego zbioru
[index,scores] = fsrftest(data, classes);
% Posortowanie cech ze względu na ich kryterium ważności
fisher_peptide_imp = sortrows([index; scores].', 1, "ascend");
fisher_peptide_imp = sortrows([features_names, array2table(fisher_peptide_imp)], 3, "descend");
fisher_peptide_index = fisher_peptide_imp(1:important_features, 2);
expression_matrix = originaldata(2:end, : );
f_score = expression_matrix(table2array(fisher_peptide_index),:);

end




