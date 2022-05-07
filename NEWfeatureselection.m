clear
clc

% Odczyt danych z pliku
originaldata = readtable('peptidome2_240.csv','VariableNamingRule','preserve');

data_name = originaldata.Properties.VariableNames{1};
originaldata.Properties.VariableNames{1} = 'Feature';

P = size(originaldata,1)-1;
N = size(originaldata,2)-1;

% Rozdzielenie zmiennych z tabeli danych
labels = table2array(originaldata(1, 2:end));
names = originaldata(2:end, 1);
data = table2array(originaldata(2:end, 2:end));

% Transpozycja macierzy (SVM przyjmuje wiersze jako próbki)
dataT = transpose(data);
labelsT = transpose(labels);

% Końcowa liczba najważniejszych cech
Z = 50;

% Liczba eliminowanych cech w każdej pętli algorytmu RFE
X = 1;

[dataT_RFE, names_RFE] = RFE(dataT, names, labelsT, P, Z, X);

[data_f, names_f] = fisher_score(originaldata, dataT, names, labelsT, Z);


% Walidacja i porównanie metod
error_fisher = svm_cassifier(data_f, labels);
error_RFE = svm_cassifier(dataT_RFE, labels);



% Fisher
function [f_score, f_names] = fisher_score(originaldata, dataT, names, labelsT, Z)
    
% Wyliczenie f-score dla całego zbioru
[index,scores] = fsrftest(dataT, labelsT);

% Posortowanie cech ze względu na ich kryterium ważności
fisher_peptide_imp = sortrows([index; scores].', 1, "ascend");
fisher_peptide_imp = sortrows([names, array2table(fisher_peptide_imp)], 3, "descend");
fisher_peptide_index = fisher_peptide_imp(1:Z, 2);
expression_matrix = originaldata(2:end, :);
expression_matrix = expression_matrix(table2array(fisher_peptide_index),:);
f_score = table2array(expression_matrix(:, 2:end));
f_names = expression_matrix(:, 1);


end


% Algorytm RFE
function [dataT, names] = RFE(dataT, names, labelsT, P, Z, X)

while P > Z

    % Trenowanie klasyfikatora SVM
    Mdl = fitcsvm(dataT, labelsT, 'KernelFunction', 'linear');

    % Posortowanie cech ze względu na ich kryterium ważności
    criterium = Mdl.Beta.^2;

    % Usunięcie X najmniej znaczących cech
    for i = 1:X
        [~,indeks] = min(criterium);
        dataT(:,indeks) = [];
        names(indeks,:) = [];
    end

    P = P-1;
end
    
end


function error = svm_cassifier(data, labels)

cv = cvpartition(size(data,2),'KFold',3);
for j = 1:3
    idx = test(cv,j);

    dataTrain = transpose(data(:, ~idx));
    classTrain = transpose(labels(:, ~idx));
    dataTest  = transpose(data(:, idx));
    classTest = transpose(labels(:, idx));

    Mdl = fitcsvm(dataTrain, classTrain, 'KernelFunction', 'linear');
    test_labels = predict(Mdl, dataTest);
    e(j) = sum(logical(transpose(test_labels)-labels(idx)))/cv.TestSize(j);
    
end
error = e;
end

