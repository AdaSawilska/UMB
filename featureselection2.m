clear
clc

% Odczyt danych z pliku
originaldata = readtable('peptidome2_918.csv','VariableNamingRule','preserve');

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
X = 50;

[data_RFE, names_RFE] = RFE(dataT, names, labelsT, P, Z, X);

[data_f, names_f] = fisher_score(originaldata, dataT, names, labelsT, Z);

% Walidacja i porównanie metod
[error_RFE, AUC_RFE] = svm_classifier(data_RFE, labels, N);
[error_fisher, AUC_fisher] = svm_classifier(data_f, labels, N);

fprintf('Błąd klasyfikacji RFE: %.2f\n', error_RFE)
fprintf('Błąd klasyfikacji f-score: %.2f\n', error_fisher)


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Funkcje ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Algorytm RFE
function [data_RFE, names] = RFE(dataT, names, labelsT, P, Z, X)

while P > Z+X

    % Trenowanie klasyfikatora SVM
    Mdl = fitcsvm(dataT, labelsT, 'KernelFunction', 'linear');
    
    % Wyznaczenie kryterium ważności cech
    criterium = Mdl.Beta.^2;

    % Usunięcie X najmniej znaczących cech
    for i = 1:X
        [~,indeks] = min(criterium);
        dataT(:,indeks) = [];
        names(indeks,:) = [];
        criterium(indeks) = [];
    end

    P = P-X;
end

% Powtórzenie algorytmu, aby ostatecznie otrzymać Z cech
if P ~= Z
    
    Mdl = fitcsvm(dataT, labelsT, 'KernelFunction', 'linear');
    criterium = Mdl.Beta.^2;
    
    for i = 1:P-Z
        [~,indeks] = min(criterium);
        dataT(:,indeks) = [];
        names(indeks,:) = [];
        criterium(indeks) = [];
    end
end

data_RFE = transpose(dataT);
end


% Fisher score
function [f_score, f_names] = fisher_score(originaldata, dataT, names, labelsT, Z)
    
% Wyliczenie f-score dla całego zbioru
[index,scores] = fsrftest(dataT, labelsT);

% Posortowanie cech ze względu na ich kryterium ważności
fisher_peptide_imp = sortrows([index; scores].', 1, "ascend");
fisher_peptide_imp = sortrows([names, array2table(fisher_peptide_imp)], 3, "descend");

% Wybranie Z cech o największych wartościach f-score
fisher_peptide_index = fisher_peptide_imp(1:Z, 2);

expression_matrix = originaldata(2:end, :);
expression_matrix = expression_matrix(table2array(fisher_peptide_index),:);
f_score = table2array(expression_matrix(:, 2:end));
f_names = expression_matrix(:, 1);
end


% Błędy klasyfikatorów
function [error, AUC] = svm_classifier(data, labels, N)

cv = cvpartition(N,'KFold',3);
e = zeros(1,3);

for j = 1:3
    
    idx = test(cv,j);

    dataTrain = transpose(data(:, ~idx));
    classTrain = transpose(labels(~idx));
    dataTest  = transpose(data(:, idx));

    % Trenowanie klasyfikatora SVM
    Mdl = fitcsvm(dataTrain, classTrain, 'KernelFunction', 'linear');
    
    % Test klasyfikatora SVM
    test_labels = predict(Mdl, dataTest);
    
    % Obliczenie błędu klasyfikacji dla danego podzbioru
    e(j) = sum(logical(transpose(test_labels)-labels(idx)))/cv.TestSize(j);
    [X, Y, T, AUC] = perfcurve(labels(idx), transpose(test_labels), 2);
end

% Ostateczny błąd klasyfikaji w procentach
error = mean(e)*100;
end

