clear
clc

% Odczyt danych z pliku
originaldata = readtable('peptidome2_240.csv','VariableNamingRule','preserve');

data_name = originaldata.Properties.VariableNames{1};
originaldata.Properties.VariableNames{1} = 'Feature';
labels = table2array(originaldata(1, 2:end));

% Określenie liczby pętli oraz eliminowanych cech w każdej pętli
loops = 5;
k = 10;

if loops*k > size(originaldata,1)-2
    error('Za duża liczba eliminowanych cech. Zmień wartość loops lub k.')
end

for j = 1:loops
    
    % Rozdzielenie zmiennych z tabeli danych
    features_names = originaldata(2:end, 1);
    data = table2array(originaldata(2:end, 2:end));

    % Podział danych do trenowania i testowania (cv - cross-validation)
    cv = cvpartition(size(data,2),'HoldOut',0.3);
    idx = cv.test;
    
    % transpozycja macierzy - SVM przyjmuje wiersze jako próbki
    dataTrain = transpose(data(:, ~idx));
    classTrain = transpose(labels(:, ~idx));
    dataTest  = transpose(data(:, idx));
    classTest = transpose(labels(:, idx));

    % [Mdl, fitinfo] = fitclinear(dataTrain, classTrain);
    Mdl = fitcsvm(dataTrain, classTrain, 'KernelFunction', 'linear');
    test_labels = predict(Mdl, dataTest);
    
    e1 = sum(logical(transpose(test_labels)-labels(idx)))/cv.TestSize;
    %SVM_error = (e1 + e2 + e3)/3;

    % Posortowanie cech ze względu na ich kryterium ważności
    criterium = array2table(Mdl.Beta.^2);
    peptide_imp = [features_names, criterium];
    peptide_imp = sortrows(peptide_imp, "Var1", "descend");

    % Usunięcie k najmniej znaczących cech
    for i = 1:k
        name = string(table2cell(peptide_imp(end,1)));
        originaldata(originaldata.Feature == name,:) = [];
        peptide_imp(end,:) = [];
    end
end






