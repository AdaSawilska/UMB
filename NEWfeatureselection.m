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
f_score = fisher_peptide_imp(1:important_features, :);

end




% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Liczba eliminowanych cech w każdej pętli algorytmu RFE
% x = 1;
% 
% % Liczba usuniętych cech (do porównania błędów klasyfikacji)
% elim = [1 10 20 50 100];
% 
% % Końcowa liczba najważniejszych cech
% z = 100;
% 
% e = zeros(1,3);
% E = [];
% 
% while height(originaldata)-1 > z
%     
%     % Rozdzielenie zmiennych z tabeli danych
%     features_names = originaldata(2:end, 1);
%     data = table2array(originaldata(2:end, 2:end));
% 
%     % Podział danych do trenowania i testowania (cv - cross-validation)
%     cv = cvpartition(size(data,2),'KFold',3);
%     
%     for j = 1:3
%         
%         % Indeksy próbek do testowania (~idx - do trenowania)
%         idx = test(cv,j);
% 
%         % Transpozycja macierzy (SVM przyjmuje wiersze jako próbki)
%         dataTrain = transpose(data(:, ~idx));
%         classTrain = transpose(labels(:, ~idx));
%         dataTest  = transpose(data(:, idx));
%         classTest = transpose(labels(:, idx));
%         
%         % Trening oraz test klasyfikatora
%         if(j ==1)
%             Mdl = fitcsvm(dataTrain, classTrain, 'KernelFunction', 'linear');
%             IncrementalMdl = incrementalLearner(Mdl);
%         else
%             IncrementalMdl = fit(IncrementalMdl, dataTrain, classTrain);
%         end
%         test_labels = predict(IncrementalMdl, dataTest);
% 
%         % Błąd klasyfikatora
%         e(j) = sum(logical(transpose(test_labels)-labels(idx)))/cv.TestSize(j);
%     end
%     
%     % Jeżeli liczba usuniętych cech zgadza się z wartością z wektora elim,
%     % zapisywany jest błąd klasyfikatora
%     if ismember(P-height(features_names),elim)
%         E(end+1) = mean(e);
%     end
%     
%     % Posortowanie cech ze względu na ich kryterium ważności
%     criterium = array2table(IncrementalMdl.Beta.^2);
%     peptide_imp = [features_names, criterium];
%     peptide_imp = sortrows(peptide_imp, "Var1", "descend");
% 
%     % Usunięcie x najmniej znaczących cech
%     for j = 1:x
%         name = string(table2cell(peptide_imp(end,1)));
%         originaldata(originaldata.Feature == name,:) = [];
%         peptide_imp(end,:) = [];
%     end
% end
% 





