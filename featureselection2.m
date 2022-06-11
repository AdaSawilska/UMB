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
Z = 100;

% Liczba eliminowanych cech w każdej pętli algorytmu RFE
X = 20;

l = 100;
for loop = 1:l
    loop
    [data_RFE, names_RFE] = RFE(dataT, names, labelsT, P, Z, X);
    data_fisher = fscore(transpose(dataT), transpose(labelsT), Z, originaldata);
    
    % Walidacja i porównanie metod
    [error_RFE, AUC_RFE, X1, Y1] = svm_classifier(data_RFE, labels, N);
    [error_fisher, AUC_fisher, X2, Y2] = svm_classifier(data_fisher, labels, N);

    er1(loop)=error_RFE;
    er2(loop)=AUC_RFE;
    er3(loop)=error_fisher;
    er4(loop)=AUC_fisher;
end

sum_error_RFE = sum(er1(:))/l;
sum_AUC_RFE = sum(er2(:))/l;
sum_error_fisher = sum(er3(:))/l;
sum_AUC_fisher = sum(er4(:))/l;
    
fprintf('Błąd klasyfikacji RFE: %.2f\n', sum_error_RFE)
fprintf('AUC dla RFE: %.2f\n', sum_AUC_RFE)
fprintf('Błąd klasyfikacji f-score: %.2f\n', sum_error_fisher)
fprintf('AUC dla f-score: %.2f\n', sum_AUC_fisher)
    
    % Krzywe ROC
%     figure(1)
%     p1 = plot(X1,Y1);
%     p1.LineWidth = 2;
%     title(['Krzywa ROC dla RFE (Z=', num2str(Z), ', X=', num2str(X), ')'])
%     xlabel('Swoistość')
%     ylabel('Czułość')
%     
%     figure(2)
%     p2 = plot(X2,Y2);
%     p2.LineWidth = 2;
%     title(['Krzywa ROC dla f-score (Z=', num2str(Z), ')'])
%     xlabel('Swoistość')
%     ylabel('Czułość')

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


function f_score=fscore(data,labels,selected_count, originaldata)
%f_score, f_names
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

[scores,fidx]=sort(f,'descend');
expression_matrix = originaldata(2:end, :);
f_score = table2array(expression_matrix(fidx(1:selected_count), 2:end));

end


% Błędy klasyfikatorów
function [error, AUC, X, Y] = svm_classifier(data, labels, N)

cv = cvpartition(N,'KFold',3);
e = zeros(1,3);
auc = zeros(1,3);

for j = 1:3
    
    idx = test(cv,j);

    dataTrain = transpose(data(:, ~idx));
    classTrain = transpose(labels(~idx));
    dataTest  = transpose(data(:, idx));

    % Trenowanie klasyfikatora SVM
    Mdl = fitcsvm(dataTrain, classTrain, 'KernelFunction', 'linear');
    
    % Test klasyfikatora SVM
    [test_labels, scores] = predict(Mdl, dataTest);
    
    % Obliczenie błędu klasyfikacji dla danego podzbioru
    e(j) = sum(logical(transpose(test_labels)-labels(idx)))/cv.TestSize(j);
    [X, Y, ~, auc(j)] = perfcurve(labels(idx), scores(:,2), 2);

end

% Ostateczny błąd klasyfikaji w procentach
error = mean(e)*100;
AUC = mean(auc);
end

