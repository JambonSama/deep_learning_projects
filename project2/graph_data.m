%% Clean the workspace

clc 
clearvars
close all

%% Graph for loss and accuracy
% ReLu
ReLu1 = readmatrix("data/logsReLu_1.csv");
ReLu2 = readmatrix("data/logsReLu_2.csv");
ReLu3 = readmatrix("data/logsReLu_3.csv");
ReLu4 = readmatrix("data/logsReLu_4.csv");
ReLu5 = readmatrix("data/logsReLu_5.csv");

% TanH
TanH1 = readmatrix("data/logsTanH_1.csv");
TanH2 = readmatrix("data/logsTanH_2.csv");
TanH3 = readmatrix("data/logsTanH_3.csv");
TanH4 = readmatrix("data/logsTanH_4.csv");
TanH5 = readmatrix("data/logsTanH_5.csv");

% Average
ReLuA = 1/5*(ReLu1+ReLu2+ReLu3+ReLu4+ReLu5);
TanHA = 1/5*(TanH1+TanH2+TanH3+TanH4+TanH5);

figure
plot(ReLuA(:,1), ReLuA(:,2), TanHA(:,1), TanHA(:,2))
legend("ReLu", "TanH", "interpreter","latex")
xlabel("Epochs", "interpreter","latex")
ylabel("Loss", "interpreter","latex")
saveas(gcf,"../report2/img/loss.eps","epsc")

figure
plot(ReLuA(:,1), ReLuA(:,3), TanHA(:,1), TanHA(:,3))
legend({"ReLu", "TanH"},"location","southeast", "interpreter","latex")
xlabel("Epochs", "interpreter","latex")
ylabel("Accuracy in \%", "interpreter","latex")
saveas(gcf,"../report2/img/accuracy.eps","epsc")

%% Result Neural Network

