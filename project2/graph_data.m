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
test_set = readmatrix("data/test_set.csv");
labelR = readmatrix("data/labelR.csv");
labelT = readmatrix("data/labelT.csv");
dataT = readmatrix("data/dataTanH.csv");
dataR = readmatrix("data/dataReLu.csv");
dataRL = readmatrix("data/dataRL.csv");

pointInR = test_set(~labelR,:);
pointOutR = test_set(~(~labelR),:);
pointInT = test_set(~labelT,:);
pointOutT = test_set(~(~labelT),:);
radius = 1/sqrt(2*pi);

figure
plot(dataR(:,1), dataR(:,2), dataT(:,1), dataT(:,2))
legend("ReLu", "TanH", "interpreter","latex")
xlabel("Epochs", "interpreter","latex")
ylabel("Loss", "interpreter","latex")
saveas(gcf,"../report2/img/loss200.eps","epsc")

figure
plot(dataR(:,1), dataR(:,3), dataT(:,1), dataT(:,3))
legend({"ReLu", "TanH"},"location","southeast", "interpreter","latex")
xlabel("Epochs", "interpreter","latex")
ylabel("Accuracy in \%", "interpreter","latex")
saveas(gcf,"../report2/img/accuracy200.eps","epsc")

figure
hold on
plot(pointOutR(:,1),pointOutR(:,2),'b.','MarkerSize',10)
plot(pointInR(:,1),pointInR(:,2),'r.','MarkerSize',10)
rectangle('Position',[0.5-radius 0.5-radius 2*radius 2*radius],'Curvature',[1 1])
axis equal
xlim([0 1])
ylim([0 1])
saveas(gcf,"../report2/img/relu_out.eps","epsc")

figure
hold on
plot(pointOutT(:,1),pointOutT(:,2),'b.','MarkerSize',10)
plot(pointInT(:,1),pointInT(:,2),'r.','MarkerSize',10)
rectangle('Position',[0.5-radius 0.5-radius 2*radius 2*radius],'Curvature',[1 1])
axis equal
xlim([0 1])
ylim([0 1])
saveas(gcf,"../report2/img/tanh_out.eps","epsc")

