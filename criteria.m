close all
clear all 
clc
% mex cec17_func.cpp -DWINDOWS
fitfun=str2func('AccSz');
fun_num=1;
dim=1665;
Max_iteration=5;
SearchAgents_no=5;
 lb=0;% vector of svm parameters ranges for polynomial
ub=1;
runs=1;
threshold=2;
%% Problem Definition
nVar=50;             % Number of Decision Variables
VarSize=[1 50];      % Decision Variables Matrix Size
VarMin=-100;         % Decision Variables Lower Bound
VarMax= 100;         % Decision Variables Upper Bound
%% ABC Settings
nOnlooker=SearchAgents_no;         % Number of Onlooker Bees
L=round(0.6*nVar*SearchAgents_no); % Abandonment Limit Parameter (Trial Limit)
d=1;            % Acceleration Coefficient Upper Bound
 for j=1:runs   %30 for final results [mean,best....]
%[Best_score(j),Best_pos,cg_curve]=ALO(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun,fun_num);
%[Best_score(j),Best_pos,cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun,fun_num);
% [Target_score(j),Target_pos,conver_iter]=WS(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun,fun_num,threshold);
% [Best_score(j),Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Target_score(j),Target_pos,GOA_cg_curve, Trajectories,fitness_history, position_history]=GOA(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Best_MSE(j),Best_NN,cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Best_score(j),Best_pos,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Best_score(j),Best_pos,cg_curve]=DA(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Best_score(j),Best_pos,SSA_cg_curve]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Best_score(j),Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Best_score(j),Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
% [Best_score(j),Best_pos,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
%[best(j)]=ABC(VarSize,VarMin,VarMax,fitfun,fun_num,Max_iteration,SearchAgents_no,nOnlooker,L,d);
%[Target_score(j),TargetPosition,Convergence_curve]=GOA(SearchAgents_no,Max_iteration, lb,ub, dim, fitfun,fun_num);
%[ bestX, Target_score(j) ] = BSA( fitfun,fun_num );
[Rabbit_Energy(j),Rabbit_Location,CNVG]=HHO(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
%[Xoptimum,Target_score(j)]=NNA(fitfun,fun_num,lb,ub,dim,SearchAgents_no,Max_iteration);
% [Best_flame_score(j),Best_flame_pos,Convergence_curve]=MFO(SearchAgents_no,Max_iteration,lb,ub,dim,fitfun);
%PSO
%GSA
%EHO
%WOA
 end
 figure,
 hold on
 semilogy(CNVG,'Color','b','LineWidth',4);
 title('Convergence Curve')
 xlabel('Iteration#');
 ylabel('Best Score Obtained So Far');
 axis tight
grid off
 box on
 axis tight
set(gcf, 'position' , [39         479        1727         267]);
 hold on
% display(['The best solution obtained by WS is : ', num2str(Target_pos)]);  
% display(['The best optimal value of the objective funciton found by WS is : ', num2str(Target_score)]);
% Mean_DA=sprintf('%.2E',mean(Best_score));
% SD_DA=sprintf('%.2E',std(Best_score));
% Mean_BSA=sprintf('%.2E',mean(Target_score));
% SD_BSA=sprintf('%.2E',std(Target_score));
% Mean_GOA=sprintf('%.2E',mean(Target_score));
% SD_GOA=sprintf('%.2E',std(Target_score));
%  Mean_WS=sprintf('%.2E',mean(Rabbit_Energy));
%  SD_WS=sprintf('%.2E',std(Rabbit_Energy));
% Mean_ABC=sprintf('%.2E',mean(best));
% SD_ABC=sprintf('%.2E',std(best));
%  best_ws=min(Rabbit_Energy);
%  worst_ws=max(Rabbit_Energy);
%  Median
% % Median_WS=sprintf('%.2E',Median(Best_score));
%  display(['The mean obtained by WOA is : ', num2str(best_ws)]);
%  display(['The standard  obtained by WOA is : ', num2str(worst_ws)]);
% display(['The  best obtained by WOA is : ', num2str(Best_score)]);
% display(['The worst  obtained by WOA is : ', num2str(Best_score)]);
% display(['The worst  obtained by WOA is : ', num2str(Median_WS)]);
  display([num2str(Mean_WS),'  ',num2str(SD_WS),'  ',num2str(best_ws),'  ',num2str(worst_ws)]);