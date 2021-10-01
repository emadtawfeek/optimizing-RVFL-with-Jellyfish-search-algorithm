function [u,fval,NumEval,fbestvl]=jellyfish(train_X, train_y, test_X, test_y, CostFunction,Lb,Ub,nd)
    tic;
    % Set parameters for algorithm input
    Npop = 50;              % Number of jellyfish
    Max_iteration = 100;     % Maximum numbef of iterations
    Stop_mse = 0.02;        % Stop value of mse
    %Ub(1)=500;
    para = [Max_iteration, Npop, Stop_mse];
    %% Problem Definition

    nVar=nd;                           % Number of Decision Variables
    VarSize=[1 nVar];                  % Size of Decision Variables Matrix
    StopVal = para(3);                    % Stop value of mse
    if length(Lb)==1
        VarMin=Lb.*ones(1,nd);         % Lower Bound of Variables
        VarMax=Ub.*ones(1,nd);         % Upper Bound of Variables
    else
        VarMin=Lb;
        VarMax=Ub;
    end

    %% AJS Parameters
    MaxIt = para(1);%1000;        % Maximum Number of Iterations
    nPop = para(2);%50;           % Population Size
    % initialization using Logistic map using Eq. (18)
    popi=initialization(nPop,nd,VarMax,VarMin);
    % Evalue population
    for i=1:nPop
        popCost(i) = CostFunction(train_X, train_y, test_X, test_y, popi(i,:));
    end
    u = popi(1, :);
    fval = popCost(1);

    %% AJS Main Loop
    for it=1:MaxIt
        Meanvl=mean(popi,1);
        [value,index]=sort(popCost);
        BestSol=popi(index(1),:);
        BestCost=popCost(index(1));

        for i=1:nPop
            % Calculate time control c(t) using Eq. (17);
            Ar=(1-it*((1)/MaxIt))*(2*rand-1);
            if abs(Ar)>=0.5
                %% Folowing to ocean current using Eq. (11)
                newsol = popi(i)+ rand(VarSize).*(BestSol - 3*rand*Meanvl);
                % Check the boundary using Eq. (19)
                newsol = simplebounds(newsol,VarMin,VarMax);
                % Evaluation
                newsolCost = CostFunction(train_X, train_y, test_X, test_y, newsol);
                % Comparision
                if newsolCost<popCost(i)
                    popi(i,:) = newsol;
                    popCost(i)=newsolCost;
                    if popCost(i) < BestCost
                        BestCost=popCost(i);
                        BestSol = popi(i,:);
                    end
                end
            else
                %% Moving inside swarm
                if rand<=(1-Ar)
                    % Determine direction of jellyfish by Eq. (15)
                    j=i;
                    while j==i
                        j=randperm(nPop,1);
                    end
                    Step = popi(i,:) - popi(j,:);
                    if popCost(j) < popCost(i)
                        Step = -Step;
                    end
                    % Active motions (Type B) using Eq. (16)
                    newsol = popi(i,:) + rand(VarSize).*Step;
                else
                    % Passive motions (Type A) using Eq. (12)
                    newsol = popi(i,:) + 0.1*(VarMax-VarMin)*rand;
                end
                % Check the boundary using Eq. (19)
                newsol = simplebounds(newsol, VarMin,VarMax);
                % Evaluation
                newsolCost = CostFunction(train_X, train_y, test_X, test_y, newsol);
                % Comparision
                if newsolCost<popCost(i)
                    popi(i,:) = newsol;
                    popCost(i)=newsolCost;
                    if popCost(i) < BestCost
                        BestCost=popCost(i);
                        BestSol = popi(i,:);
                    end
                end
            end
        end

        %% Store Record for Current Iteration
        fbestvl(it)=BestCost;
        if it>=2000
            if abs(fbestvl(it)-fbestvl(it-100))<1e-350
                break;
            end
        end
%         if u < BestSol
%             u = BestSol;
%         end
%         if fval < fbestvl(it)
%             fval = fbestvl(it);
%         end
        BestSol(1) = round(BestSol(1));
        BestSol(2) = round(BestSol(2));
        BestSol(3) = round(BestSol(3));
%         BestSol(5) = round(BestSol(5));

        u = BestSol;
        fval = fbestvl(it);
        NumEval=it*nPop;
        fprintf('\n')
        fprintf(' - Iteration %d \n', it)
        fprintf(' * Best Solution: [ %d, %d, %d, %.4f ] \n', BestSol(1), BestSol(2), BestSol(3), BestSol(4));
        fprintf(' * Best Cost: %.4f \n', fbestvl(it));

        if fval <= StopVal
            disp('>>> Reach to Stop Condition! >>>');
            break
        end
    end
    toc;
end


%% This function is for checking boundary by using Eq. 19
function s=simplebounds(s,Lb,Ub)
    ns_tmp=s;
    I=ns_tmp<Lb;
    % Apply to the lower bound
    while sum(I)~=0
        ns_tmp(I)=Ub(I)+(ns_tmp(I)-Lb(I));
        I=ns_tmp<Lb;
    end
    % Apply to the upper bound
    J=ns_tmp>Ub;
    while sum(J)~=0
        ns_tmp(J)=Lb(J)+(ns_tmp(J)-Ub(J));
        J=ns_tmp>Ub;
    end
    % Check results
    s=ns_tmp;
end