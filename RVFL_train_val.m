function [eval_result]=RVFL_train_val(trainX,trainY,testX,testY,option, FINAL)

% This is the  function to train and evaluate RVFL for classification
% problem.
% Option.N :      number of hidden neurons
% Option.bias:    whether to have bias in the output neurons
% option.link:    whether to have the direct link.
% option.ActivationFunction:Activation Functions used.
% option.seed:    Random Seeds
% option,mode     1: regularized least square, 2: Moore-Penrose pseudoinverse
% option.RandomType: different randomnization methods. Currently only support Gaussian and uniform.
% option.Scale    Linearly scale the random features before feedinto the nonlinearactivation function.
%                 In this implementation, we consider the threshold which lead to 0.99 of the maximum/minimum value of the activation function as the saturating threshold.
%                 Option.Scale=0.9 means all the random features will be linearly scaled
%                 into 0.9* [lower_saturating_threshold,upper_saturating_threshold].
%option.Scalemode Scalemode=1 will scale the features for all neurons.
%                 Scalemode=2  will scale the features for each hidden
%                 neuron separately.
%                 Scalemode=3 will scale the range of the randomization for
%                 uniform diatribution.

% This software package has been developed by Le Zhang(c) 2015
% based on this paper: A Comprehensive Evaluation of Random Vector Functional Link Neural Network Variants
%  For technical support and/or help, please contact Lzhang027@e.ntu.edu.sg
% This package has been downloaed from https://sites.google.com/site/zhangleuestc/
if nargin == 5
    FINAL = 0;
end
if ~isfield(option,'N')|| isempty(option.N)
    option.N=100;
end
if ~isfield(option,'bias')|| isempty(option.bias)
    option.bias=false;
end
if ~isfield(option,'link')|| isempty(option.link)
    option.link=true;
end
if ~isfield(option,'ActivationFunction')|| isempty(option.ActivationFunction)
    option.ActivationFunction='radbas';
end
if ~isfield(option,'seed')|| isempty(option.seed)
    option.seed=0;
end
if ~isfield(option,'RandomType')|| isempty(option.RandomType)
    option.RandomType='Uniform';
end
if ~isfield(option,'mode')|| isempty(option.mode)
    option.mode=1;

end
if ~isfield(option,'Scale')|| isempty(option.Scale)
    option.Scale=1;

end
if ~isfield(option,'Scalemode')|| isempty(option.Scalemode)
    option.Scalemode=1;
end
rand('state',option.seed);
randn('state',option.seed);
U_trainY=unique(trainY);
nclass=numel(U_trainY);
trainY_temp=zeros(numel(trainY),nclass);
% 0-1 coding for the target
for i=1:nclass
         idx= trainY==U_trainY(i);

         trainY_temp(idx,i)=1;
end
[Nsample,Nfea]=size(trainX);
N=option.N;
if strcmp(option.RandomType,'Uniform')
    if option.Scalemode==3
         Weight= option.Scale*(rand(Nfea,N)*2-1);
         Bias= option.Scale*rand(1,N);
      %   fprintf('linearly scale the range of uniform distribution to %d\n',  option.Scale);
    else
         Weight=rand(Nfea,N)*2-1;
          Bias=rand(1,N);
    end
else if strcmp(option.RandomType,'Gaussian')
          Weight=randn(Nfea,N);
          Bias=randn(1,N);
    else
        error('only Gaussian and Uniform are supported')
    end
end
Bias_train=repmat(Bias,Nsample,1);
H=trainX*Weight+Bias_train;

switch lower(option.ActivationFunction)
    case {'sig','sigmoid'}

        if option.Scale

            Saturating_threshold=[-4.6,4.6];
            Saturating_threshold_activate=[0,1];
            if option.Scalemode==1;

           [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale);
            elseif option.Scalemode==2

           [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale);
            end
        end
        H = 1 ./ (1 + exp(-H));
    case {'sin','sine'}


        if option.Scale

           Saturating_threshold=[-pi/2,pi/2];
           Saturating_threshold_activate=[-1,1];
           if option.Scalemode==1

            [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale);
          elseif option.Scalemode==2
            [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale);
           end
        end
        H = sin(H);
    case {'hardlim'}

        H = double(hardlim(H));
    case {'tribas'}

         if option.Scale

            Saturating_threshold=[-1,1];
            Saturating_threshold_activate=[0,1];
            if option.Scalemode==1

            [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale);
            elseif option.Scalemode==2

             [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale);
            end
        end
        H = tribas(H);
    case {'radbas'}

        if option.Scale

            Saturating_threshold=[-2.1,2.1];
            Saturating_threshold_activate=[0,1];
            if option.Scalemode==1

            [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale);
           elseif option.Scalemode==2

            [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale);
            end
        end
        H = radbas(H);

    case {'sign'}
        H = sign(H);
end
if option.bias
   H=[H,ones(Nsample,1)];
end
if option.link

        switch option.Scalemode
            case 1
            trainX_temp=trainX.*k+b;
            H=[H,trainX_temp];
            case 2
            [trainX_temp,ktr,btr]=Scale_feature_separately(trainX,Saturating_threshold_activate,option.Scale);
             H=[H,trainX_temp];
            otherwise
            H=[H,trainX];
        end

end
H(isnan(H))=0;
if option.mode==2

beta=pinv(H)*trainY_temp;
else if option.mode==1

    if ~isfield(option,'C')||isempty(option.C)
        option.C=0.1;
    end
    C=option.C;
    if N<Nsample
     beta=(eye(size(H,2))/C+H' * H) \ H'*trainY_temp;
    else
     beta=H'*((eye(size(H,1))/C+H* H') \ trainY_temp);
    end
    else
      error('Unsupport mode, only Regularized least square and Moore-Penrose pseudoinverse are allowed. ')
    end
end
trainY_temp=H*beta;
Y_temp=zeros(Nsample,1);
% decode the target
for i=1:Nsample
    [maxvalue,idx]=max(trainY_temp(i,:));
    Y_temp(i)=U_trainY(idx);
 end

Bias_test=repmat(Bias,numel(testY),1);
H_test=testX*Weight+Bias_test;
 switch lower(option.ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
         if option.Scale
             if option.Scalemode==1
            H_test=H_test.*k+b;
            elseif option.Scalemode==2
            nSamtest=size(H_test,1);
            kt=repmat(k,nSamtest,1);
            bt=repmat(b,nSamtest,1);
            H_test=H_test.*kt+bt;
             end
         end
        H_test = 1 ./ (1 + exp(-H_test));
    case {'sin','sine'}

       if option.Scale
           if option.Scalemode==1
            H_test=H_test.*k+b;
           elseif option.Scalemode==2
            nSamtest=size(H_test,1);
            kt=repmat(k,nSamtest,1);
            bt=repmat(b,nSamtest,1);
            H_test=H_test.*kt+bt;
           end
        end
        H_test = sin(H_test);
    case {'hardlim'}

        H_test = double(hardlim(H_test));
    case {'tribas'}

          if option.Scale
              if option.Scalemode==1
            H_test=H_test.*k+b;
             elseif option.Scalemode==2
            nSamtest=size(H_test,1);
            kt=repmat(k,nSamtest,1);
            bt=repmat(b,nSamtest,1);
            H_test=H_test.*kt+bt;
              end
         end
        H_test = tribas(H_test);
    case {'radbas'}


          if option.Scale
              if option.Scalemode==1
                H_test=H_test.*k+b;
              elseif option.Scalemode==2
              nSamtest=size(H_test,1);
              kt=repmat(k,nSamtest,1);
              bt=repmat(b,nSamtest,1);
              H_test=H_test.*kt+bt;
              end
         end
        H_test = radbas(H_test);
     case {'sign'}

        H_test = sign(H_test);

end

if option.bias
   H_test=[H_test,ones(numel(testY),1)];
end
if option.link
       switch option.Scalemode
            case 1
           testX_temp=testX.*k+b;
           H_test=[H_test,testX_temp];
            case 2
            nSamtest=size(H_test,1);
            kt=repmat(ktr,nSamtest,1);
            bt=repmat(btr,nSamtest,1);
            testX_temp=testX.*kt+bt;
            H_test=[H_test,testX_temp];
            otherwise
            H_test=[H_test,testX];
        end
end
H_test(isnan(H_test))=0;
testY_temp=H_test*beta;
Yt_temp=zeros(numel(testY),1);

for i=1:numel(testY)
    [maxvalue,idx]=max(testY_temp(i,:));
    Yt_temp(i)=U_trainY(idx);
end

eval_result.test = calc_metrics(testY, Yt_temp, FINAL);
eval_result.train = calc_metrics(trainY, Y_temp, FINAL);
eval_result.TrainOut = Y_temp;
eval_result.TestOut = Yt_temp;

end

function [eval_result] = calc_metrics(testY, Yt_temp, FINAL)
  if FINAL == 0
      eval_result.mse = round(mse(testY, Yt_temp), 4);
      eval_result.acc = length(find(Yt_temp==testY))/numel(testY);
  else

      [cmat, res, refRes] = confusion.getMatrix(testY, Yt_temp);
      eval_result.confusion = res;
      eval_result.cmat = cmat;
    % Errors = testY - Yt_temp;
    % ErrorMean = mean(Errors);
    % ErrorStd = std(Errors);
    % MSE = mean(Errors.^2);
    % RMSE = sqrt(MSE);

    % confusion_mat = confusionmat(testY,Yt_temp );
    % TP = confusion_mat(1);
    % FP = confusion_mat(2);
    % FN = confusion_mat(3);
    % TN = confusion_mat(4);
    % accuracy = (TP + TN) / (TP + FP + FN + TN);
    % eval_result.acc = length(find(Yt_temp==testY))/numel(testY);
    % eval_result.acc = round(accuracy, 4);
    % eval_result.confmat.tp = TP;
    % eval_result.confmat.fp = FP;
    % eval_result.confmat.fn = FN;
    % eval_result.confmat.tn = TN;
    Qo = testY;
    Qf = Yt_temp;
    Qom = mean(Qo);
    Qfm = mean(Qf);
    bias = Qo - Qf;
    sq_bias = bias .^ 2;
    abs_bias = abs(Qo - Qf);

    RMSE = (sum(sq_bias) / numel(Qo)) ^ .5;
    MAE = (sum(abs_bias)) / numel(Qo);
    % MSE = 1-((sum(sq_bias)) / (sum((Qo - Qom) .^ 2)));
    % mdl = fitlm(Qf, Qo);
    % R2 = mdl.Rsquared.Ordinary;

    eval_result.mse = round(mse(testY, Yt_temp), 4);
    % eval_result.mse = round(MSE, 4);

    % eval_result.rmse = sqrt(mean((testY - Yt_temp).^2));
    eval_result.rmse = round(RMSE, 4);

    % eval_result.mae = round(mae(testY - Yt_temp), 4);
    eval_result.mae = round(MAE, 4);

    eval_result.mre = round(sum((testY - Yt_temp)./testY)/length(testY), 4);
    eval_result.mapre = round(sum(abs((testY - Yt_temp)./testY))*100/length(testY), 4);

    % MSE = 1-((sum(sq_bias)) / (sum((Qo - Qom) .^ 2)));
    % mdl = fitlm(Qf, Qo);
    % R2 = mdl.Rsquared.Ordinary;
    % eval_result.r2 = round(R2, 4);
    eval_result.r2 = round(1 - sum((testY - Yt_temp).^2 ./ (testY - mean(testY)).^2), 4);
  end

end

function [Output,k,b]=Scale_feature(Input,Saturating_threshold,ratio)
Min_value=min(min(Input));
Max_value=max(max(Input));
min_value=Saturating_threshold(1)*ratio;
max_value=Saturating_threshold(2)*ratio;
k=(max_value-min_value)/(Max_value-Min_value);
b=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value);
Output=Input.*k+b;
end

function [Output,k,b]=Scale_feature_separately(Input,Saturating_threshold,ratio)
nNeurons=size(Input,2);
k=zeros(1,nNeurons);
b=zeros(1,nNeurons);
Output=zeros(size(Input));
min_value=Saturating_threshold(1)*ratio;
max_value=Saturating_threshold(2)*ratio;
for i=1:nNeurons
Min_value=min(Input(:,i));
Max_value=max(Input(:,i));
k(i)=(max_value-min_value)/(Max_value-Min_value);
b(i)=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value);
Output(:,i)=Input(:,i).*k(i)+b(i);
end

end