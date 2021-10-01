function print_result(eval_result)
    fprintf('\n >>> For training dataset >>> \n');
%
%     TP=eval_result.train.confmat.tp;
%     FP=eval_result.train.confmat.fp;
%     TN=eval_result.train.confmat.tn;
%     FN=eval_result.train.confmat.fn;
%     disp([' - TP: ', num2str(TP), ...
%         ', FP: ', num2str(FP), ...
%         ', TN: ', num2str(TN), ...
%         ', FN: ', num2str(FN)])
%     sens=(TP/(TP+FN)*100);
%     spes=(TN/(FP+TN)*100);
%     ppv=(TP/(TP+FP)*100);
%     npv=(TN/(FN+TN)*100);
%     fpr=FP/(TN+FP);
%     recall=TP/(TP+FN);
%     prec=TP/(TP+FP);
%     f1=2*(prec*recall)/(prec+recall);
    res = eval_result.train.confusion;
%     disp('- Confusion Matrix')
%     disp(reshape_cmat(eval_result.train.cmat))
    fprintf('\n - Accuracy:  %.4f', res.Accuracy);
    fprintf('\n - Sensitivity:  %.4f', res.Sensitivity);
    fprintf('\n - Specitifity:  %.4f', res.Specificity);
%     fprintf('\n - Precision: %.4f', res.Precision);
    fprintf('\n - False Positive Rate :  %.4f', res.FalsePositiveRate);
    fprintf('\n - F1-score :  %.4f \n' , res.F1_score);
%     fprintf('\n - MRE:  %.4f', eval_result.train.mre);
%     fprintf('\n - RMSE:  %.4f', eval_result.train.rmse);
%     fprintf('\n - MAPRE:  %.4f', eval_result.train.mapre);
%     fprintf('\n - MAE:  %.4f', eval_result.train.mae);
%     fprintf('\n - R2:  %.4f \n', eval_result.train.r2);

    fprintf('\n >>> For test dataset >>> \n');
%      TPt=eval_result.test.confmat.tp;
%     FPt=eval_result.test.confmat.fp;
%     TNt=eval_result.test.confmat.tn;
%     FNt=eval_result.test.confmat.fn;
%     disp([' - TP: ', num2str(TPt), ...
%         ', FP: ', num2str(FPt), ...
%         ', TN: ', num2str(TNt), ...
%         ', FN: ', num2str(FNt)])
%     senst=(TPt/(TPt+FNt)*100);
%     spest=(TNt/(FPt+TNt)*100);
%     ppvt=(TPt/(TPt+FPt)*100);
%     npvt=(TNt/(FNt+TNt)*100);
%         fprt=FPt/(TNt+FPt);
%     recallt=TPt/(TPt+FNt);
%     prect=TPt/(TPt+FPt);
%     f1t=2*(prect*recallt)/(prect+recallt);
%     disp([' - TP: ', num2str(TPt), ...
%             ', FP: ', num2str(FPt), ...
%             ', TN: ', num2str(TNt), ...
%             ', FN: ', num2str(FNt)])
    res = eval_result.test.confusion;
%     disp(' - Confusion Matrix')
%     disp(reshape_cmat(eval_result.test.cmat))
    fprintf('\n - Accuracy:  %.4f', res.Accuracy);
    fprintf('\n - Sensitivity:  %.4f', res.Sensitivity);
    fprintf('\n - Specitifity:  %.4f', res.Specificity);
%     fprintf('\n - Precision: %.4f', res.Precision);
    fprintf('\n - False Positive Rate :  %.4f', res.FalsePositiveRate);
    fprintf('\n - F1-score :  %.4f \n', res.F1_score);

%     fprintf('\n - MRE:  %.4f', eval_result.test.mre);
%     fprintf('\n - RMSE:  %.4f', eval_result.test.rmse);
%     fprintf('\n - MAPRE:  %.4f', eval_result.test.mapre);
%     fprintf('\n - MAE:  %.4f', eval_result.test.mae);
%     fprintf('\n - R2:  %.4f \n', eval_result.test.r2);
end

function [cmat] = reshape_cmat(mat)
    a = sum(mat);
    b = [mat; a];
    c = sum(b');
    d = [b, c'];
    cmat = array2table(d, ...
         'VariableNames', {'0', '1', '2', '3', '4', 'Total'}, ...
         'RowNames', {'0', '1', '2', '3', '4', 'Total'});
end