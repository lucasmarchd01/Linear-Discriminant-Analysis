function a4_20144315
% Function for CISC271, Winter 2021, Assignment #4

    % Read the test data from a CSV file
    dmrisk = csvread('dmrisk.csv',1,0);

    % Columns for the data and labels; DM is diabetes, OB is obesity
    jDM = 17;
    jOB = 16;

    % Extract the data matrices and labels
    XDM = dmrisk(:, (1:size(dmrisk,2))~=jDM);
    yDM = dmrisk(:,jDM);
    XOB = dmrisk(:, (1:size(dmrisk,2))~=jOB);
    yOB = dmrisk(:,jOB);

    % Reduce the dimensionality to 2D using PCA
    [~,rDM] = pca(zscore(XDM), 'NumComponents', 2);
    [~,rOB] = pca(zscore(XOB), 'NumComponents', 2);

    % Find the LDA vectors and scores for each data set
    [qDM zDM qOB zOB] = a4q1(rDM, yDM, rOB, yOB);


    % Plot 2D PCA for diabetes data
    f1 = figure;
    gscatter(rDM(:, 1), rDM(:, 2), yDM, 'cm');
    title('PCA for Diabetes data')
    
    % Plot 2D PCA for obesity data
    f2 = figure;
    gscatter(rOB(:, 1), rOB(:, 2), yOB, 'cm');
    title('PCA for Obesity data')
    
    % Plot LDA scores for diabetes data
    f3 = figure;
    gscatter(zDM, zDM, yDM, 'kg');
    title('LDA scores for Diabetes')
    xlabel('zDM')
    ylabel('zDM')
    
    % Plot LDA scores for obesity data
    f4 = figure;
    gscatter(zOB, zOB, yOB, 'kg');
    title('LDA scores for Obesity')
    xlabel('zOB')
    ylabel('zOB')

    
    % Find roc curve and display important data
    [xrocDM, yrocDM, aucDM, boptDM] = roccurve(yDM, zDM);
    [xrocOB, yrocOB, aucOB, boptOB] = roccurve(yOB, zOB);
    disp('Best Threshold value for DM:')
    disp(boptDM)
    disp('Best Threshold value for OB:')
    disp(boptOB)
    disp('AUC for DM:')
    disp(aucDM)
    disp('AUC for OB:')
    disp(aucOB)
    
    % plot ROC for diabetes
    fig5 = figure;
    plot(xrocDM, yrocDM,'.b')
    title('ROC for Diabetes data')
    xlabel('FPR: specificity')
    ylabel('TPR: sensitivity')
    
    % plot ROC for obesity
    fig6 = figure;
    plot(xrocOB, yrocOB,'.b')
    title('ROC for Obesity data')
    xlabel('FPR: specificity')
    ylabel('TPR: sensitivity')
    

% END OF FUNCTION
end

function [q1, z1, q2, z2] = a4q1(Xmat1, yvec1, Xmat2, yvec2)
% [Q1 Z1 Q2 Z2]=A4Q1(X1,Y1,X2,Y2) computes an LDA axis and a
% score vector for X1 with Y1, and for X2 with Y2.
%
% INPUTS:
%         X1 - MxN data, M observations of N variables
%         Y1 - Mx1 labels, +/- computed as ==/~= 1
%         X2 - MxN data, M observations of N variables
%         Y2 - Mx1 labels, +/- computed as ==/~= 1
% OUTPUTS:
%         Q1 - Nx1 vector, LDA axis of data set #1
%         Z1 - Mx1 vector, scores of data set #1
%         Q2 - Nx1 vector, LDA axis of data set #2
%         Z2 - Mx1 vector, scores of data set #2

    q1 = [];
    z1 = [];
    q2 = [];
    z2 = [];
    
    % Compute the LDA axis for each data set
    q1 = lda2class(Xmat1(yvec1==1,:), Xmat1(yvec1~=1, :));
    q2 = lda2class(Xmat2(yvec2==1,:), Xmat2(yvec2~=1, :));
   
    % Compute LDA scores by projection
    z1 = Xmat1 * q1;
    z2 = Xmat2 * q2;
    
% END OF FUNCTION
end

function qvec = lda2class(X1, X2)
% QVEC=LDA2(X1,X2) finds Fisher's linear discriminant axis QVEC
% for data in X1 and X2.  The data are assumed to be sufficiently
% independent that the within-label scatter matrix is full rank.
%
% INPUTS:
%         X1   - M1xN data with M1 observations of N variables
%         X2   - M2xN data with M2 observations of N variables
% OUTPUTS:
%         qvec - Nx1 unit direction of maximum separation

    qvec = ones(size(X1,2), 1);
    xbar1 = mean(X1);
    xbar2 = mean(X2);

    % Compute the within-class means and scatter matrices
    S1 = (X1 - ones(length(X1), 1) * xbar1)'*(X1 - ones(length(X1), 1) * xbar1);
    S2 = (X2 - ones(length(X2), 1) * xbar2)'*(X2 - ones(length(X2), 1) * xbar2);
    Sw = S1 + S2;
    
    % Compute the between-class scatter matrix
    X = [X1; X2];
    Xbar = xbar1 + xbar2;
    Sb = [xbar1 - Xbar; xbar2 - Xbar]'*[xbar1 - Xbar; xbar2 - Xbar];
    
    % Fisher's linear discriminant is the largest eigenvector
    % of the Rayleigh quotient
    [qvec, ~] = eigs(Sw\Sb, 1);

    % May need to correct the sign of qvec to point towards mean of X1
    if (xbar1 - xbar2)*qvec < 0
        qvec = -qvec;
    end
% END OF FUNCTION
end

function [fpr tpr auc bopt] = roccurve(yvec_in,zvec_in)
% [FPR TPR AUC BOPT]=ROCCURVE(YVEC,ZVEC) computes the
% ROC curve and related values for labels YVEC and scores ZVEC.
% Unique scores are used as thresholds for binary classification.
%
% INPUTS:
%         YVEC - Mx1 labels, +/- computed as ==/~= 1
%         ZVEC - Mx1 scores, real numbers
% OUTPUTS:
%         FPR  - Kx1 vector of False Positive Rate values
%         TPR  - Kx1 vector of  True Positive Rate values
%         AUC  - scalar, Area Under Curve of ROC determined by TPR and FPR
%         BOPT - scalar, optimal threshold for accuracy

    % Sort the scores and permute the labels accordingly
    [zvec zndx] = sort(zvec_in);
    yvec = yvec_in(zndx);
        
    % Sort and find a unique subset of the scores; problem size
    bvec = unique(zvec);
    bm = numel(bvec);
    
    % Compute a confusion matrix for each unique threshold value;
    % extract normalized entries into TPR and FPR vectors; track
    % the accuracy and optimal B threshold
    tpr = [];
    fpr = [];
    acc = -inf;
    bopt = -inf;
    for jx = 1:bm
        
        % Find tpr, fpr, and optimal threshold value.
        cmat = confmat(yvec, zvec, bvec(jx));
        tp = cmat(1, 1);
        fn = cmat(1, 2);
        fp = cmat(2, 1);
        tn = cmat(2, 2);
        tpr = [tpr tp/(tp + fn)];
        fpr = [fpr fp/(fp + tn)];
        if (tp + tn)/sum([tp fn fp tn]) > acc
            acc = (tp + tn)/sum([tp fn fp tn]);
            bopt = bvec(jx);
        end
    end
    
    % Show optimal threshold confusion matrix.
    disp('Confusion matrix for optimal threshold:');
    disp(bopt);
    disp(confmat(yvec, zvec, bopt));
    
    
    % Ensure that the rates, from these scores, will plot correctly
    tpr = sort(tpr);
    fpr = sort(fpr);
    
    % Compute AUC for this ROC
    auc = aucofroc(tpr, fpr);
end
    
function cmat = confmat(yvec, zvec, theta)
% CMAT=CONFMAT(YVEC,ZVEC,THETA) finds the confusion matrix CMAT for labels
% YVEC from scores ZVEC and a threshold THETA. YVEC is assumed to be +1/-1
% and each entry of ZVEC is scored as -1 if <THETA and +1 otherwise. CMAT
% is returned as [TP FN ; FP TN]
%
% INPUTS:
%         YVEC  - Mx1 values, +/- computed as ==/~= 1
%         ZVEC  - Mx1 scores, real numbers
%         THETA - threshold real-valued scalar
% OUTPUTS:
%         CMAT  - 2x2 confusion matrix; rows are +/- labels,
%                 columns are +/- classifications

    % Find the plus/minus 1 vector of quantizations
    qvec = sign((zvec >= theta) - 0.5);
    
    % Compute the confusion matrix by entries
    TP=0;FP=0;TN=0;FN=0;
    for idx = 1: length(yvec)
        if yvec(idx) == 1 && qvec(idx) == 1
            TP = TP + 1;
        elseif yvec(idx) == 1 && qvec(idx) == -1
            FN = FN + 1;
        elseif yvec(idx) == -1 && qvec(idx) == -1
            TN = TN + 1;
        else
            FP = FP + 1;
        end
    end
    
    % Set up confusion matrix
    cmat = [TP FN ; FP TN];
    
end

function auc = aucofroc(tpr, fpr)
% AUC=AUCOFROC(TPR,FPR) finds the Area Under Curve of the
% ROC curve specified by the TPR, True Positive Rate, and
% the FPR, False Positive Rate.
%
% INPUTS:
%         TPR - Kx1 vector, rate for underlying score threshold 
%         FPR - Kx1 vector, rate for underlying score threshold 
% OUTPUTS:
%         AUC - integral, from Trapezoidal Rule on [0,0] to [1,1]

    [X undx] = sort(reshape(tpr, 1, numel(tpr)));
    Y = sort(reshape(fpr(undx), 1, numel(undx)));
    auc = abs(trapz([0 Y 1] , [0 X 1]));
end
