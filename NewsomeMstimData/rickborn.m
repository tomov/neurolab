clear all;

files = {'es2aRaw.xlsx', 'es5bRaw.xlsx', 'js21aRaw.xlsx', 'js25aRaw.xlsx', 'js92dRaw.xlsx'};

disp('------------------------------------------------------');

%loglikelihood = sum(log(poisspdf(spikeTrain,lambda1)));     %log likelihood
%AIC1 = -2 * loglikelihood1 + 2 * 2;

for file = files
    data = xlsread(file{1});

    y = data(:,3);
    Mstim = data(:, 1);
    Coh = data(:, 2);
    X = [Mstim Coh];
    %X = [Mstim Coh];

    fprintf('\n--- %s ----\n', file{1});

    [b,dev,stats] = glmfit(X,y,'bi;nomial','logit');
    [prob, CI1, CI2] = glmval(b, X, 'logit', stats);
    choice = prob > 0.5;
    accuracy = sum(choice == y) / size(y, 1);
    fprintf('accuracy = %.2f%%\n', accuracy * 100);

    
    plot_Coh = [-20:0.01:20]';

    figure;
    hold on;
    Mstim = logical(Mstim);
    y = logical(y);
    
    % stimulated
    %
    X = [ones(size(plot_Coh)) plot_Coh];
    preferred_PD = glmval(b, X, 'logit', stats);
    plot(plot_Coh, preferred_PD, 'LineWidth', 2, 'Color', 'red');
    
    coh_bins = -20:2:20;
    probs = hist(Coh(Mstim & y), coh_bins) ./ hist(Coh(Mstim), coh_bins);
    scatter(coh_bins, probs, [], 'red');
    
    % not stimulated
    %
    X = [zeros(size(plot_Coh)) plot_Coh];
    preferred_PD = glmval(b, X, 'logit', stats);
    plot(plot_Coh, preferred_PD, 'LineWidth', 2, 'Color', 'blue');
    
    coh_bins = -20:2:20;
    probs = hist(Coh(~Mstim & y), coh_bins) ./ hist(Coh(~Mstim), coh_bins);
    scatter(coh_bins, probs, [], 'blue');
    
    hold off;
    title(file{1});
    xlabel('Coherence');
    ylabel('Probability PD');
    
  %  break;
end