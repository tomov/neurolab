%% PCA on simulated neurons
%
% Guidance is provided via comments and example code below.
%
% To make this script easier to use, the task is broken into  sections,
% each section with a bold header (those starting with '%%') can be run by
% themselves (without running the entire script) using "ctrl + enter"
% (Windows) or "command + enter" (MAC). Just place your cursor within one of
% these sections (the section will become highlighted) to allow this functionality.
% 
% AVB & SLH 4/2016 editted by LND 2017

%% Close figures and clear workspace
clear;      % Delete all  variables in workspace, you will lose unsaved variables
close all;  % Close all of the open figure windows
%% Load data
% This line loads three variables: data, stim, time
load('pca_data.mat')

% data is a 58x5000 matrix, Neurons x Time Points
% Each row of data is the PSTH of a neuron's response to the stimuli
% stim is a 1x5000 column vector of the Stimuli over time
% time is a 1x5000 column vector of the Time in second
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% PART ONE %%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Plot data for five neurons
% This section of the code plots the stimulus and the responses for the
% first five neurons in the same figure.  You can copy and change this code
% to create your other figures.

% If you can't see the data in the figure, maximize the figure so you can. 

figure
ax(1) = subplot(6,1,1); % subplot allows you to plot multiple graphs in the same figure
plot(time,stim','r') % Plot the stimulus in red ('r')
ylabel('Odor concentration')
title('Stimulus')
for i = 1:5 % Loop over first 5 neurons to plot their responses
    ax(i+1) = subplot(6,1,i+1);
    plot(time,data(i,:)) % This plots row i of data
    title(['Response of neuron ',num2str(i)])
end
xlabel('Time (seconds)')
ylabel('Spike rate (Hz)')


% Now plot more example neurons

% Then try to plot the responses of all neurons together in a single plot

figure;
imagesc([stim * 3; data]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% PART TWO %%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Explore ways to compare how similar the responses are of neuronal pairs.
% Think about what would be good similarity metrics to compare neuronal
% responses. Try to implement these if you want.


%% Generate the covariance matrix
% Use the 'cov' function to calculate the covariance matrix.  The 'cov'
% function automatically centers the data so you do not need to do this. 
% This function will return a matrix that has variance along the diagonal
% entries and the covariance in the off-diagonal entries.

% Note that for this data we could either calculate a covariance matrix of
% how different neurons covary together or of how different time points
% covary together. (i.e. we calculate either how the rows or columns of the
% data matrix covary). Recall that our goal in the end is to use PCA to
% reduce the dimensionality of the data from 58 neurons to a smaller number
% of �principal component� neurons so think about which covariance matrix
% is most useful to us if we want to achieve this goal.

% Hint: Type 'doc cov' in the command window to look up how the 'cov'
% function works. 

% You have done this correctly if dataCov(1,1) = 1.0474

% Replace the [] with your own code below.

% =======================
% Insert/Modify code here

corrcoefs = nan(size(data, 1), 1);
for i = 1:size(data,1)
    c = corrcoef(stim, data(i,:));
    corrcoefs(i) = c(1,2);
end
datac = [corrcoefs, data];
datac = sortrows(datac);

figure;
imagesc(datac);

% =======================

%% Plot the covariance matrix
% You can use the 'imagesc' function to visualise the covariance matrix.
% Calling the 'colorbar' function adds the color scale on your graph.
% 

% =======================
% Insert/Modify code here

figure;
imagesc(cov(datac'));


% =======================


%% Cluster the covariance matrix
% use the kmeans function with the covariance matrix as input. Think about
% how many cluster (k) to input.

% The output will be a list of cluster IDs. Find all the neurons that
% belong to cluster 1, cluster 2, and so on. Plot the responses of all the
% neurons in cluster 1 and compare them to the stimulus vs. time plot. Use
% similar plotting approaches to those used in Part 1.

% Insert code here

idx = kmeans(data, 3);

figure;
datak = [idx, data];
datak = sortrows(datak);
imagesc(datak);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% PART THREE %%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Perform PCA
% Use the 'pca' function to run PCA on the data matrix. Your goal is to
% analyze the relationships among neurons, not the relationships among
% time points. Your hypothesis is that the responses of all 58 neurons
% can be reduced to linear combinations of a few orthogonal basis
% functions, where each basis function can be conceptualized as a
% distinct "response type". You should set up the PCA so that you obtain
% 58 PCs; you hypothesize that only a few of these PCs are needed to
% explain most of the variance in the data.

% The 'pca' function returns several values. For us the important ones are:
% 
%   'score' - This is a matrix containing the representation of the data
%       set in PC space. Essentially, this is the data set after it has
%       been rotated. Recall that the original data set consisted of 58
%       vectors, with each vector representing a neural response measured
%       at 5000 time points; this new matrix therefore also consists of 58
%       vectors measured at 5000 time points. Note that there is some
%       confusion of terminology in the literature.  Sometimes the PC
%       scores are referred to as the "Principal components (PCs)" while
%       other times our new axes (the eigenvectors) are called the
%       "Principal components". We prefer the second usage and hence we
%       will refer to the scores as "PC scores" and to the axes as PCs but
%       you should be familiar with both.
%   'explained' - This is a list of numbers quantifying the percentage of
%       the variance in the data explained by each of the PCs, in 
%       descending order of variance explained.
%   'coeff' - This is a matrix that quantifies the importance of each
%       variable (here, each neuron in the original dataset) in accounting 
%       for the variability of the associated PC. (This matrix gets the 
%       name 'coeff' because it contains the correlation coefficients 
%       between the data in PC space and the original data.) These values 
%       are also known as loadings. Each column of coeff contains 
%       coefficients for one principal component, and the columns are in 
%       descending order of component variance.

% Read the help documentation on pca for further information. 

% If you're confused, we recommend first performing PCA and the plotting
% steps below and then coming back to revise your code based on your
% results. Plotting the outputs from the pca function can be helpful for
% understanding what pca is doing.

% You have done this correctly if coeff(1,1) = 0.0724

% =======================
% Insert/Modify code here

% =======================

% each neuron is a variable, each time point is an observation

% eigenvectors = how much does each of the original neurons contribute to
% the given eigen"neuron"
% score matrix = how much weight you're putting on each eigenneuron at each
% time point
%
% each vector in the space is a set of firing rates for all the neurons = (r1, r2, r3, .... r58)
% each column = a time point = another vector in the space
% so we have 5000 vectors of 58 firing rates each
%
% each component in old vector space = firing of a given neuron
% each PC (in new space) = linear sum of firing of the given neurons = an "eigenneuron"
%                        = (1, 0, 0, ... 0) * w1 + (0, 1, 0 ... 0) * w2 + 
%                          ... + (0, 0, ..., 1) * w58
%      they're all orthogonal (almost)
%
% score matrix: each column = weights for how much each principle component
% contributes to the given vector of firing rates (the given time point)
%
% coefficients = each neuron (component/axis in the old space) can be expressed
% as a linear combination of the eigenvectors/new axes = the "eigenneurons"
%
% multiply coeff by score to get the original data in terms of the PCs --
% can only use e.g. top 3 PCs
% e.g. coeff(58,1) * score(t,1) = how neuron 58 maps to PC 1 at time t
%      coeff(58,2) * score(t,2) = how neuron 58 maps to PC 2 at time t
%      coeff(58,3) * score(t,3) = how neuron 58 maps to PC 3 at time t
%
% so score(t,1) = the projected data; the magnitude of the vector at time t
% onto PC 1
%    score(t,2) = the projection of vector at time t on PC 2
%
% btw -- # of clusters and # of PCs are generally unrelated!!!
% can have many clusters with few PCs, or just 1 cloud and many PCs
% separate ideas!
%

% to reconstruct the data matrix, multiply 

[coeff,score,latent,tsquared,explained,mu] = pca(data');

figure;
imagesc(score(:,1:3)') ;

figure;
imagesc(datak);

%% Plot explained variance (~Scree plot)
% Use the output from the 'pca' function above to make a plot of the
% different PC contributions to explained variance in the data.
%
%   hint: To make it easy to see the variance explained by each pc when you
%       plot 'explained' also pass '-o' to the plot function, like this
%       example: plot(explained,'-o')

% =======================
% Insert/Modify code here

figure;
plot(explained,'-o');

% =======================



%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% PART FOUR %%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Use the output of 'pca' to plot the first six principal component scores
% The first six PC scores are the first six vectors in the score matrix,
% where each vector is a list of 5000 numbers.

% =======================
% Insert/Modify code here

% =======================

%%
figure;
imagesc(coeff(1:58,1:3) * score(:,1:3)', [0 7])

figure;
imagesc(data, [0 7]);



%% Find the covariance matrix of data in the PC space and plot it
% Use the 'imagesc' function and the 'colorbar' function for plotting. You
% should have 58 PCs, so this should be a 58-by-58 matrix (i.e. the same
% size as the previous covariance matrix you plotted).
% 
% You have done this correctly if the first entry in the matrix = 47.1669

% =======================
% Insert/Modify code here

% =======================

figure;
imagesc(cov(score));
colorbar;

% notice all of them are (almost) 0 => the PC's are orthogonal


%% Make a 3D plot of each neuron's loadings for the first 3 PCs
% Use 'plot3' to make a 3D plot. Plot each loading as a discrete dot or
% circle for clarity, and please label the axes. Remember the loadings
% correspond to the 'coeff' output of the 'pca' function.
%
%   hint: type 'doc plot3' for info about how to label the axes 
%   hint: pass 'o' to the plot3 function, to plot each loading as a 
%       discrete dot, like this example: 
%       plot3(x_variable, y_variable, z_variable,'o') 
%   hint: using 'grid on' might make your graph
%       more easily viewable

% =======================
% Insert/Modify code here

% =======================

figure;
% plot each neuron as a sum of the top 3 PCs = a vector in 3D space (each
% component = how much the given PC (eigenneuron) contributes to the real
% neuron)
plot3(coeff(:,1), coeff(:,2), coeff(:, 3), 'o'); grid on;


%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% PART FIVE %%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Perform PCA to analyze the relationship between time points
% Now instead of analyzing the relationship between neurons, we are 
% interested in how the population as a whole evolves over time. Your 
% goal is to visualize that population trajectory.

% Make sure you still have the output of your PCA in the workspace or redo
% the PCA


% This code uses a for loop to show the temporal evolution of the stimulus
% in a specified step size. All previous time points are shown in black and
% the most recent past time points are shown in red. One subplot is used
% for the stimulus. Fill in the code to make in the second subplot a 3D
% plot of the first 3 PC scores. At the end of each iteration of the for
% loop, a pause is included to allow you inspect the figure. A key press is
% required to start the next iteration. Play with the step size if you want
% faster or slower steps through the temporal evolution of stimulus and
% neural activity.

stepSize = 50;
figure; hold on;
for i = stepSize+1:stepSize:length(stim)
    ax1 = subplot(2,1,1);
    cla;
    plot(time(1:i),stim(1:i),'k-');
    hold on;
    plot(time(i-stepSize:i),stim(i-stepSize:i),'r-','LineWidth',3);
    axis(ax1,[0 time(end) 0 1])
    ax2 = subplot(2,1,2);
    cla;
    % Insert code to make a 3D plot of the first 3 PCs, like is done in the
    % same loop for stimulus
    %
    % plot individual neurons
    %
    %plot3(coeff(:,1) * score(i,1), coeff(:,2) * score(i,2), coeff(:,3) * score(i,3), 'o'); grid on;
    %
    % plot principle components
    %
    plot3(score(1:i,1), score(1:i,2), score(1:i,3));
    hold on;
    plot3(score(i,1), score(i,2), score(i,3), 'o', 'Color', 'red');
    grid on;
    axis(ax2,[-10 20 -10 10 -10 10]);
    pause;
end



%% Extension problems
% If you feel confident or would like to gain additional practice, please
% continue by answering the extension problems as outlined in the
% instructions. Some problems will require more coding, you may complete
% this below.

