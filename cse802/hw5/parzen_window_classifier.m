function [error_rate] = parzen_window_classifier(training, testing)
    h = 10; % Change window size here
    training_data = training(:, 1:size(training,2)-1);
    training_labels = training(:, size(training,2));
    testing_data = training(:, 1:size(testing,2)-1);
    testing_labels = training(:, size(testing,2));
    unique_labels = unique(training_labels);
    errors = 0;
    for index = 1:size(testing_data,1)
        densities = [0 0 0];
        for training_index = 1:size(training_data,1)
            training_label = training_labels(training_index,1);
            delta = training_data(training_index,:)-...
                    testing_data(index,:);
            densities(training_label) = densities(training_label) + ...
                                        normpdf(delta*transpose(delta)/h);
        end
        highest_weight = 0;
        highest_label = -1;
        for label = 1:size(unique_labels,1)
            if densities(label) > highest_weight
                highest_weight = densities(label);
                highest_label = label;
            end
        end
        if (highest_label ~= testing_labels(index,:))
            errors = errors + 1;
        end
    end
    error_rate = errors / size(testing_data,1);    
end
