function [error_rate] = nn_classifier(training, testing)
    training_data = training(:, 1:size(training,2)-1);
    training_labels = training(:, size(training,2));
    testing_data = training(:, 1:size(testing,2)-1);
    testing_labels = training(:, size(testing,2));
    errors = 0;
    for index = 1:size(testing_data,1)
        closest_distance = realmax();
        closest_label = -1;
        for training_index = 1:size(training_data,1)
            delta = training_data(training_index,:)-...
                    testing_data(index,:);
            distance = delta*transpose(delta);
            if distance < closest_distance
                closest_distance = distance;
                closest_label = training_labels(training_index,1);
            end
        end
        if (closest_label ~= testing_labels(index,:))
            errors = errors + 1;
        end
    end
    error_rate = errors / size(testing_data,1);    
end
