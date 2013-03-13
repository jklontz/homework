function [error_rate] = bayesian_classifier(training, testing)
    training_data = training(:, 1:size(training,2)-1);
    training_labels = training(:, size(training,2));
    testing_data = training(:, 1:size(testing,2)-1);
    testing_labels = training(:, size(testing,2));
    unique_labels = unique(training_labels);
    class_means = {};
    class_covariances = {};
    for label = 1:size(unique_labels,1)
        class_data = training_data(training_labels==label,:);
        class_means{label} = mean(class_data);
        class_covariances{label} = cov(class_data);
    end
    errors = 0;
    for index = 1:size(testing_data,1)
        highest_pdf = 0;
        highest_label = -1;
        for label = 1:size(unique_labels,1)
            pdf = mvnpdf(testing_data(index,:),...
                         class_means{label},...
                         class_covariances{label});
            if pdf > highest_pdf
                highest_pdf = pdf;
                highest_label = label;
            end
        end
        if (highest_label ~= testing_labels(index,:))
            errors = errors + 1;
        end
    end
    error_rate = errors / size(testing_data,1);    
end

