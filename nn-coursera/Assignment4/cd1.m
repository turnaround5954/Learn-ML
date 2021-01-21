function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data);
    %---positive phase
    hidden_probs_on_data = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hidden_state_on_data = sample_bernoulli(hidden_probs_on_data);
    gradient_data = configuration_goodness_gradient(visible_data, hidden_state_on_data);
    %---negative phase
    reconstruct_visible_probs = hidden_state_to_visible_probabilities(rbm_w, hidden_state_on_data);
    reconstruct_visible_state = sample_bernoulli(reconstruct_visible_probs);
    reconstruct_hidden_probs = visible_state_to_hidden_probabilities(rbm_w, reconstruct_visible_state);
    %reconstruct_hidden_state = sample_bernoulli(reconstruct_hidden_probs);
    %gradient_model = configuration_goodness_gradient(reconstruct_visible_state, reconstruct_hidden_state);
    gradient_model = configuration_goodness_gradient(reconstruct_visible_state, reconstruct_hidden_probs);
    %---calculate CD1
    ret = gradient_data - gradient_model;

end
