function partition = calculate_log_partition(rbm_w)
    hidden_state = rem(floor([0:2^size(rbm_w,1)-1]' * pow2(-(size(rbm_w,1)-1):0)),2)';
    partition = 1;
    size(hidden_state)
    size(rbm_w(:,1))
    size(hidden_state' * rbm_w(:,1))
    for idx = 1 : size(rbm_w, 2)
        partition = partition .* (1 + exp(hidden_state' * rbm_w(:,idx)));
    end
    partition = log(sum(partition(:)));
end