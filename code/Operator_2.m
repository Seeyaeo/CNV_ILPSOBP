function [current_fitness_Q, pop] = Operator_2(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% ���ѡ������Ԫ�ؽ��н��棬�������Ԫ�ز��뵽ǰ���Ԫ��ǰ�棬�����½�

    % ��ȡ��ĳ���
    parent = elite_archive(idx_parent, :);
    n = length(parent);

    % ���ѡ��������ͬ������
    index1 = randi(n);
    index2 = randi(n);
    
    % ȷ������Ԫ��������ͬ
    while index1 == index2
        index2 = randi(n);
    end
    
    % �����½Ⲣ�������Ԫ�ز��뵽ǰ���Ԫ��ǰ��
    new_solution = parent;
    if index1 < index2   
        temp = new_solution(index2);
        new_solution(index2) = [];  %ɾ�������Ԫ�أ�������Ϊ��
        new_solution = [new_solution(1:index1-1), temp, new_solution(index1:end)];
    else
        temp = new_solution(index1);
        new_solution(index1) = [];  %ɾ�������Ԫ�أ�������Ϊ��
        new_solution = [new_solution(1:index2-1), temp, new_solution(index2:end)];
    end
    
    %������Ӧ��ֵ
    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;

    %����pop
    pop(idx_parent, :) = new_solution;
    
end

