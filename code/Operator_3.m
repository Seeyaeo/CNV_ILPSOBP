function [current_fitness_Q, pop] = Operator_3(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% ���ѡ������Ԫ�أ���ת���м��Ԫ�أ������½�

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
    
    % �����½Ⲣ��ת��������λ��֮���Ԫ��
    new_solution = parent;
    new_solution(index1:index2) = new_solution(index2:-1:index1);
    
    %������Ӧ��ֵ
    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;
    
    %����pop
    pop(idx_parent, :) = new_solution;
end

