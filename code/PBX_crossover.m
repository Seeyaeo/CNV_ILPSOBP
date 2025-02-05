function [current_fitness, pop, elite_archive] = PBX_crossover(elite_archive, pop, idx_parent1,idx_parent2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% PBX-�����������(����λ�õĽ���)
% ���룺parent1, parent2  ������������Ⱦɫ��
% �����child1,child2     ���������Ӵ�Ⱦɫ��

parent1 = pop(idx_parent1, :);
parent2 = pop(idx_parent2, :);

len = length(parent1);
child1 = NaN(1, len);
child2 = NaN(1, len);

% ���ѡ��λ�ü���
positions = randperm(len, randi([1, len-1]));

% ����ѡ��λ�õĻ����Ӵ�
child1(positions) = parent1(positions);
child2(positions) = parent2(positions);

% ����Ӵ������ಿ��
child1 = fill_remaining_positions(child1, parent2);
child2 = fill_remaining_positions(child2, parent1);

function child = fill_remaining_positions(child, parent)
    len = length(parent);
    parent_pos = 1;
    for i = 1:len
        if isnan(child(i))
            % Ѱ�Ҹ����в����Ӵ��еĻ���
            while ismember(parent(parent_pos), child)
                parent_pos = parent_pos + 1;
                if parent_pos > len
                    error('All parent genes are already in the offspring');
                end
            end
            child(i) = parent(parent_pos);
            parent_pos = parent_pos + 1;
        end
    end
end

fitness_parent1 = fun(parent1,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_parent2 = fun(parent2,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_child1 = fun(child1,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_child2 = fun(child2,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);

% ѡ����ŵ��Ӵ�����ϲ�ĸ���
if fitness_child1 < fitness_child2
    if fitness_parent1 < fitness_parent2
        pop(idx_parent2, :) = child1;
    else
        pop(idx_parent1, :) = child2;
    end
    current_fitness = fitness_child1; %������Ӧ��ֵ��С�Ӵ�����Ӧ��ֵ
    elite_archive(end + 1, :) = child1;  %����Ӧ��ֵ��С���Ӵ�����浵����
else
    if fitness_parent1 < fitness_parent2
        pop(idx_parent2, :) = child2;
    else
        pop(idx_parent1, :) = child1;
    end
    current_fitness = fitness_child2; %������Ӧ��ֵ��С�Ӵ�����Ӧ��ֵ
    elite_archive(end + 1, :) = child2;  %����Ӧ��ֵ��С���Ӵ�����浵����
end

end

