%% 该代码为基于BP神经网络的预测算法
%% 清空环境变量
clc
clear
%% 训练数据预测数据提取及归一化
%节点个数
inputnum = 5;
hiddennum = 4;
outputnum = 4;

data1 = load('data\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2 = load('data\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data3 = load('data\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data4 = load('data\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data5 = load('data\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data6 = load('data\0.4_6x_mat\sim1_6_6100_read_trains.mat');

data_trains1=load('data\0.2_4x_11.1\sim1_4_4100_trains.txt');
data_trains2=load('data\0.3_4x_11.1\sim1_4_4100_trains.txt');
data_trains3=load('data\0.4_4x_11.1\sim1_4_4100_trains.txt');
data_trains4=load('data\0.2_6x_11.1\sim1_6_6100_trains.txt');
data_trains5=load('data\0.3_6x_11.1\sim1_6_6100_trains.txt');
data_trains6=load('data\0.4_6x_11.1\sim1_6_6100_trains.txt');

data_trains = [data_trains1;data_trains2;data_trains3;data_trains5;data_trains6;];
column = [2,3,4,5,6];
[m1,n1] = size(data_trains);

trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);

%从1到trainlines间随机排序���������
k = rand(1,trainLines);
[m,n] = sort(k);
%得到输入输出数据��������
ginput = gdata(:,column);
goutput1 = gdata(:,7);
%输出从一维变成四维：0正常，1gain，2hemi_loss，3homo_loss;
goutput = zeros(trainLines,4);
for i = 1:trainLines
    switch goutput1(i)
        case 0
            goutput(i,:) = [1 0 0 0];
        case 1
            goutput(i,:) = [0 1 0 0];
        case 2
            goutput(i,:) = [0 0 1 0];
        case 3
            goutput(i,:) = [0 0 0 1];
    end
end

%找出训练数据和预测数据
ginput_train = ginput(n(1:trainLines),:)';
goutput_train = goutput(n(1:trainLines),:)';

%样本输入输出数据归一化
[ginputn,ginputps] = mapminmax(ginput_train);
[outputn,outputps] = mapminmax(goutput_train);

%% BP网络训练
% %初始化网络结构��ṹ
net = newff(ginputn,goutput_train,hiddennum);

%节点总数
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

% 参数初始化
%粒子群算法中的两个参数�������
c1 = 1.49445;
c2 = 1.49445;
lr = 0.5;
history = [];
init_fitness = [];%初始归档中的较好适应度值�ֵ
init_index = [];  %初始归档中的位置
maxgen=2;      %进化次数   
sizepop=2;     %种群规模

Vmax = 1;
Vmin = -1;
popmax = 5;
popmin = -5;

%初始化（大种群）
for i = 1:sizepop
    pop(i,:) = 5 * rands(1,numsum);
    V(i,:) = rands(1,numsum);
    fitness(i) = fun(pop(i,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
end

% 个体极值和群体极值
[bestfitness, bestindex] = min(fitness);
%将极端最优值放入归档集
while bestfitness <= 100
    init_fitness = [init_fitness bestfitness];
    init_index = [init_index bestindex];
    history = [history bestfitness]; %记录所有存在的最优适应度值
    pop(bestindex,:) = 5*rands(1,numsum);
    V(bestindex,:) = rands(1,numsum);
    fitness(bestindex) = fun(pop(bestindex,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    [bestfitness, bestindex] = min(fitness);
end
%整合归档集
init_best = [init_fitness;init_index];
init_pop = pop;%暂存初始pop

zbest = pop(bestindex,:);  %全局最佳
gbest = pop;    %个体最佳
fitnessgbest = fitness;   %个体最佳适应度值ֵ
fitnesszbest = bestfitness;   %全局最佳适应度值�ֵ
%初始化博弈参数
strategy_num = 3;
mutation_num = 3;  %变异��
flag = zeros(1,strategy_num);%控制变异的参数��
p = ones(1,strategy_num) * (1 / strategy_num);
success_mem = zeros(1,strategy_num);
failure_mem = zeros(1,strategy_num);
rk = cumsum(ones(1,strategy_num) ./ strategy_num);
strategy_improve = zeros(1,strategy_num);

%初始化Sarsa相关参数�
alpha = 0.1;  %学习速率
gamma = 0.8; %折扣因子
epsilon = 0.95; %贪婪因子�
delt_fitness = 0;
Qtable = zeros(4, 4);
current_fitness = 0;
elite_archive = [];  %将pop执行Sarsa操作后得到的较优解放到elite_archive精英集中进行存档

%初始化Q-learning相关参数
delt_fitness_Q = 0;
Qtable_Q = zeros(4, 4);
current_fitness_Q = 0 ;

%% 迭代寻优
for i = 1:maxgen  
    for j = 1:sizepop
        probility = rand;
        %大群体
        if probility <= rk(1)
            strategy = 1;
            V(j,:) = V(j,:) + (c1 + c2) * rand * (gbest(j,:) - pop(j,:));
            V(j,find(V(j,:) > Vmax)) = Vmax;
            V(j,find(V(j,:) < Vmin)) = Vmin;
        elseif probility <= rk(2)
            strategy = 2;
            V(j,:) = V(j,:) + (c1 + c2) * rand * (zbest - pop(j,:));
            V(j,find(V(j,:) > Vmax)) = Vmax;
            V(j,find(V(j,:) < Vmin)) = Vmin;
        elseif probility <= rk(3)
            strategy = 3;
            V(j,:) = V(j,:) + c1 * rand * (gbest(j,:) - pop(j,:)) + c2 * rand * (zbest - pop(j,:));
            V(j,find(V(j,:) > Vmax)) = Vmax;
            V(j,find(V(j,:) < Vmin)) = Vmin;
        end
        %控制范围
        pop(j,:) = pop(j,:) + V(j,:);
        pop(j,find(pop(j,:) > popmax)) = popmax;
        pop(j,find(pop(j,:) < popmin)) = popmin;
        
        %适应度值
        fitness(j) = fun(pop(j,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
       
        %个体最优更新
        if fitness(j) < fitnessgbest(j)
            strategy_improve(strategy) = strategy_improve(strategy) + (fitnessgbest(j) - fitness(j)) / fitnessgbest(j);
            gbest(j,:) = pop(j,:); %记录最好的位置和适应度值�ֵ
            fitnessgbest(j) = fitness(j);
            success_mem(strategy) = success_mem(strategy) + 1;
        else
            failure_mem(strategy) = failure_mem(strategy) + 1;
        end
    
        %群体最优更新 �� 
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
            history = [history fitnesszbest]; %记录群体历史最优
        end
    end
    
    %复制动态
    total = success_mem + failure_mem;
    total(find(total == 0)) = 1;
    strategy_improve = strategy_improve ./ total;
    if isequal(strategy_improve,zeros(1,strategy_num))  %初始化
       strategy_improve = ones(1,strategy_num);
    end
    strategy_improve(find(strategy_improve == 0)) = 0.1 * min(strategy_improve(strategy_improve ~= 0)); %防止某些策略的被选的次数为0
    strategy_improve = strategy_improve ./ sum(strategy_improve);
    
    f = strategy_improve;
    p = p + (f - sum(p .* f)) .* p .* lr; %更新概率p
    
    p(find(p <= 0)) = 0;  %确保概率p非负
    p = p ./ (sum(p));  
    rk = cumsum(p);   %计算概率累计
    
    %% 自适应变异��
    for j=1:3
        if flag(1,j) >= mutation_num && p(1,j) > 0.5
            flag(1,j) = flag(1,j)+1;
        elseif flag(1,j) < mutation_num && p(1,j) > 0.5
            part_improve = p(1,j) - 0.5;  %p超过0.5的部分，这部分用于调整其他策略的概率
            distribe = p;
            distribe(:,j) = 0;
            distribe = part_improve .* (distribe ./ sum(distribe)); %将part_improve按比例分配给其他策略（通过distribe./sum(distribe) 确保分配比例）
            p = distribe + p; %更新p，把调整后的distribe加回原概率矩阵p
            p(1,j) = 0.5; %确保每次变异后的三种策略比重之和为1
            flag(1,j) = flag(1,j) + 1;
        end
    end
    %随机点变异����
    pos = unidrnd(numsum);  %生成一个（1，numsum）之间的随机整数pos，用于定位变异点
    if rand > 0.90
      pop(j,pos) = 5 * rands(1,1);  %以10%概率执行随机变异
    end
    success_mem = zeros(1,strategy_num);  %每种策略成功的次数
    failure_mem = zeros(1,strategy_num);  %每种策略失败的次数
    strategy_improve = zeros(1,strategy_num);

    % global search based on SARSA
    last_fitness = current_fitness;
    current_state = GetState(delt_fitness);
    %choose and execute action
    if rand < epsilon || all(Qtable(current_state, :) == 0)
        action = ceil(rand * 4);
    else
        [~, action] = max(Qtable(current_state, :));
    end
    % randomly select two individuals
    selected_individual_1 = randi(size(pop, 1)); 
    selected_individual_2 = randi(size(pop, 1));
    while selected_individual_1 == selected_individual_2
        selected_individual_2 = randi(size(pop, 1));
    end
    switch(action)
        case 0
            [current_fitness, pop, elite_archive] = PMX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps); %PMX_crossover
        case 1
            [current_fitness, pop, elite_archive] = OX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);  %OX_crossover
        case 2
            [current_fitness, pop, elite_archive] = CX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);  %CX_crossover
        case 3
            [current_fitness, pop, elite_archive] = PBX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps); %PBX_crossover
    end
    %update Qtable
    %calculate the fitness of the operated individual
    delt_fitness = last_fitness - current_fitness;
    if delt_fitness > 0 % find a nice solution
        reward = 10;
    else
        reward = 0;
        
    end
    next_state = GetState(delt_fitness);
    col_idx = randi(4);
    maxReward = Qtable(next_state, col_idx);
    Qtarget = reward + gamma * maxReward;
    Qtable(current_state, action) = Qtable(current_state, action) + alpha * (Qtarget - Qtable(current_state, action));
    %Qnew(st, at)=Q(st, at)+alpha[gamma*Q(st+1, at)-Q(st, at)]   SARSA算法更新Q的公式
    
    % local search based on Q-learning
    if isempty(elite_archive)
    else
        last_fitness_Q = current_fitness_Q;
        current_state_Q = GetState(delt_fitness_Q);
        % choose and execute action
        if rand < epsilon || all(Qtable_Q(current_state_Q, :) == 0)
            action_Q = ceil(rand * 4);
        else
            [~, action_Q] = max(Qtable_Q(current_state_Q, :));
        end
        % randomly select one individual from best_pop
        selected_individual = randi(size(elite_archive, 1)); 
        switch(action_Q)
            case 0
                [current_fitness_Q, pop] = Operator_1(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 1
                [current_fitness_Q, pop] = Operator_2(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 2
                [current_fitness_Q, pop] = Operator_3(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 3
                [current_fitness_Q, pop] = Operator_4(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 4
                [current_fitness_Q, pop] = Operator_5(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
        end
        %update Qtable
        %calculate the fitness of the operated individual
        delt_fitness_Q = last_fitness_Q - current_fitness_Q;
        if delt_fitness_Q > 0
            reward_Q = 10;
        else
            reward_Q = 0;
        end
        next_state_Q = GetState(delt_fitness_Q);
        maxReward_Q = max(Qtable_Q(next_state_Q, :));
        Qtarget_Q = reward_Q + gamma * maxReward_Q;
        Qtable_Q(current_state_Q, action_Q) = Qtable_Q(current_state_Q, action_Q) + alpha * (Qtarget_Q - Qtable_Q(current_state_Q, action_Q));
        
    end
       
end

%归档比较
if isempty(init_best)   
else    
    [init_min, index] = min(init_best(1,:));
    if init_min < fitnesszbest
        zbest = init_pop(index,:);
        fitnesszbest = init_min;
    end
end

% %% 结果分析
x = zbest;
figure(1)
f1 = plot(history,'rd');
xlabel('time','Fontname','Times New Roman');
ylabel('zbest','Fontname','Times New Roman');

%% 把最优初始阀值权值赋予网络预测
% %用遗传算法优化的BP网络进行值预测
w1 = x(1:inputnum * hiddennum);
B1 = x(inputnum * hiddennum + 1:inputnum * hiddennum + hiddennum);
w2 = x(inputnum * hiddennum + hiddennum + 1:inputnum * hiddennum + hiddennum + hiddennum * outputnum);
B2 = x(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1:inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);
B2 = B2';

net.iw{1,1} = reshape(w1,hiddennum,inputnum);
net.lw{2,1} = reshape(w2,outputnum,hiddennum);
net.b{1} = reshape(B1,hiddennum,1);
net.b{2} = B2;

%% BP网络训练
%网络进化参数����
net.trainParam.epochs = 200;
net.trainParam.lr = 0.1;
net.trainParam.goal = 0.00001;

%网络训练�
[net,per2] = train(net,ginputn,goutput_train);
save ('PSOBP','net');
