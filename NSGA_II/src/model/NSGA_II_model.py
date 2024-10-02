import math
import random



class NAGA_II():
    def __init__(self,
                 inital_data:list=None,
                 max_gen=100,
                 ):
        self.pop_size=len(inital_data)
        self.max_gen=max_gen
        self.min_x=min(inital_data)
        self.max_x=max(inital_data)
        self.inital_data=inital_data
    # First function to optimize
    def function1(self,x):
        value = -x ** 2
        return value


    # Second function to optimize
    def function2(self,x):
        value = -(x - 2) ** 2
        return value


    # Function to find index of list,且是找到的第一个索引
    def index_of(self,a, list):
        for i in range(0, len(list)):
            if list[i] == a:
                return i
        return -1


    # Function to sort by values 找出front中对应值的索引序列
    def sort_by_values(self,list1, values):
        sorted_list = []
        while (len(sorted_list) != len(list1)):
            if self.index_of(min(values), values) in list1:
                sorted_list.append(self.index_of(min(values), values))
            values[self.index_of(min(values), values)] = math.inf
        return sorted_list


    # Function to carry out NSGA-II's fast non dominated sort
    def fast_non_dominated_sort(self,values1, values2):
        S = [[] for i in range(0, len(values1))]  # len(values1)个空列表
        front = [[]]
        n = [0 for i in range(0, len(values1))]
        rank = [0 for i in range(0, len(values1))]
        # 将front0全部整理出来了，并未对front1-n等进行整理
        for p in range(0, len(values1)):
            S[p] = []
            n[p] = 0
            for q in range(0, len(values1)):
                if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                    values1[p] >= values1[q] and values2[p] > values2[q]) or (
                    values1[p] > values1[q] and values2[p] >= values2[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                    values1[q] >= values1[p] and values2[q] > values2[p]) or (
                    values1[q] > values1[p] and values2[q] >= values2[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)
        i = 0
        # 该循环能将所有的个体全部进行分类，显然最后一层的个体中，没有可以支配的个体了
        while (front[i] != []):
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            front.append(Q)

        del front[len(front) - 1]  # 删除了最后一层无支配个体的front层,最后一层是空集
        return front


    # Function to calculate crowding distance  同层之间的一个计算
    def crowding_distance(self,values1, values2, front):
        # distance = [0 for i in range(len(front))]
        lenth = len(front)
        for i in range(lenth):
            distance = [0 for i in range(lenth)]
            sorted1 = self.sort_by_values(front, values1[:])  # 找到front中的个体索引序列
            sorted2 = self.sort_by_values(front, values2[:])  # 找到front中的个体索引序列
            distance[0] = 4444
            distance[lenth - 1] = 4444
            for k in range(2, lenth - 1):
                distance[k] = distance[k] + (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (
                        max(values1) - min(values1))
                # print("/n")
                print("k:", k)
                print("distance[{}]".format(k), distance[k])
            for k in range(2, lenth - 1):
                distance[k] = distance[k] + (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                        max(values2) - min(values2))
        return distance


    # #Function to carry out the crossover
    def crossover(self,a, b):
        r = random.random()
        if r > 0.5:
            return self.mutation((a + b) / 2)
        else:
            return self.mutation((a - b) / 2)


    # #Function to carry out the mutation operator
    def mutation(self,solution):
        mutation_prob = random.random()
        if mutation_prob < 1:
            solution = self.min_x + (self.max_x - self.min_x) * random.random()
        return solution

    def run(self):
        solution = self.inital_data
        print('solution', solution)
        gen_no = 0
        while (gen_no < self.max_gen):
            print('\n')
            print('gen_no:迭代次数', gen_no)
            function1_values = [self.function1(solution[i]) for i in range(0, self.pop_size)]
            function2_values = [self.function2(solution[i]) for i in range(0, self.pop_size)]
            print('function1_values:', function1_values)
            print('function2_values:', function2_values)
            non_dominated_sorted_solution = self.fast_non_dominated_sort(function1_values[:], function2_values[:])
            print('front', non_dominated_sorted_solution)
            # print("The best front for Generation number ",gen_no, " is")
            # for valuez in non_dominated_sorted_solution[0]:
            #     print("solution[valuez]",round(solution[valuez],3),end=" ")
            #     print("\n")
            crowding_distance_values = []
            for i in range(0, len(non_dominated_sorted_solution)):
                crowding_distance_values.append(
                    self.crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
            print("crowding_distance_values", crowding_distance_values)
            solution2 = solution[:]
            # Generating offsprings
            while (len(solution2) != 2 * self.pop_size):
                a1 = random.randint(0, self.pop_size - 1)
                b1 = random.randint(0, self.pop_size - 1)
                solution2.append(self.crossover(solution[a1], solution[b1]))
            print('solution2', solution2)
            function1_values2 = [self.function1(solution2[i]) for i in range(0, 2 * self.pop_size)]
            function2_values2 = [self.function2(solution2[i]) for i in range(0, 2 * self.pop_size)]
            non_dominated_sorted_solution2 = self.fast_non_dominated_sort(function1_values2[:],
                                                                     function2_values2[:])  # 2*pop_size
            print('non_dominated_sorted_solution2', non_dominated_sorted_solution2)
            # print("\n")
            crowding_distance_values2 = []
            for i in range(0, len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(
                    self.crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

            print('crowding_distance_values2', crowding_distance_values2)
            new_solution = []
            for i in range(0, len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2_1 = [
                    self.index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                    range(0, len(non_dominated_sorted_solution2[i]))]
                print('non_dominated_sorted_solution2_1:', non_dominated_sorted_solution2_1)
                front22 = self.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
                print("front22", front22)
                front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                         range(0, len(non_dominated_sorted_solution2[i]))]
                print('front', front)
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == self.pop_size):
                        break
                if (len(new_solution) == self.pop_size):
                    break
            solution = [solution2[i] for i in new_solution]
            gen_no = gen_no + 1
        #
        # Lets plot the final front now
        self.fc1 =function1_values
        self.fc2 =function2_values
        self.optim_out=solution
        self.out=[self.optim_out,self.fc1,self.fc2]
        print(self.out)
        for i in self.out:
            print(i)
    def result(self,output):
        with open(output,'w') as f:
            f.write("result\n")
            for i in range(len(self.optim_out)):
                f.write(str(self.optim_out[i])+"\n")


#
# naga=NAGA_II()
# naga.run()
