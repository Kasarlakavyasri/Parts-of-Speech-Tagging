###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids: Kavya Sri Kasarla(kkasarla) and Prathyusha Reddy Thumma(pthumma)
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    parts_of_speech=['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
    init_dist=[0]*12
    fin_dist=[0]*12
    cnt=float(1/100000000000000000)
    mcmc_lst=[]
    emm_cnt={}
    emm_prob={}
    trans_cnt=[[0 ]*12 for i in range(12)]
    trans_prob=[[0]*12 for i in range(12)]
    str_trans1=[[0]*12 for i in range(12)]
    str_trans2=[[0]*12 for i in range(12)]
    mcmc_lst=[]

    # Cal probability of sentnce
    
    def posterior(self, model, sentence, label):
        if model == "Simple":
           
            return (sum(self.simplified(sentence)[1]))
        elif model == "HMM":
            return(self.hmm_viterbi(sentence)[1])
        elif model == "Complex":
            return ((self.hmm_viterbi(sentence)[1]+sum(self.simplified(sentence)[1]))/2)
        else:
            print("Unknown algo!")

    
    # training
    def train(self, data):
        fst_cnt=[0]*12
        initwrd_cnt=[0]*12
        finwrd_cnt=[0]*12
        self.tot_wrds=0
        tot_sent=0
        
        self.emm_cnt=self.emm_prob={j:[Solver.cnt]*12 for i in data for j in i[0]}
        for i in range(len(data)):
            self.tot_wrds+=len(data[i][0])
            tot_sent+=1
            #Cal init state distribution
            initwrd_cnt[Solver.parts_of_speech.index(data[i][1][0])]+=1
            finwrd_cnt[Solver.parts_of_speech.index(data[i][1][-1])]+=1
            for j in range(len(data[i][1])):
                index=Solver.parts_of_speech.index(data[i][1][j])
                fst_cnt[index]+=1
                #Counts for prob
                if(j<len(data[i][1])-1):
                    Solver.trans_cnt[index][Solver.parts_of_speech.index(data[i][1][j+1])]+=1

                #Count emission prob
                self.emm_cnt[data[i][0][j]][index]+=1

        for i in range(12):
            

            if(initwrd_cnt[i]==0):
                Solver.init_dist[i]=Solver.cnt
            else:
                Solver.init_dist[i]=float(initwrd_cnt[i])/len(data)

            if(finwrd_cnt[i]==0):
                Solver.fin_dist[i]=Solver.cnt
            else:
                Solver.fin_dist[i]=float(finwrd_cnt[i]/len(data))
            #Cal transition prob
            for j in range(12):
                if(sum(Solver.trans_cnt[i])==0 or Solver.trans_cnt[i][j]==0):
                    Solver.trans_prob[i][j]=Solver.cnt
                else:
                    Solver.trans_prob[i][j]=float(Solver.trans_cnt[i][j])/sum(Solver.trans_cnt[i])

        self.prior_probability=[]
        dfg=0

        for i in range(12):
            self.prior_probability.append(float(fst_cnt[i])/self.tot_wrds)

        for i in self.emm_prob:
            for j in range(12):
                if(fst_cnt[j]==0 or self.emm_cnt[i][j]==0):
                    self.emm_prob[i][j]=Solver.cnt
                else:
                    self.emm_prob[i][j]=float(self.emm_cnt[i][j])/fst_cnt[j]



        p1dict={j:[0]*12 for j in Solver.parts_of_speech}
        p2dict={j:[0]*12 for j in Solver.parts_of_speech}

        for i in range(len(Solver.parts_of_speech)):
            p1dict={j:[0]*12 for j in Solver.parts_of_speech}
            for k in range(len(data)):
                if data[k][1][len(data[k][0])-1]==Solver.parts_of_speech[i]:
                    p1dict[data[k][1][0]][Solver.parts_of_speech.index(data[k][1][len(data[k][0])-2])]+=1
            Solver.mcmc_lst.append(p1dict)
      
        for k in range(len(data)):
            p2dict[data[k][1][0]][Solver.parts_of_speech.index(data[k][1][len(data[k][0])-2])]+=1  

        for i in range(0,len(Solver.mcmc_lst)):
            for j in Solver.parts_of_speech:
                for k in range(0,12):
                    if Solver.mcmc_lst[i][j][k]==0:
                        Solver.mcmc_lst[i][j][k]=float(Solver.cnt)
                    else:
                        Solver.mcmc_lst[i][j][k]=float(Solver.mcmc_lst[i][j][k]/p2dict[j][k])



        for i in range(len(data)):
            Solver.str_trans1[Solver.parts_of_speech.index(data[i][1][0])][Solver.parts_of_speech.index(data[i][1][len(data[i][0])-1])]+=1
            Solver.str_trans2[Solver.parts_of_speech.index(data[i][1][len(data[i][0])-2])][Solver.parts_of_speech.index(data[i][1][len(data[i][0])-1])]+=1


        for i in range(12):

            for j in range(12):
                if(sum(Solver.str_trans1[i])==0 or Solver.str_trans1[i][j]==0):
                    Solver.str_trans1[i][j]=Solver.cnt
                else:
                    Solver.str_trans1[i][j]=float(Solver.str_trans1[i][j])/sum(Solver.str_trans1[i])
                if(sum(Solver.str_trans2[i])==0 or Solver.str_trans2[i][j]==0):
                    Solver.str_trans2[i][j]=Solver.cnt
                else:
                    Solver.str_trans2[i][j]=float(Solver.str_trans2[i][j])/sum(Solver.str_trans2[i])



    
    def simplified(self, sentence):
        speech_part=[]
        post=[]
        for i in range(len(sentence)):
            self.most_likely_prob=[0]*12
            if(sentence[i] not in self.emm_prob):
               
                self.emm_prob[sentence[i]]=[Solver.cnt,Solver.cnt,Solver.cnt,Solver.cnt,Solver.cnt,1-(Solver.cnt)*11,Solver.cnt,Solver.cnt,Solver.cnt,Solver.cnt,Solver.cnt,Solver.cnt]
            for j in range(12):
                self.most_likely_prob[j]= math.log(self.emm_prob[sentence[i]][j])+math.log(self.prior_probability[j])
              

            speech_part.append(Solver.parts_of_speech[self.most_likely_prob.index(max(self.most_likely_prob))])
            post.append(max(self.most_likely_prob))
        return  (speech_part,post)

    def hmm_viterbi(self, sentence):
       
        res = [[0 for t in range(len(sentence))]for j in range(12)]
        mem = [[0 for i in range(12)]for t in range(len(sentence))]
        for t in range(len(sentence)):
            for j in range(12):
                if t==0:
                    mem[t][j] = math.log(Solver.init_dist[j]) + math.log(self.emm_prob[sentence[t]][j])
                else:
                    pos_cost = [mem[t - 1][i]+math.log(Solver.trans_prob[i][j] )for i in range(12)]
                    maxc = max(pos_cost)
                    mem[t][j] = math.log(self.emm_prob[sentence[t]][j]) + maxc
                    res[j][t] = pos_cost.index(maxc)
        str_temp = []

        ind = mem[len(sentence) - 1].index(max(mem[len(sentence) - 1]))

        str_temp.append(Solver.parts_of_speech[ind])
        i = len(sentence) - 1

        while (i > 0):
            ind = res[ind][i]
            str_temp.append(Solver.parts_of_speech[ind])
            i -= 1
        return (str_temp[::-1],max(mem[len(sentence) - 1]))
    
    def mostlikely(sample_lst):
        return [Counter(column).most_common(1)[0][0] for column in zip(*sample_lst)]

    def complex_mcmc(self, sentence):
        
        temp_sample=["noun"]*len(sentence)
        samples=[]
        
        for no_of_iteration in range(500):  
            for i in range(0,len(sentence)):
                
                prob_vals=[]
                prob_tot=0
                for j in Solver.parts_of_speech:
                    if len(sentence)>2:
                        if i==0:
                            a_1 = Solver.init_dist[Solver.parts_of_speech.index(j)]
                            b_1=self.emm_prob[sentence[i]][Solver.parts_of_speech.index(j)]
                            c_1=self.trans_prob[Solver.parts_of_speech.index(j)][Solver.parts_of_speech.index(temp_sample[i+1])]
                            d_1=self.mcmc_lst[Solver.parts_of_speech.index(temp_sample[len(sentence)-1])][j][Solver.parts_of_speech.index(temp_sample[len(sentence)-2])]
                            p=float(a_1*b_1*c_1*d_1)

                        if i >0 and i <len(sentence)-2:
                            e_1=self.emm_prob[sentence[i]][Solver.parts_of_speech.index(j)]
                            f_1=self.trans_prob[Solver.parts_of_speech.index(j)][Solver.parts_of_speech.index(temp_sample[i+1])]
                            g_1=self.trans_prob[Solver.parts_of_speech.index(temp_sample[i-1])][Solver.parts_of_speech.index(j)]
                            p=float(e_1*f_1*g_1)
                        if i==len(sentence)-1:
                            h_1=self.emm_prob[sentence[i]][Solver.parts_of_speech.index(j)]
                            k_1=self.mcmc_lst[Solver.parts_of_speech.index(j)][temp_sample[0]][Solver.parts_of_speech.index(temp_sample[len(sentence)-2])]
                            p=float(h_1*k_1)
                        if i== len(sentence)-2:
                            l_1=self.emm_prob[sentence[i]][Solver.parts_of_speech.index(j)]
                            q_1=self.mcmc_lst[Solver.parts_of_speech.index(temp_sample[len(sentence)-1])][temp_sample[0]][Solver.parts_of_speech.index(j)]
                            z_1=self.trans_prob[Solver.parts_of_speech.index(temp_sample[len(sentence)-3])][Solver.parts_of_speech.index(j)]
                            p=float(l_1*q_1*z_1)
                    if len(sentence)==1:
                        a1 = Solver.init_dist[Solver.parts_of_speech.index(j)]
                        b1=self.emm_prob[sentence[i]][Solver.parts_of_speech.index(j)]
                        p=float(a1*b1)
                    if len(sentence)==2:
                        if i==0:
                            ak1 = Solver.init_dist[Solver.parts_of_speech.index(j)]
                            b1=self.emm_prob[sentence[i]][Solver.parts_of_speech.index(j)]
                            c1=self.trans_prob[Solver.parts_of_speech.index(j)][Solver.parts_of_speech.index(temp_sample[i+1])]
                            p=float(ak1*b1*c1)
                        if i==1:
                            g1=self.trans_prob[Solver.parts_of_speech.index(temp_sample[i-1])][Solver.parts_of_speech.index(j)]
                            e1=self.emm_prob[sentence[i]][Solver.parts_of_speech.index(j)]
                            p=float(g1*e1)
                    prob_tot += p
                    prob_vals.append(p)
                c=0
                r = random.uniform(0.00,1.00)
                for q in range(0, len(prob_vals)):
                    prob_vals[q] =(prob_vals[q]/prob_tot)
                    c += prob_vals[q]
                    prob_vals[q] = c
                    if r < prob_vals[q]:
                        o = q
                        break
                temp_sample[i]=Solver.parts_of_speech[o]
            samples.append(temp_sample)
        del samples[:100]
        list_count=[]
        samp_lst= Solver.mostlikely(samples)
        for i in range(0,len(sentence)):
            n=0
            for k in range(0,len(samples)):
                if samples[k][i]==samp_lst[i]:
                    n=n+1
            list_count.append(n)
        return(samp_lst)
    
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)[0]
        elif model == "HMM":
            return self.hmm_viterbi(sentence)[0]
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")


