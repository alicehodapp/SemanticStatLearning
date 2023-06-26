import pandas as pd
import numpy as np
import argparse 
from scipy.special import digamma, gammaln
import os
import re 

class SequentialBayesianDcat:
    """Sequential Bayesian learner model with Dirichlet-categorical distribution.
    Inputs: Mobs, N x dim matrix of observed classes, one-hot encoded and ordered vertically from first to last observation.
    functions:
        compute_alphas(tau): computes the parameters of the Dirichlet distribution at each observation, for a given value of tau
        compute_surprise(tau): computes the model's surprise values (bs, pe) for a given value of tau. Calls compute_alphas."""
        
    def __init__(self,Mobs, model):
        self.dim = Mobs.shape[-1]
        self.length = len(Mobs)
        self.model= model

        if self.model == 'CF':
            self.Mobs = np.concatenate((np.ones((1,self.dim)), Mobs)) # add artificial row of prior observations
        elif self.model == 'TP':
            self.orig = Mobs
            transitions = np.zeros((self.length,5,5),dtype=int)
            for indx in np.arange(1,self.length):
                transitions[indx,Mobs[indx-1,:]==1,Mobs[indx,:]==1] = 1
                transitions[0,:,:] = np.ones((1,5,5)) # add artificial row of prior observations
                self.Mobs = transitions

    # estimate parameters of the Dirichlet distribution
    def compute_alphas(self, tau):

        if self.model == 'CF':
            alphas = [self.Mobs[0]] # prior
            for idx in np.arange(1,len(self.Mobs)): # loop through all observations excluding prior
                # exponential filter for previous trials
                memfilt = idx - np.tile(np.arange(idx+1), (self.dim,1)).T # last observation is always fully included
                memfilt[0] = np.zeros(self.dim)  # the prior row is always fully included
                ofilt = np.exp(-memfilt/tau)*self.Mobs[:idx+1] # filter observations
        
                # compute alpha
                a = np.sum(ofilt, axis=0)
                alphas.append(a)

        if self.model == 'TP':
            alphas = []
            alphas = [self.Mobs[0,:,:]] # prior
            for idx in np.arange(1,len(self.Mobs)): # loop through all observations excluding prior
                memfilt = idx - np.tile(np.arange(idx+1), (len(self.Mobs[0,0,:]),len(self.Mobs[0,0,:]),1)).T # last observation is always fully included
                memfilt[0] = np.zeros((len(self.Mobs[0,0,:]),len(self.Mobs[0,0,:])))  # the prior row is always fully included
                ofilt = np.exp(-memfilt/tau)*self.Mobs[:idx+1] # filter observations

                # compute alpha
                a = np.sum(ofilt, axis=0)
                alphas.append(a)
            alphas = np.array(alphas)

        return np.array(alphas)

    # compute surprise measures
    def compute_surprise(self, tau):
        alphas = self.compute_alphas(tau)

        if self.model == 'CF':
            a_ps, a_s = alphas[:-1], alphas[1:] # predictive and updated alphas
        elif self.model == 'TP':
            a_ps=np.zeros((len(alphas)-1,5), float)
            a_s=np.zeros((len(alphas)-1,5), float)

            a_ps[0,:]=np.ones(5)
            TP = int(np.where(self.Mobs[1,:,:]==1)[0])
            a_s[0,:]=alphas[1,TP,:]
            for x in np.arange(1,len(alphas)-1):
                TP = int(np.where(self.Mobs[x+1,:,:]==1)[0]) #get relevant row (the current category)
                a_ps[x,:] = alphas[x,TP,:] #predictive alphas 
                a_s[x,:] = alphas[x+1,TP,:] # updated alphas

        # compute Bayesian surprise
        a_part = gammaln(np.sum(a_ps, axis=1)) - np.sum(gammaln(a_ps),axis=1)
        b_part = gammaln(np.sum(a_s,axis=1)) - np.sum(gammaln(a_s),axis=1)
        ab_part = np.sum((a_s-a_ps)*(digamma(a_ps)-np.tile(digamma(np.sum(a_ps, axis=1)), (self.dim,1)).T),axis=1)  
        bs = a_part - b_part + ab_part

        # compute prediction error
        if self.model == 'CF':
            pe = -np.sum((self.Mobs[1:])*np.log(a_ps/np.tile(np.sum(a_ps, axis=1), (self.dim,1)).T), axis=1)
        elif self.model == 'TP':
            pe = -np.sum((self.orig[1:])*np.log(a_ps/np.tile(np.sum(a_ps, axis=1), (self.dim,1)).T), axis=1)
            #no transition for first presentation
            bs = np.insert(bs, 0, np.nan, axis=0)
            pe = np.insert(pe, 0, np.nan, axis=0)
        
        return bs, pe

def main(taus, model):
    # load in all data files and concatinate them
    df_all = []
    directory = os.fsencode('/Users/alice/transition_probabilities/BCL_data')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            df =  pd.read_csv('/Users/alice/transition_probabilities/BCL_data/' + filename, header=None) 
            df_all.append(df) 

    df_all = pd.concat(df_all, axis=0, ignore_index=True)
    df_all.columns = ['subj','trial', 'catID']
    df_all = df_all[df_all.catID != 999]# drop rows with non-words (catch trials)

    # Compute surprise values for all subjects and save in csv file
    surp_all = []
    for subject, df in df_all.groupby('subj'):
        print('Processing subject ' + str(subject))
        
        prev_cat = np.concatenate((np.array([5]), df['catID'].values[:-1])) #cat moved by 1
        df['catSwitch'] = (df['catID'] != prev_cat).astype(int)
        
        cats_onehot = pd.get_dummies(df['catID']).values
        subct = SequentialBayesianDcat(cats_onehot, model)
        
        for tau in taus:
            bs, pe = subct.compute_surprise(tau)
            tmp = pd.DataFrame(data = {'BS': bs, 'PE': pe, 'tau': tau, 'trial': df.trial})
            df_surp = pd.merge(df, tmp, on='trial')
            surp_all.append(df_surp)              

    surp_all = pd.concat(surp_all)
    output_folder = '/Users/alice/transition_probabilities/transition_prob/BCL/output/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    surp_all.to_csv(output_folder + 'surprise_' + model + '.csv')

if __name__ == "__main__":

    def parseNumList(string):
        m = re.match(r'(\d+)(?:-(\d+))?$', string)
        start = m.group(1)
        end = m.group(2) or start
        return list(range(int(start,10), int(end,10)+1))

    parser = argparse.ArgumentParser()
    parser.add_argument('-taus', '--forget_params', action="store",
                        default='1-50', type=parseNumList,
                        help='Exponential weighting parameters for memory: enter a range e.g. 0-5')
    parser.add_argument('-model', '--model', action="store", default="CF",
                        type=str,
                        help='Category frequeny (CF) or transition probability (TP)')
    args = parser.parse_args()

    taus = args.forget_params
    model = args.model

    main(taus, model)