"""
.. module:: Experiment

Experiment
*************

:Description: Experiment

    

:Authors: bejar
    

:Version: 

:Created on: 30/10/2017 13:00 

"""
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from RatBrain.Config.Constants import datafiles, datapath


__author__ = 'bejar'


class Experiment:
    data = None
    cord = None
    filename = None
    maxX = 512
    maxY = 672
    corr = None

    def __init__(self, filename):
        """
        Class for the Experiment data
        :param filename:
        """

        self.filename = filename
        self.data = pd.read_csv(datapath + self.filename + '.csv')
        ncol = self.data.shape[1]
        self.data.columns = ['cell%d' % (i+1) for i in range(ncol)]
        self.coord = pd.read_csv(datapath + self.filename + 'C.csv')

    def info(self):
        """
        Info for the experiment
        :return:
        """
        print self.data.head()
        print self.data.shape

    def corr_plot(self):
        """
        Cell Signal correlation

        :return:
        """

        self.corr = self.data.corr()
        mask = np.zeros_like(self.corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))
        # cmap = sn.diverging_palette(220, 10, as_cmap=True)

        sn.heatmap(self.corr, mask=mask, vmax=1, vmin=0, center=0.5, cmap=plt.get_cmap('Reds'),
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=False)
        plt.yticks(rotation=0)
        # plt.title(df1.split('.')[0][2:] + ' ' + df2.split('.')[0][2:])
        # plt.savefig('cross-' + df1.split('.')[0][2:] + '-' + df2.split('.')[0][2:] + '.pdf', orientation='landscape', format='pdf')
        plt.show()
        plt.close()

    def coord_plot(self):
        """
        Plot coords

        :return:
        """
        f, ax = plt.subplots(figsize=(11, 9))
        plt.plot(self.coord['x'], self.coord['y'], 'r.')
        plt.show()
        plt.close()

    def connection_plot(self, thresh):
        """
        Plot coords

        :return:
        """
        corr = self.data.corr()

        lpairs = []
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= thresh:
                    lpairs.append((i,j))

        print lpairs
        f, ax = plt.subplots(figsize=(11, 9))
        plt.plot(self.coord['x'], self.coord['y'], 'r.')

        for p in lpairs:
            print self.coord['x'].iloc[p[0]], self.coord['y'].iloc[p[0]], \
                self.coord['x'].iloc[p[1]], self.coord['y'].iloc[p[1]]
            plt.plot([self.coord['x'].iloc[p[0]], self.coord['x'].iloc[p[1]]],
                     [self.coord['y'].iloc[p[0]], self.coord['y'].iloc[p[1]]], 'b-')

        plt.show()
        plt.close()


if __name__ == '__main__':

    exp = Experiment(datafiles[4])

    # exp.info()


    exp.corr_plot()
    exp.connection_plot(0.6)

