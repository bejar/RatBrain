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
from scipy.signal import detrend
from scipy.signal import savgol_filter
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

    def corr_plot(self, begin):
        """
        Cell Signal correlation

        :return:
        """

        self.corr = self.data.iloc[begin:begin+1200, :].corr()
        mask = np.zeros_like(self.corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))
        # cmap = sn.diverging_palette(220, 10, as_cmap=True)

        sn.heatmap(self.corr, mask=mask, vmax=1, vmin=0.6, center=0.8, cmap=plt.get_cmap('Reds'),
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
        fig = plt.figure(figsize=(15, 15))
        for i, begin in enumerate([0, 1200, 2400, 3600]):

            corr = self.data.iloc[begin:begin+1200, :].corr()

            ax = fig.add_subplot(2,2, i+1)
            lpairs = []
            for i in range(corr.shape[0]):
                for j in range(i+1, corr.shape[0]):
                    if corr.iloc[i,j] >= thresh:
                        lpairs.append((i,j))

            # print lpairs
            # f, ax = plt.subplots(figsize=(11, 9))
            plt.plot(self.coord['x'], self.coord['y'], 'r.')

            for p in lpairs:
                # print self.coord['x'].iloc[p[0]], self.coord['y'].iloc[p[0]], \
                #     self.coord['x'].iloc[p[1]], self.coord['y'].iloc[p[1]]
                plt.plot([self.coord['x'].iloc[p[0]], self.coord['x'].iloc[p[1]]],
                         [self.coord['y'].iloc[p[0]], self.coord['y'].iloc[p[1]]], 'b-')

        plt.show()
        plt.close()

    def connection_plot_pairs(self, thresh):
        """
        Plot coords

        :return:
        """

        step = int(self.data.shape[0]/4.0)
        data = self.data.values.copy()
        for i in range(data.shape[1]):
            data[:,i] = savgol_filter(data[:,i], 41, 3)
        print data.shape
        for s in range(4):

            dtrended = detrend(data[s*step:(s+1)*step, :], axis=0)

            # corr = self.data.iloc[s*step:(s+1)*step, :].corr()
            corr = np.corrcoef(dtrended, rowvar=False)
            acorr = np.corrcoef(data[s*step:(s+1)*step, :], rowvar=False)

            print corr.shape
            lpairs = []
            for i in range(corr.shape[0]):
                for j in range(i+1, corr.shape[0]):
                    if corr[i,j] >= thresh:
                        lpairs.append((i,j))

            alpairs = []
            for i in range(acorr.shape[0]):
                for j in range(i+1, acorr.shape[0]):
                    if acorr[i,j] >= thresh:
                        alpairs.append((i,j))

            fig = plt.figure(figsize=(15, 15))
            plt.title(self.filename + ' STEP= ' + str(s))
            plt.plot(self.coord['x'], self.coord['y'], 'r.')
            for p in lpairs:
                print(p)
                # print self.coord['x'].iloc[p[0]], self.coord['y'].iloc[p[0]], \
                #     self.coord['x'].iloc[p[1]], self.coord['y'].iloc[p[1]]
                plt.plot([self.coord['x'].iloc[p[0]], self.coord['x'].iloc[p[1]]],
                         [self.coord['y'].iloc[p[0]], self.coord['y'].iloc[p[1]]], 'b-')

            plt.show()
            plt.close()

            fig = plt.figure(figsize=(15, 15))
            plt.title(self.filename + ' STEP= ' + str(s))
            plt.plot(self.coord['x'], self.coord['y'], 'r.')
            for p in alpairs:
                # print self.coord['x'].iloc[p[0]], self.coord['y'].iloc[p[0]], \
                #     self.coord['x'].iloc[p[1]], self.coord['y'].iloc[p[1]]
                plt.plot([self.coord['x'].iloc[p[0]], self.coord['x'].iloc[p[1]]],
                         [self.coord['y'].iloc[p[0]], self.coord['y'].iloc[p[1]]], 'b-')

            plt.show()
            plt.close()

            for p in lpairs:
                if p not in alpairs:
                    fig = plt.figure(figsize=(15, 15))
                    plt.title(str(p[0]) + ' DT ' + str(p[1]))

                    ax1 = fig.add_subplot(2,1,1)
                    plt.plot(data[s*step:(s+1)*step,p[0]], 'r')
                    plt.plot(data[s*step:(s+1)*step,p[1]], 'b')

                    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
                    plt.plot(dtrended[:,p[0]], 'r')
                    plt.plot(dtrended[:,p[1]], 'b')
                    plt.show()
                    plt.close()

            for p in alpairs:
                if p not in lpairs:
                    fig = plt.figure(figsize=(15, 15))
                    plt.title(str(p[0]) + ' NDT ' + str(p[1]))
                    ax1 = fig.add_subplot(2,1,1)

                    plt.plot(data[s*step:(s+1)*step,p[0]], 'r')
                    plt.plot(data[s*step:(s+1)*step,p[1]], 'b')

                    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
                    plt.plot(dtrended[:,p[0]], 'r')
                    plt.plot(dtrended[:,p[1]], 'b')

                    plt.show()
                    plt.close()

            # for p in lpairs:
            #     fig = plt.figure(figsize=(15, 15))
            #     plt.title(str(p[0]) + ' ' + str(p[1]))
            #     plt.plot(data[s*step:(s+1)*step,p[0]], 'b')
            #     plt.plot(data[s*step:(s+1)*step,p[1]], 'k')
            #     plt.plot(dtrended[:,p[0]], 'r')
            #     plt.plot(dtrended[:,p[1]], 'g')
            #     plt.show()
            #     plt.close()
            #
            # for p in alpairs:
            #     fig = plt.figure(figsize=(15, 15))
            #     plt.title(str(p[0]) + ' ' + str(p[1]))
            #     plt.plot(data[s*step:(s+1)*step,p[0]], 'b')
            #     plt.plot(data[s*step:(s+1)*step,p[1]], 'k')
            #     plt.plot(dtrended[:,p[0]], 'r')
            #     plt.plot(dtrended[:,p[1]], 'g')
            #     plt.show()
            #     plt.close()



    def data_plot(self):
        """
        Plots Graphics of the data
        :return:
        """

        for i in range(len(self.data.columns)):

            self.data[self.data.columns[i]].plot()
            data = self.data[self.data.columns[i]].values

            dataf = savgol_filter(data, 61, 3)
            plt.plot(dataf)
            dataf = savgol_filter(data, 101, 3)
            plt.plot(dataf)
            plt.show()
            plt.close()

    def compute_fft(self, begin):
        """
        FFT of signals

        :return:
        """
        self.data_fft = np.fft.rfft(self.data.iloc[begin:begin+1200,:], axis=0)

        # for i in range(self.data_fft.shape[1]):
        #     plt.plot(np.abs(self.data_fft[20:,i])**2)
        #     plt.show()
        #     plt.close()

        self.data_fft[25:,:] = 0

        inv = np.fft.irfft(self.data_fft, axis=0)

        for i in range(inv.shape[1]):
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(311)
            plt.plot(inv[:, i])
            ax = fig.add_subplot(312)
            plt.plot(self.data_fft[:, i])
            ax = fig.add_subplot(313)
            plt.plot(self.data.iloc[begin:begin+1200, i])
            plt.show()
            plt.close()


        #  self.data_fft.columns = self.data.columns
        #
        # print self.data_fft.head()
        # print self.data_fft.shape
        #
        # for i in range(len(self.data.columns)):
        #
        #     self.data_fft[self.data_fft.columns[i]].plot()
        #     plt.show()
        #     plt.close()

if __name__ == '__main__':

    for e in datafiles:
        print e
        exp = Experiment(e)

        # exp.info()
        # exp.compute_fft(3600)

       #  exp.corr_plot(0)
       #  exp.corr_plot(1200)
       #  exp.corr_plot(2400)
       #  exp.corr_plot(3600)
       #  exp.connection_plot(0.6)
        exp.connection_plot_pairs(0.6)

        # exp.data_plot()



