import os
from GAN_tools import generate_pic_with_label,generate_pic
import time
import numpy as np
import matplotlib.pyplot as plt

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
class recoder():
    def __init__(self,name,models,update_func,epoch=1_0000,save_span=100,labels=None,isUselabel=False,plot_values={}):
        self.name=name
        self.models=models
        self.update=update_func
        self.epoch=epoch
        self.save_span=save_span
        self.isUselabel=isUselabel
        self.hist=[]
        self.graph_label=labels
        self.plot_value=plot_values
        makedir("GAN_data/{0}".format(name))
        os.chdir("GAN_data/{0}".format(name))
        makedir("models")
        makedir("pics")
        makedir("hist")
        makedir("hist_data")
        self.generator=self.get_generator()
    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            print("keyboad interrupted")
            self.save("final")
            self.save_history()
    def _run(self):
        self.start_time=time.perf_counter()
        for i in range(1,self.epoch+1):
            print("{0}/{1}".format(i,self.epoch))
            data=self.update()
            self.hist.append(data)
            print(data)
            if i%self.save_span==1:
                passed=time.perf_counter()-self.start_time
                print("Passed {:.1f}s after started".format(passed))
                expEndTime=(float(self.epoch)/i-1)*passed
                print("Excepted end time is {:.1f}s after".format(expEndTime))
                version=i//self.save_span
                self.save(version)
        self.save("final")
        self.save_history()
    def save_history(self):
        self.adjust_hist()
        for i,j in enumerate(self.adjusted):
            if self.graph_label is not None:
                label=self.graph_label[i]
            else:
                label=i
            np.save("hist_data/{}".format(label),j)
        for i,j in enumerate(self.adjusted):
            if self.graph_label is not None:
                label=self.graph_label[i]
                path=label
                title=label
            else:
                path=i
                label=None
                title=None
            try:
                self.create_fig(j,path="hist/{}".format(path),labels=label,title=title)
            except Exception as e:
                print("cannnot save file",title)
                print(e)
        for title,ls in self.plot_value.items():
            x=[self.adjusted[self.graph_label.index(i)] for i  in ls]
            self.create_fig(x,"hist/{}".format(title),labels=ls,title=title)
    def adjust_hist(self):
        hist=[np.asarray([x[i] for x in self.hist]) for i in range(len(self.hist[0]))]
        self.adjusted=self.flatten(hist)
    def flatten(self,a):
        hist = []
        for i in a:
            x = i.reshape(i.shape[0], -1)
            for j in range(x.shape[1]):
                hist.append(x[:, j])
        return hist
    def create_fig(self,x,path,labels=None,title=None):
        plt.clf()
        if type(x) is not list:
            x=[x]
        if type(labels) is not list:
            labels=[labels]
        if title is not None:
            plt.title(title)
        if labels is None:
            for i,arr in enumerate(x):
                plt.plot(arr)
        else:
            for i,arr in enumerate(x):
                plt.plot(arr,label=labels[i])
            plt.legend()
        plt.savefig(path+".png")
    def save(self,version):
        self.model_save(version)
        self.save_pic(version)
    def model_save(self,version):
        for model in self.models:
            model.save("models/{0}_v{1}.h5".format(model.name,version))
    def get_generator(self):
        for i in self.models:
            if i.name=="generator":
                return i
        raise Exception("cannot find generator")
    def save_pic(self,version):
        if self.isUselabel:
            generate_pic_with_label(self.generator,filepath="pics",filename=version,isUsePicFolder=False)
        else:
            generate_pic(self.generator,filepath="pics",filename=version,isUsePicFolder=False)

