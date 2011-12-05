import os
from pyExcelerator import *

fileFormat = '.res'

graph = {}

filelist = ['USA-road-d.NY.gr.res', 'USA-road-d.BAY.gr.res', 'USA-road-d.COL.gr.res',
        'USA-road-d.FLA.gr.res', 'USA-road-d.NW.gr.res', 'USA-road-d.NE.gr.res']


for root, dirs, files in os.walk(os.curdir):
    for dir in dirs:
        files = os.listdir(dir)
        dirname = os.getcwd() + "\\" + dir;

        print dir

        for file in filelist:
            try:
                print file
                fid = open(os.path.join(root, dir, file), 'r')
                print os.path.join(root, dir, file)
                
                if os.path.splitext(file)[1] == fileFormat:
                    line = ''
                    while True:
                        line = fid.readline()
                        if line.find('Time') != -1:
                            break
                    beg = line.find(':') + 2
                    end = line.find('secs')
                    graph[(dir, os.path.splitext(file)[0])] = line[beg:end]
                
            except IOError:
                pass

w = Workbook()

ws = w.add_sheet("RunningTime")

method = set()
#obj = set()

for m, o in graph.keys():
    method.add(m)
    #obj.add(o)

j = 0

for m in method:
    j = j + 1
    i = 0
    for o in filelist:
        i = i + 1
        ws.write(i, j, float(graph[(m, os.path.splitext(o)[0])]))

j = 0
for m in method:
    j = j + 1
    ws.write(0, j, m)

i = 0
for o in filelist:
    i = i + 1
    ws.write(i, 0, os.path.splitext(os.path.splitext(os.path.splitext(o)[0])[0])[1][1:])

w.save("RunningTime.xls")





            
