import os

filelist = ['USA-road-d.NY.gr', 'USA-road-d.BAY.gr', 'USA-road-d.COL.gr',
        'USA-road-d.FLA.gr', 'USA-road-d.NW.gr', 'USA-road-d.NE.gr',
        'USA-road-d.CAL.gr', 'USA-road-d.LKS.gr', 'USA-road-d.E.gr',
        'USA-road-d.W.gr', 'USA-road-d.CTR.gr', 'USA-road-d.USA.gr']

print "Test Shortest Path Alogrithm Script"
print "Choose one method to exectue:"
print "0) all"
print "1) cuda-dijkstra"
print "2) cuda-bellman"
print "3) cuda-delta-stepping"
print "4) cuda-csr-bellman-scalar"
method = raw_input("method(0~4):")

resPathSet = ['dijkstra', 'crauser', 'delta-stepping', 'bellman-ford']
root = "d:\\graph\\"


if method == '0':
    for i in range(len(resPathSet)):
        print resPathSet[i]

        if os.path.exists(resPathSet[i]) == False:
            os.mkdir(resPathSet[i])

        cmd = "ShortestPath.exe " + str(i + 1) + " "
        
        for f in filelist:
            if os.path.exists(root + f):
                print "Processing " + f
                output = " > " + resPathSet[i] + "\\" + f + ".res"
                os.system(cmd + root + f + output)
            else:
                print f + " not exists"
        print
else:
    resPath = ''
    if method == '1':
        resPath = 'sequential'
    elif method == '2':
        resPath = 'crauser'
    elif method == '3':
        resPath = 'cuda-delta-stepping'
    elif method == '4':
        resPath = 'bellman-ford'

    cmd = "ShortestPath.exe " + method + " "

    if os.path.exists(resPath) == False:
        os.mkdir(resPath)

    for f in filelist:
        if os.path.exists(root + f):
            print "Processing " + f
            output = " > " + resPath + "\\" + f + ".res"
            os.system(cmd + root + f + output)
        else:
            print f + " not exists"

print 'Done!'
