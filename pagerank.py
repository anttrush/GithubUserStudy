import networkx as nx
import time
USERFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\user.csv"
FOLLOWFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\follow.csv"
MLDATAFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\mldata.csv"
t = []

class User():
    def __init__(self, userId, followNum, reportIssueNum, closeIssueNum, followtoNum, inProjNum, inProjStar, prNum):
        self.userId = userId
        self.followNum = followNum or 0
        self.reportIssueNum = reportIssueNum or 0
        self.closeIssueNum = closeIssueNum or 0
        self.followtoNum = followtoNum or 0
        self.inProjNum = inProjNum or 0
        self.inProjStar = inProjStar or 0
        self.prNum = prNum or 0
    def __repr__(self):
        return str(self.userId)+','+str(self.followNum)+','+str(self.reportIssueNum)+','+str(self.closeIssueNum)+','+str(self.followtoNum)+','+str(self.inProjNum)+','+str(self.inProjStar)+','+str(self.prNum)

def getdata(): # filter user whose followNum < fThreshold
    fThreshold = 5
    print("get data from .csv file...")
    userdata = []
    userset = set()
    with open(USERFILEDIR,'r') as userfile:
        field = userfile.readline().split(',')
        for line in userfile.readlines():
            uargs = list(map(int, line.split(',')))
            if uargs[1] >= fThreshold: # followNum
                userdata.append(User(*uargs))
                userset.add(uargs[0]) # userId
    followdata = []
    with open(FOLLOWFILEDIR,'r') as followfile:
        for line in followfile.readlines():
            peer = [int(line.split(',')[0]), int(line.split(',')[1])]
            if peer[0] in userset and peer[1] in userset:
                followdata.append(peer)
    print("getting data done.")
    t.append(time.clock())
    print(t[-1]-t[-2])
    return userdata, followdata

def pagerank():
    DG = nx.DiGraph()
    userdata, followdata = getdata()
    t.append(time.clock())
    print("build up DiGraph...")
    DG.add_edges_from(followdata)
    t.append(time.clock())
    print(t[-1]-t[-2])
    print("get PageRank list...")
    pr = nx.pagerank(DG, alpha=0.85)
    t.append(time.clock())
    print(t[-1]-t[-2])
    print("fielter and write to file...")
    with open(MLDATAFILEDIR,'w') as mlfile:
        mlfile.write("userId, followNum, reportIssueNum, closeIssueNum, followtoNum, inProjNum, inProjStar, prNum, prvalue\n")
        for u in userdata:
            if u.userId in pr.keys():
                mlfile.write(repr(u)+",%.17f" % pr[u.userId]+"\n")
    t.append(time.clock())
    print(t[-1]-t[-2])
    # layout = nx.spring_layout(DG)
    # plt.figure(1)
    # nx.draw(DG, pos=layout, node_size=[x*6000 for x in pr.values()], node_color="m", with_labels=True)
    # plt.show()
    # print(len(userDict), len(prSortedList))
    return pr

t.append(time.clock())
print("begin:")
pagerank()
print("PR done!")
t.append(time.clock())
print(t[-1]-t[-2])
print(t[-1]-t[0])