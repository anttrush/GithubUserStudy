from neo4j.v1 import GraphDatabase
from functools import reduce
USERFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\user.csv"
FOLLOWFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\follow.csv"

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

def getuser():
    uri = "bolt://10.1.1.4:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "buaaxzl"))
    # idlist = [] # 18944566
    userlist = []

    with driver.session() as session:
        id = 1
        step = 500 # 500
        rounds = 37889 # 37889
        for r in range(rounds):
            with session.begin_transaction() as tx:
                idlist = [str(x) for x in range(id+r*step, id+(r+1)*step)]
                idlist = reduce(lambda x,y:x+','+y, map(str, idlist))
                records = tx.run("start n=node(%s) MATCH (n:User) RETURN n.userId, n.followNum, n.reportIssueNum, n.closeIssueNum, n.followtoNum, n.inProjNum, n.inProjStar, n.prNum" % idlist)
                print("round %d\tnode %d" %(r, id+(r+1)*step))
                for record in records:
                    userlist.append(User(record["n.userId"],record["n.followNum"],record["n.reportIssueNum"],record["n.closeIssueNum"],record["n.followtoNum"],record["n.inProjNum"],record["n.inProjStar"],record["n.prNum"]))
    return userlist

def storeuser(userlist):
    with open(USERFILEDIR,'w') as csvfile:
        csvfile.write("userId, followNum, reportIssueNum, closeIssueNum, followtoNum, inProjNum, inProjStar, prNum\n")
        for u in userlist:
            csvfile.write(repr(u)+"\n")

def getfollow():
    print("begin to collect follow info:")
    uri = "bolt://10.1.1.4:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "buaaxzl"))

    followdata = []
    with driver.session() as session:
        with session.begin_transaction() as tx:
            records = tx.run(
                "Match (n:User)-[r1:Follow]->(m:User) return n.userId, m.userId")
            tx.sync()
            print("follow info collection done")
            followers = []
            for record in records:
                followdata.append([record["n.userId"], record["m.userId"]])
            print("follow info transformation done")
    return followdata

def storefollow(followdata):
    print("store to follow info")
    with open(FOLLOWFILEDIR,'w') as followfile:
        for peer in followdata:
            followfile.write(str(peer[0]) + ',' + str(peer[1]) + '\n')

userdata = getuser()
storeuser(userdata)
# followdata = getfollow()
# storefollow(followdata)