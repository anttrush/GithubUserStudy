from neo4j.v1 import GraphDatabase
import pandas as pd
from matplotlib import pyplot as plt

def fenbu():
    uri = "bolt://10.1.1.4:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "buaaxzl"))
    fns = []
    # uid = []

    with driver.session() as session:
        with session.begin_transaction() as tx:
            reports = tx.run("Match (n:User) where n.followNum >= 1 return n.userId, n.followNum")

            for rep in reports:
                # uid.append(rep["n.userId"])
                fns.append(rep["n.followNum"])
            # uid = pd.Series(uid)
            fns = pd.Series(fns)
            # df = pd.DataFrame(uid,fns)
            fns = fns.sort_values()
            print(fns.describe(percentiles=[.25, .5, .75, .9, .99, .999, .9999, .99999]))
            ## print(df.describe())
            #xx = pd.Series([x+1 for x in range(len(fns))])
            #plt.figure()
            #plt.plot(xx, fns.reshape(-1), 'bo')
            #plt.show()
def fenbu2():
    dir = r'D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\mldata.csv'
    print("getting data...")
    data = pd.read_csv(dir)['prvalue']
    data = pd.Series(data)
    print(data.describe())
    data.sort_values(inplace=True)
    map = [1.0e-07,1.0e-06,1.0e-05,1.0e-04,1.0e-03,1.0e-02]
    for threshold in map:
        print("> %f: %d" % (threshold, data[data > threshold].count()))
    # print(data)


# fenbu()
fenbu2()